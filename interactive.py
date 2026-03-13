"""
Interactive Digital Twin Demo
Usage:
  - Click a UE (circle) to select it (turns gold)
  - Click an AP (triangle) to toggle its connection with the selected UE
  - Power is recomputed automatically (average allocation)
  - GNN predictions update in real time
  - Use Prev / Next to browse test samples
  - Drag the SNR slider to change noise level
"""
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

from process import (AP_num_test, UE_num_test, loc_test_norm,
                     A_test, P_test, signal_test, interf_test, rate_test,
                     signal_mean, signal_std, interf_mean, interf_std,
                     rate_mean, rate_std)
from signal_map import SignalModel
from interf_map import InterfModel
from rate_map import RateModel
from generate import loc_mean, loc_std, cf_test, UE_test
from main import BS_loc

# ─────────────────────────── palette ───────────────────────────
BG    = '#1a1a2e'
AX    = '#0f0f1f'
GRID  = '#2a2a4a'
C_AP  = '#F65314'
C_UE  = '#7CBB00'
C_SEL = '#FFD700'
C_OFF = '#555577'
C_EDGE= '#4a9eff'
C_TRUE= '#FF6E40'
C_PRED= '#40C4FF'


class InteractiveDigitalTwin:

    def __init__(self, test_idx: int = 0, snr: int = 15):
        self.snr = snr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[demo] device: {self.device}')

        self._load_models()
        self._load_sample(test_idx)
        self._setup_figure()
        self.predict()
        self.draw()

    # ────────────────────────── model loading ──────────────────────────

    def _load_models(self):
        self.signal_model = SignalModel().to(self.device)
        p = 'model/signal_map.pth'
        if os.path.exists(p):
            self.signal_model.load_state_dict(torch.load(p, weights_only=True))
        self.signal_model.eval()

        self.interf_model = InterfModel().to(self.device)
        p = 'model/interf_map.pth'
        if os.path.exists(p):
            self.interf_model.load_state_dict(torch.load(p, weights_only=True))
        self.interf_model.eval()

        self._reload_rate_model()

    def _reload_rate_model(self):
        """Load rate model for the current SNR."""
        self.rate_model = RateModel(snr=self.snr).to(self.device)
        p = f'model/rate_map_{self.snr}dB.pth'
        if os.path.exists(p):
            self.rate_model.load_state_dict(torch.load(p, weights_only=True))
        self.rate_model.eval()

    # ────────────────────────── sample loading ─────────────────────────

    def _load_sample(self, test_idx: int):
        self.test_idx = test_idx
        self.ap_num   = int(AP_num_test[test_idx].item())
        self.ue_num   = int(UE_num_test[test_idx].item())

        # Normalised location tensor (stays on device for inference)
        self.loc_norm = loc_test_norm[test_idx].to(self.device)

        # Denormalise for plotting
        xy = loc_test_norm[test_idx].numpy() * loc_std + loc_mean
        self.ap_loc = xy[:self.ap_num]                             # (L, 2)
        self.ue_loc = xy[self.ap_num: self.ap_num + self.ue_num]  # (K, 2)

        # Find original AP/UE indices for true-value recomputation
        self.ap_orig_idx = np.array([
            np.argmin(np.sum((BS_loc[:, :2] - loc) ** 2, axis=1))
            for loc in self.ap_loc
        ])
        ue_test_2d = UE_test[:, :2]
        self.ue_orig_idx = np.array([
            np.argmin(np.sum((ue_test_2d - loc) ** 2, axis=1))
            for loc in self.ue_loc
        ])
        cf_test.K = self.ue_num
        cf_test.loc_select = UE_test[self.ue_orig_idx, :2]
        cf_test.CSI_select = cf_test.CSI[:, self.ue_orig_idx, :]

        # Association and power matrices (mutable)
        self.A = (A_test[test_idx, :self.ap_num, :self.ue_num]
                  .numpy().copy().astype(np.float32))
        self.P = (P_test[test_idx, :self.ap_num, :self.ue_num]
                  .numpy().copy().astype(np.float32))

        # Ground-truth labels
        self.true_signal = signal_test[test_idx, :self.ue_num].numpy()
        self.true_interf = interf_test[test_idx, :self.ue_num].numpy()
        self.true_rate   = rate_test[self.snr // 5, test_idx, :self.ue_num].numpy()

        # Prediction placeholders
        self.pred_signal = np.zeros(self.ue_num)
        self.pred_interf = np.zeros(self.ue_num)
        self.pred_rate   = np.zeros(self.ue_num)

        self.selected_ue = None
        self.raw_weights = self.P.copy()  # unnormalized power weights, init from actual P
        self.hovered_edge = None

    # ────────────────────────── helpers ────────────────────────────────

    def _recompute_true(self):
        """Recompute ground-truth values from the current A, P using the real channel."""
        A_full = np.zeros((cf_test.L, self.ue_num))
        P_full = np.zeros((cf_test.L, self.ue_num))
        for i, orig_ap in enumerate(self.ap_orig_idx):
            A_full[orig_ap, :] = self.A[i, :]
            P_full[orig_ap, :] = self.P[i, :]
        signal, interf, rate = cf_test.calculate_sinr(A_full, P_full)
        # unserved UEs: signal=-inf dBW (log10(0)), interf stays real, rate=0
        unserved = np.where(A_full.sum(axis=0) < 0.5)[0]
        rate[:, unserved] = 0.0
        self.true_signal = signal
        self.true_interf = interf
        self.true_rate   = rate[self.snr // 5]

    def _renormalize_power(self):
        """Normalize raw_weights per AP row to produce self.P."""
        P = np.zeros_like(self.A, dtype=np.float32)
        for l in range(self.ap_num):
            w = self.raw_weights[l] * self.A[l]
            total = w.sum()
            if total > 0:
                P[l] = w / total
        self.P = P

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            return np.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len2))
        return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def _nearest_edge(self, x, y):
        """Return (l, k, distance) of the nearest active AP-UE edge."""
        best_d, best_l, best_k = float('inf'), None, None
        for l in range(self.ap_num):
            for k in range(self.ue_num):
                if self.A[l, k] > 0.5:
                    x1, y1 = self.ap_loc[l]
                    x2, y2 = self.ue_loc[k]
                    d = self._point_to_segment_dist(x, y, x1, y1, x2, y2)
                    if d < best_d:
                        best_d, best_l, best_k = d, l, k
        return best_l, best_k, best_d

    def _pad(self, arr: np.ndarray, dtype) -> torch.Tensor:
        """Zero-pad arr to (12, 30) and move to device."""
        full = np.zeros((12, 30), dtype=dtype)
        full[:self.ap_num, :self.ue_num] = arr
        return torch.from_numpy(full).to(self.device)

    # ────────────────────────── inference ──────────────────────────────

    def predict(self):
        A_t = self._pad(self.A.astype(np.int32),   np.int32)
        P_t = self._pad(self.P.astype(np.float32), np.float32)
        loc = self.loc_norm
        L, K = self.ap_num, self.ue_num

        with torch.no_grad():
            # Signal power
            ei, ea = self.signal_model.create_edges(loc, L, K, A_t, P_t)
            norm = self.signal_model(loc, ei, ea)
            self.pred_signal = (norm * signal_std + signal_mean).cpu().numpy()

            # Interference power
            e1i, e1a, e2i, e2a = self.interf_model.create_edges(loc, L, K, A_t, P_t)
            norm = self.interf_model(loc, e1i, e1a, e2i, e2a)
            self.pred_interf = (norm * interf_std + interf_mean).cpu().numpy()

            # Achievable rate
            norm = self.rate_model(loc, L, K, A_t, P_t)
            self.pred_rate = (norm * rate_std + rate_mean).cpu().numpy()

        # unserved UEs: signal/rate are meaningless; interf model still has edge2
        unserved = np.where(self.A.sum(axis=0) < 0.5)[0]
        self.pred_signal[unserved] = 0.0
        self.pred_rate[unserved]   = 0.0

    # ────────────────────────── figure setup ───────────────────────────

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(17, 9), facecolor=BG)
        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            left=0.05, right=0.97, top=0.90, bottom=0.14,
            hspace=0.55, wspace=0.15,
            width_ratios=[1.6, 1],
        )
        self.ax_net  = self.fig.add_subplot(gs[:, 0])
        self.ax_sig  = self.fig.add_subplot(gs[0, 1])
        self.ax_int  = self.fig.add_subplot(gs[1, 1])
        self.ax_rate = self.fig.add_subplot(gs[2, 1])

        for ax in (self.ax_net, self.ax_sig, self.ax_int, self.ax_rate):
            ax.set_facecolor(AX)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)

        # Title and hint
        self.fig.text(0.5,  0.955,
                      'Radio Map-Enabled Digital Twin  —  Interactive Demo',
                      ha='center', fontsize=15, color='white', fontweight='bold')
        self.fig.text(0.27, 0.925,
                      'Click UE to select  ·  Click AP to toggle connection  '
                      '·  Scroll on edge to adjust power  '
                      '·  ←/→ prev/next  ·  R reset  ·  Esc deselect',
                      ha='center', fontsize=9, color='#8888bb')

        # ── buttons ──────────────────────────────────────────────────
        def _btn(rect, label):
            ax = plt.axes(rect, facecolor='#2d4a6e')
            b = Button(ax, label, color='#2d4a6e', hovercolor='#4a70a0')
            b.label.set_color('white')
            b.label.set_fontsize(9)
            return b

        self.btn_reset = _btn([0.050, 0.040, 0.080, 0.052], 'Reset')
        self.btn_prev  = _btn([0.138, 0.040, 0.065, 0.052], '< Prev')
        self.btn_next  = _btn([0.211, 0.040, 0.065, 0.052], 'Next >')

        self.btn_reset.on_clicked(self._on_reset)
        self.btn_prev .on_clicked(self._on_prev)
        self.btn_next .on_clicked(self._on_next)

        # ── SNR slider ───────────────────────────────────────────────
        ax_snr = plt.axes([0.55, 0.058, 0.36, 0.030], facecolor=GRID)
        self.slider_snr = Slider(ax_snr, 'SNR (dB)', 0, 30,
                                 valinit=self.snr, valstep=5, color='#4a9eff')
        self.slider_snr.label.set_color('white')
        self.slider_snr.valtext.set_color('white')
        self.slider_snr.on_changed(self._on_snr_change)

        # ── info label ───────────────────────────────────────────────
        self.info_txt = self.fig.text(0.38, 0.068, '', color='#aaaacc', fontsize=9.5)
        self._refresh_info()

        self.fig.canvas.mpl_connect('button_press_event',  self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',    self._on_key)
        self.fig.canvas.mpl_connect('scroll_event',       self._on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def _refresh_info(self):
        n_conn = int(self.A.sum()) if hasattr(self, 'A') else 0
        self.info_txt.set_text(
            f'Sample #{self.test_idx}  |  APs: {self.ap_num}  '
            f'|  UEs: {self.ue_num}  |  Connections: {n_conn}')

    # ────────────────────────── drawing ────────────────────────────────

    def draw(self):
        self._draw_network()
        self._draw_bars()
        self._refresh_info()
        self.fig.canvas.draw_idle()

    def _draw_network(self):
        ax = self.ax_net
        ax.cla()
        ax.set_facecolor(AX)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        # ── edges ────────────────────────────────────────────────────
        for l in range(self.ap_num):
            for k in range(self.ue_num):
                if self.A[l, k] > 0.5:
                    x1, y1 = self.ap_loc[l]
                    x2, y2 = self.ue_loc[k]
                    pw = float(self.P[l, k])
                    is_hov = (self.hovered_edge == (l, k))
                    ax.plot([x1, x2], [y1, y2],
                            color='#FFD700' if is_hov else C_EDGE,
                            linewidth=(2.5 + 3.0 * pw) if is_hov else (1.0 + 2.5 * pw),
                            alpha=0.95 if is_hov else (0.30 + 0.70 * pw),
                            zorder=2 if is_hov else 1)
                    # Power label on mid-point
                    xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(xm, ym, f'{pw:.2f}', fontsize=5.5,
                            color='#FFD700' if is_hov else '#99ccff',
                            ha='center', va='center', zorder=5,
                            bbox=dict(boxstyle='round,pad=0.1',
                                      fc=AX, alpha=0.55, ec='none'))

        # ── APs ──────────────────────────────────────────────────────
        for i, (x, y) in enumerate(self.ap_loc):
            ax.scatter(x, y, c=C_AP, marker='^', s=190, zorder=4,
                       edgecolors='white', linewidths=0.7)
            ax.text(x, y + 3.8, f'AP{i}', fontsize=7.5,
                    color='#ffaa88', ha='center', va='bottom', zorder=5)

        # ── UEs ──────────────────────────────────────────────────────
        for k in range(self.ue_num):
            x, y = self.ue_loc[k]
            if k == self.selected_ue:
                ax.scatter(x, y, c=C_SEL, marker='o', s=140, zorder=4,
                           edgecolors='white', linewidths=1.5)
                ax.scatter(x, y, c='none', marker='o', s=360, zorder=3,
                           edgecolors=C_SEL, linewidths=2)
            else:
                served = self.A[:, k].sum() > 0.5
                ax.scatter(x, y,
                           c=C_UE if served else C_OFF,
                           marker='o', s=80, zorder=4,
                           edgecolors='white', linewidths=0.5)
            ax.text(x, y - 4.8, f'{k}', fontsize=6.5,
                    color='#aaffaa', ha='center', va='top', zorder=5)

        ax.set_xlabel('X (m)', color='white', fontsize=10)
        ax.set_ylabel('Y (m)', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        ax.invert_xaxis()
        ax.set_title('Cell-Free Network Topology', color='white', fontsize=11, pad=7)
        ax.set_aspect('equal')
        ax.grid(True, ls=':', color=GRID, alpha=0.4)

        handles = [
            mpatches.Patch(color=C_AP,  label='AP'),
            mpatches.Patch(color=C_UE,  label='UE'),
            mpatches.Patch(color=C_SEL, label='UE (selected)'),
        ]
        ax.legend(handles=handles, loc='upper right', fontsize=7.5,
                  facecolor='#111122', edgecolor=GRID, labelcolor='white')

    def _draw_bars(self):
        x = np.arange(self.ue_num)
        w = 0.38

        specs = [
            (self.ax_sig,  self.true_signal, self.pred_signal,
             'dBW',       'Signal Power'),
            (self.ax_int,  self.true_interf, self.pred_interf,
             'dBW',       'Interference Power'),
            (self.ax_rate, self.true_rate,   self.pred_rate,
             'bits/s/Hz', f'Achievable Rate  (SNR={self.snr} dB)'),
        ]

        for ax, true_v, pred_v, ylabel, title in specs:
            ax.cla()
            ax.set_facecolor(AX)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)

            # Build per-bar colors; highlight selected UE in gold
            true_colors = [C_SEL if (self.selected_ue is not None and i == self.selected_ue)
                           else C_TRUE for i in range(self.ue_num)]
            pred_colors = [C_SEL if (self.selected_ue is not None and i == self.selected_ue)
                           else C_PRED for i in range(self.ue_num)]

            ax.bar(x - w/2, true_v, w, label='True',      color=true_colors, alpha=0.85)
            ax.bar(x + w/2, pred_v, w, label='Predicted', color=pred_colors, alpha=0.85)

            # Dynamic y-axis: span covers both true and predicted with 10 % padding
            all_v = np.concatenate([true_v, pred_v])
            vmin, vmax = all_v.min(), all_v.max()
            margin = max((vmax - vmin) * 0.15, 0.5)
            ax.set_ylim(vmin - margin, vmax + margin)

            ax.set_ylabel(ylabel, color='white', fontsize=8)
            ax.set_title(title,  color='white', fontsize=9)
            ax.set_xlim(-0.7, self.ue_num - 0.3)
            ax.set_xticks(x)
            step = max(1, self.ue_num // 15)   # avoid label crowding
            labels = [str(i) if i % step == 0 else '' for i in range(self.ue_num)]
            ax.set_xticklabels(labels, fontsize=6.5, color='#aaaacc')
            ax.set_xlabel('UE index', color='#777799', fontsize=7)
            ax.tick_params(colors='white', labelsize=7)
            ax.grid(True, ls=':', color=GRID, alpha=0.4)

            # Manual legend entries (bar colors already used for highlighting)
            legend_handles = [
                mpatches.Patch(color=C_TRUE, label='True'),
                mpatches.Patch(color=C_PRED, label='Predicted'),
            ]
            ax.legend(handles=legend_handles, fontsize=7.5, facecolor='#111122',
                      edgecolor=GRID, labelcolor='white', loc='upper right', ncol=2,
                      handlelength=1, handletextpad=0.4)

            mae = np.mean(np.abs(pred_v - true_v))
            ax.text(0.01, 0.97, f'MAE = {mae:.3f}',
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=8, color='#ffdd66')

            # Annotate selected UE with its value
            if self.selected_ue is not None:
                k = self.selected_ue
                ax.annotate(f'{true_v[k]:.2f}',
                            xy=(k - w/2, true_v[k]), xytext=(0, 4),
                            textcoords='offset points', ha='center',
                            fontsize=6.5, color=C_SEL)
                ax.annotate(f'{pred_v[k]:.2f}',
                            xy=(k + w/2, pred_v[k]), xytext=(0, 4),
                            textcoords='offset points', ha='center',
                            fontsize=6.5, color=C_SEL)

    # ────────────────────────── events ─────────────────────────────────

    def _nearest_node(self, x, y):
        """Return ('ap'|'ue', index, sq_distance)."""
        best_d, best_t, best_i = float('inf'), None, None
        for i, (ax, ay) in enumerate(self.ap_loc):
            d = (x - ax) ** 2 + (y - ay) ** 2
            if d < best_d:
                best_d, best_t, best_i = d, 'ap', i
        for k, (ux, uy) in enumerate(self.ue_loc):
            d = (x - ux) ** 2 + (y - uy) ** 2
            if d < best_d:
                best_d, best_t, best_i = d, 'ue', k
        return best_t, best_i, best_d

    def _click_threshold(self) -> float:
        """Adaptive click threshold: 5 % of the X span, squared."""
        span = np.ptp(np.concatenate([self.ap_loc[:, 0], self.ue_loc[:, 0]]))
        return (span * 0.05) ** 2

    def _on_click(self, event):
        if event.inaxes != self.ax_net or event.button != 1:
            return
        if event.xdata is None:
            return

        ntype, nidx, dist2 = self._nearest_node(event.xdata, event.ydata)
        if dist2 > self._click_threshold():
            self.selected_ue = None
            self.draw()
            return

        if ntype == 'ue':
            self.selected_ue = nidx
            self.draw()
        elif ntype == 'ap' and self.selected_ue is not None:
            l, k = nidx, self.selected_ue
            # prevent disconnecting the last serving AP for UE k
            if self.A[l, k] > 0.5 and self.A[:, k].sum() <= 1:
                return
            self.A[l, k] = 1.0 - self.A[l, k]
            self.raw_weights[l, k] = 1.0 if self.A[l, k] > 0.5 else 0.0
            self._renormalize_power()
            self._recompute_true()
            self.predict()
            self.draw()

    def _on_reset(self, _):
        self._load_sample(self.test_idx)
        self.predict()
        self.draw()

    def _on_prev(self, _):
        if self.test_idx > 0:
            self._load_sample(self.test_idx - 1)
            self.predict()
            self.draw()

    def _on_next(self, _):
        if self.test_idx < len(AP_num_test) - 1:
            self._load_sample(self.test_idx + 1)
            self.predict()
            self.draw()

    def _on_snr_change(self, val):
        self.snr = int(val)
        # Reload fine-tuned weights for the new SNR (each SNR has its own .pth)
        self._reload_rate_model()
        # Update ground-truth rate for the new SNR
        self.true_rate = rate_test[self.snr // 5, self.test_idx, :self.ue_num].numpy()
        self.predict()
        self.draw()

    def _on_key(self, event):
        """Keyboard shortcuts:
          ←/→ or a/d  — previous / next sample
          r            — reset current sample
          Escape       — deselect UE
        """
        if event.key in ('left', 'a'):
            self._on_prev(None)
        elif event.key in ('right', 'd'):
            self._on_next(None)
        elif event.key == 'r':
            self._on_reset(None)
        elif event.key == 'escape':
            self.selected_ue = None
            self.draw()

    def _edge_threshold(self):
        """5% of X-span as hover/scroll detection radius."""
        span = np.ptp(np.concatenate([self.ap_loc[:, 0], self.ue_loc[:, 0]]))
        return span * 0.05

    def _on_scroll(self, event):
        """Scroll wheel on an edge to adjust its power weight (scroll up = increase)."""
        if event.inaxes != self.ax_net or event.xdata is None:
            return
        l, k, d = self._nearest_edge(event.xdata, event.ydata)
        if l is None or d > self._edge_threshold():
            return
        factor = 1.25 if event.button == 'up' else (1 / 1.25)
        self.raw_weights[l, k] = max(0.05, self.raw_weights[l, k] * factor)
        self._renormalize_power()
        self._recompute_true()
        self.predict()
        self.draw()

    def _on_motion(self, event):
        """Highlight the edge nearest to the cursor."""
        if event.inaxes != self.ax_net or event.xdata is None:
            new_hover = None
        else:
            l, k, d = self._nearest_edge(event.xdata, event.ydata)
            new_hover = (l, k) if (l is not None and d <= self._edge_threshold()) else None
        if new_hover != self.hovered_edge:
            self.hovered_edge = new_hover
            self._draw_network()
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == '__main__':
    demo = InteractiveDigitalTwin(test_idx=0, snr=15)
    demo.show()

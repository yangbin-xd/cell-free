"""Logic tests for components/ue_map/ue_map.js, run under QuickJS with a stub
DOM / Plotly.  Requires `pip install --user quickjs`; skipped otherwise.
"""
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
JS_PATH = os.path.join(HERE, "..", "components", "ue_map", "ue_map.js")

try:
    import quickjs
except ImportError:      # pragma: no cover
    quickjs = None

STUBS = r"""
var __posted = [], __reacts = [], __timers = [];
var window = {
  parent: { postMessage: function (m) { __posted.push(m); } },
  addEventListener: function (type, fn) { if (type === "message") window.__onmessage = fn; },
};
var document = { getElementById: function () { return gd; } };
var gd = {
  _fullLayout: null,
  addEventListener: function () {},
  getBoundingClientRect: function () { return { left: 0, top: 0 }; },
  setPointerCapture: function () {}, releasePointerCapture: function () {},
};
var Plotly = { react: function (el, data, layout) { __reacts.push({ data: data, layout: layout }); } };
var performance = { now: function () { return __now; } };
var __now = 0;
function setInterval(fn, ms) { __timers.push(fn); return __timers.length; }
function clearInterval(id) { __timers[id - 1] = null; }
"""


def _grid():
    pts = []
    for i in range(0, 50, 2):
        for j in range(0, 50, 2):
            pts.append([500.0 + i, 200.0 + j])
    return pts


def _args(**over):
    a = dict(
        epoch=1, positions=[[510.0, 210.0], [520.0, 220.0]],
        ap_loc=[[500.0, 200.0], [548.0, 248.0]],
        A=[[1, 0], [0, 1]], P=[[1.0, 0.0], [0.0, 1.0]],
        grid=_grid(), r_max=3.0, moving=False, sigma=1.0,
        tick_ms=100, emit_ms=300, selected_ue=0, selected_ap=None,
        added=[], removed=[], changed=[],
        fig={"data": [{"type": "scatter"} for _ in range(8)],
             "layout": {"annotations": [{"text": "<b>UE 0</b><br>x", "x": 0, "y": 0}]}},
        height=550,
    )
    a.update(over)
    return a


@unittest.skipUnless(quickjs, "quickjs not installed")
class UeMapJsTests(unittest.TestCase):
    def setUp(self):
        self.ctx = quickjs.Context()
        self.ctx.eval(STUBS)
        with open(JS_PATH, encoding="utf-8") as f:
            self.ctx.eval(f.read())          # syntax errors surface here

    def render(self, args):
        self.ctx.eval("window.__onmessage({data: {type: 'streamlit:render', args: %s}})" % json.dumps(args))

    def state(self):
        return json.loads(self.ctx.eval("JSON.stringify(window.__ueMapTest.state())"))

    def posted(self):
        return json.loads(self.ctx.eval("JSON.stringify(__posted)"))

    def test_component_ready_sent_on_load(self):
        self.assertEqual(self.posted()[0]["type"], "streamlit:componentReady")

    def test_render_adopts_positions_and_reacts(self):
        self.render(_args())
        st = self.state()
        self.assertEqual(st["epoch"], 1)
        self.assertEqual(st["pos"], [[510.0, 210.0], [520.0, 220.0]])
        reacts = json.loads(self.ctx.eval("JSON.stringify(__reacts)"))
        self.assertEqual(len(reacts), 1)
        data = reacts[0]["data"]
        self.assertEqual(len(data), 9)                     # 8 python traces + region
        self.assertEqual(data[0]["name"], "measured")        # region first (behind)
        self.assertEqual(data[4]["x"], [510.0, 520.0])       # UE trace shifted by one
        self.assertEqual(data[1]["x"][:2], [500.0, 510.0])   # edge AP0->UE0
        self.assertEqual(reacts[0]["layout"]["annotations"][0]["x"], 510.0)
        self.assertFalse(reacts[0]["layout"]["xaxis"]["autorange"])
        # height reported once
        self.assertTrue(any(m["type"] == "streamlit:setFrameHeight" for m in self.posted()))

    def test_same_epoch_keeps_local_positions(self):
        self.render(_args())
        self.ctx.eval("window.__ueMapTest.state().pos[0][0] = 515.0")
        self.render(_args())                                  # same epoch
        self.assertEqual(self.state()["pos"][0][0], 515.0)
        self.render(_args(epoch=2))                           # python authority
        self.assertEqual(self.state()["pos"][0][0], 510.0)

    def test_clamp_matches_python_rule(self):
        self.render(_args())
        out = json.loads(self.ctx.eval("JSON.stringify(window.__ueMapTest.clampToRegion(553.0, 248.0))"))
        # nearest grid point is (548,248): 5 m away -> pulled back to 3 m
        self.assertAlmostEqual(out[0], 551.0, places=5)
        self.assertAlmostEqual(out[1], 248.0, places=5)
        inside = json.loads(self.ctx.eval("JSON.stringify(window.__ueMapTest.clampToRegion(511.0, 211.0))"))
        self.assertEqual(inside, [511.0, 211.0])

    def test_tick_keeps_ues_in_region_and_emits_throttled(self):
        self.render(_args(moving=True))
        self.assertTrue(self.state()["timer"])
        n0 = len(self.posted())
        for i in range(30):
            self.ctx.eval("__now += 100; window.__ueMapTest.tick()")
        st = self.state()
        for p in st["pos"]:
            d = self.ctx.eval("window.__ueMapTest.nearestDist(%r, %r)" % (p[0], p[1]))
            self.assertLessEqual(d, 3.0)
        values = [m for m in self.posted()[n0:] if m["type"] == "streamlit:setComponentValue"]
        # 3 s of ticks at 300 ms min interval, ack-gated (no render between) -> 1 first + 1 per 1.5 s
        self.assertGreaterEqual(len(values), 1)
        self.assertLessEqual(len(values), 4)
        self.assertEqual(values[0]["value"]["epoch"], 1)
        self.assertTrue(values[0]["value"]["moving"])

    def test_stop_forces_final_emit(self):
        self.render(_args(moving=True))
        n0 = len(self.posted())
        self.render(_args(moving=False))
        vals = [m for m in self.posted()[n0:] if m["type"] == "streamlit:setComponentValue"]
        self.assertEqual(len(vals), 1)
        self.assertFalse(vals[0]["value"]["moving"])
        self.assertFalse(self.state()["timer"])


if __name__ == "__main__":
    unittest.main()

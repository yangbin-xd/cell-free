# generate PBS jobs for the ablation campaign (TWC revision):
# every table variant retrained at seeds 0/1/2 with NO epoch cap -- training
# ends on early stopping only (--max-epochs 1000000; early stop 20 unchanged).
#
# usage: python3 jobs/gen_jobs.py           # writes jobs/*.pbs
#        for f in jobs/*.pbs; do qsub "$f"; done
import os

ROOT = '/srv/scratch/z5380367/GNN-final'
ME = '--max-epochs 1000000'
SEEDS = (0, 1, 2)

# base, stage1 cmds (run before the prerequisite check), prereq file (None =
# no gate), stage2 cmds, completion marker. {s} = seed.
VARIANTS = [
    ('main', [], None,
     ['ablation/main_map.py --seed {s}'],
     'result/pred/pred_rate_main_seed{s}_finetune.npy'),
    ('gcn', [], None,
     ['ablation/gnn_map.py --conv gcn --seed {s}'],
     'result/pred/pred_rate_gcn_seed{s}_finetune.npy'),
    ('gat', [], None,
     ['ablation/gnn_map.py --conv gat --seed {s}'],
     'result/pred/pred_rate_gat_seed{s}_finetune.npy'),
    ('gat_edge', [], None,
     ['ablation/gnn_map.py --conv gat_edge --seed {s}'],
     'result/pred/pred_rate_gat_edge_seed{s}_finetune.npy'),
    ('sage', [], None,
     ['ablation/gnn_map.py --conv sage --seed {s}'],
     'result/pred/pred_rate_sage_seed{s}_finetune.npy'),
    ('sage_sum', [], None,
     ['ablation/gnn_map.py --conv sage --aggr sum --seed {s}'],
     'result/pred/pred_rate_sage_sum_seed{s}_finetune.npy'),
    ('h1', [], None,
     ['ablation/heads_map.py --head 1 --seed {s}'],
     'result/pred/pred_rate_h1_seed{s}_finetune.npy'),
    ('h2', [], None,
     ['ablation/heads_map.py --head 2 --seed {s}'],
     'result/pred/pred_rate_h2_seed{s}_finetune.npy'),
    ('angle', [], None,
     ['ablation/signal_angle_map.py --seed {s}',
      'ablation/interf_angle_map.py --seed {s}',
      'ablation/rate_angle_map.py --seed {s}'],
     'result/pred/pred_rate_angle_seed{s}_finetune.npy'),
    ('dist', [], None,
     ['ablation/signal_dist_map.py --seed {s}',
      'ablation/interf_dist_map.py --seed {s}',
      'ablation/rate_dist_map.py --seed {s}'],
     'result/pred/pred_rate_dist_seed{s}_finetune.npy'),
    ('power', [], None,
     ['ablation/signal_power_map.py --seed {s}',
      'ablation/interf_power_map.py --seed {s}',
      'ablation/rate_power_map.py --seed {s}'],
     'result/pred/pred_rate_power_seed{s}_finetune.npy'),
    ('end2end', [], None,
     ['ablation/end2end_map.py --seed {s}'],
     'result/pred/pred_end2end_seed{s}_mae.npy'),
    ('scratch', [], None,
     ['ablation/rate_scratch_map.py --seed {s}'],
     'result/pred/pred_rate_scratch_seed{s}_mae.npy'),
    # interference-only variants: train their interf branch first (stage 1),
    # then gate on the SAME-SEED es main signal branch before the rate stage
    ('single', ['ablation/interf_single_map.py --seed {s}'],
     'model/signal_main_seed{s}.pth',
     ['ablation/rate_single_map.py --seed {s}'],
     'result/pred/pred_rate_single_seed{s}_finetune.npy'),
    ('rep', ['ablation/interf_rep_map.py --seed {s}'],
     'model/signal_main_seed{s}.pth',
     ['ablation/rate_rep_map.py --seed {s}'],
     'result/pred/pred_rate_rep_seed{s}_finetune.npy'),
    ('nopow2', ['ablation/interf_nopow2_map.py --seed {s}'],
     'model/signal_main_seed{s}.pth',
     ['ablation/rate_nopow2_map.py --seed {s}'],
     'result/pred/pred_rate_nopow2_seed{s}_finetune.npy'),
]

TEMPLATE = '''#!/bin/bash
#PBS -N cf_{name}
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:gpu_rp6000=False
#PBS -l walltime=01:59:00
#PBS -j oe
#PBS -o {root}/logs/{name}.pbslog

set -uo pipefail
cd {root}

# PBS only copies -o back once the job ends; write to scratch so a chained run
# appends to one readable log
exec >> {root}/logs/{name}.out 2>&1

NAME={name}
MARKER="{marker}"
PREREQ="{prereq}"
SELF={root}/jobs/{name}.pbs
QSUB=$(command -v qsub || echo /opt/pbs/bin/qsub)

ATT_FILE={root}/logs/{name}.attempt
ATT=$(cat "$ATT_FILE" 2>/dev/null || echo 0)
ATT=$((ATT + 1))
echo "$ATT" > "$ATT_FILE"

echo "=== $NAME | attempt $ATT | host $(hostname) | $(date '+%F %T') ==="

if [ -f "$MARKER" ]; then
    echo "=== $NAME COMPLETE (marker already present) ==="
    exit 0
fi

# no-epoch-cap runs may need many 2h chunks; cap total GPU time regardless
if [ "$ATT" -gt 90 ]; then
    echo "=== $NAME GAVE UP after 90 attempts ==="
    exit 1
fi

module load python/3.8.15
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
# interference-only variants must fine-tune against THIS campaign's main
# signal branch, not the archived paper checkpoints
export CF_MAIN_SIGNAL_TMPL='model/signal_main_seed{{seed}}.pth'
nvidia-smi --query-gpu=name --format=csv,noheader

# probe with a real kernel rather than matching compute capability against
# torch.cuda.get_arch_list() -- an L40S reports sm_89, absent from the list,
# yet works fine
if ! python3 -c "import torch; torch.zeros(8, device='cuda').sum().item()" 2>/dev/null; then
    echo "--- GPU unusable by this torch build: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) -- requeueing ---"
    "$QSUB" -a "$(date -d '+3 minutes' +%m%d%H%M)" "$SELF"
    exit 0
fi

DEADLINE=$(($(date +%s) + 6600))

run_stage() {{
    local rem=$((DEADLINE - $(date +%s)))
    if [ "$rem" -lt 180 ]; then
        return 124
    fi
    timeout "$rem" python3 $1
}}

requeue_timeout() {{
    echo "--- soft time limit reached, checkpoint saved -- requeueing ---"
    "$QSUB" "$SELF"
    exit 0
}}

step() {{
    run_stage "$1"
    local rc=$?
    if [ $rc -eq 124 ]; then requeue_timeout; fi
    if [ $rc -ne 0 ]; then
        echo "=== $NAME FAILED (exit $rc in: $1) ==="
        exit $rc
    fi
}}

{stage1}
# Wait on prerequisites without occupying the GPU: requeue 30 minutes out.
for p in $PREREQ; do
    if [ ! -f "$p" ]; then
        echo "--- prerequisite missing: $p -- requeueing in 30 min ---"
        "$QSUB" -a "$(date -d '+30 minutes' +%m%d%H%M)" "$SELF"
        exit 0
    fi
done

{stage2}
if [ -f "$MARKER" ]; then
    echo "=== $NAME COMPLETE $(date '+%F %T') ==="
    exit 0
fi

echo "=== $NAME FAILED (marker $MARKER not written) ==="
exit 1
'''


def steps(cmds, seed):
    return '\n'.join(f'step "{c.format(s=seed)} {ME}"' for c in cmds)


if __name__ == '__main__':
    os.chdir(ROOT)
    written = []
    for base, stage1, prereq, stage2, marker in VARIANTS:
        for s in SEEDS:
            name = f'{base}_s{s}'
            body = TEMPLATE.format(
                root=ROOT, name=name,
                marker=marker.format(s=s),
                prereq=(prereq or '').format(s=s),
                stage1=steps(stage1, s),
                stage2=steps(stage2, s))
            path = f'jobs/{name}.pbs'
            with open(path, 'w') as f:
                f.write(body)
            written.append(path)
    print(f'{len(written)} job files written:')
    for p in written:
        print(' ', p)

import argparse
import subprocess
import sys
from common import ROOT, assert_open
from io_utils import save, digest


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-render', action='store_true')
    args = parser.parse_args()
    steps = [
        ['stability_metrics.py', '--fits', 'baseline', '--output', 'baseline_stability.json'],
        ['constrained_thumb.py', '--output', 'constrained_axes_v3', '--steps', '150',
            '--palm-weight', '10', '--twist-weight', '30', '--preserve-palm-distance'],
        ['guard_stability.py', '--candidate', 'constrained_axes_v3', '--output', 'guarded_axes_v3'],
        ['evaluate_stability_trials.py', '--fits', 'baseline', 'guarded_axes_v3', '--prefix', 'replay'],
        ['gate_no_regression.py', '--candidate', 'guarded_axes_v3', '--prefix', 'replay',
            '--visual-review', 'VISUAL_REVIEW_RECORD.json'],
        ['select_stability.py', '--candidate', 'guarded_axes_v3', '--gate', 'replay_guarded_axes_v3_gate.json']]
    if not args.skip_render:
        steps += [['render_comparison.py', '--fit', 'selected_review', '--inputs', 'selected_inputs',
            '--hands', 'left', '--track-crop', '--output-subdir', 'closeup_comparison',
            '--speed', '.5', '--discard-frame-images'],
            ['review_videos.py', '--fit', 'selected_review', '--subdir', 'closeup_comparison']]
    sources = {path.name:digest(path) for path in ROOT.glob('*.py')}
    records = []
    for index, arguments in enumerate(steps):
        command = [sys.executable, '-B', str(ROOT / arguments[0]), *arguments[1:]]
        with (ROOT / ('replay_stage_' + str(index) + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if any(digest(ROOT / name) != value for name, value in sources.items()):
            raise RuntimeError('Source changed during replay')
        records.append(dict(command=command, completed=True))
        save(ROOT / 'pipeline_execution.json', dict(steps=records, source_sha256=sources,
            inherited_visual_review_requires_identical_candidate_hash=True,
            newly_rendered_video_visual_review_pending=not args.skip_render,
            RGB_pose_loss=False, no_manual_per_case_parameters=True))
        print('COMPLETED', arguments[0], flush=True)


if __name__ == '__main__':
    main()

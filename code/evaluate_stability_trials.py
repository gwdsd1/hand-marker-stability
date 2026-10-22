import argparse
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from common import ROOT, assert_open
from io_utils import save, digest


def main():
    assert_open()
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits', nargs='+', required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    specifications = [('evaluate_candidates.py', args.prefix + '_metrics.json', []),
        ('stability_metrics.py', args.prefix + '_stability.json', []),
        ('evaluate_finger_alignment.py', args.prefix + '_finger', ['--reference-records', 'evaluation_finger_records.json']),
        ('measure_silhouettes.py', args.prefix + '_silhouette', ['--reference', 'evaluation_reference']),
        ('evaluate_side_views.py', args.prefix + '_side', ['--reference', 'evaluation_reference'])]

    def run(specification):
        script, output, extra = specification
        command = [sys.executable, '-B', str(ROOT / script), '--fits', *args.fits, '--output', output, *extra]
        source_sha = digest(ROOT / script)
        with (ROOT / (output.replace('.json', '') + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        if digest(ROOT / script) != source_sha:
            raise RuntimeError('Evaluation source changed during execution: ' + script)
        return dict(script=script, source_sha256=source_sha, command=command, completed=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        records = list(executor.map(run, specifications))
    save(ROOT / (args.prefix + '_execution.json'), dict(records=records,
        RGB_used_only_for_offline_global_candidate_evaluation=True,
        fixed_legacy_RGB_reference='evaluation_reference', shared_observations='inputs_v2'))
    print('EVALUATED', args.fits, flush=True)


if __name__ == '__main__':
    main()

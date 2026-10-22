import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    destination = Path(args.destination).resolve()
    allowed = Path('/data2/wangzhengwei')
    if not destination.is_relative_to(allowed) or destination.is_relative_to(source) or destination.exists():
        raise ValueError('Destination must be a new sibling research directory under /data2/wangzhengwei')
    manifest = json.loads((source / 'code_manifest.json').read_text())
    required = ['baseline', 'baseline_profile', 'inputs_v2', 'wrist_articulation', 'evaluation_reference']
    for name in required:
        if not (source / name).is_dir():
            raise ValueError('Code archive alone lacks required recorded input metadata: ' + name)
    destination.mkdir()
    for name in manifest['python_files']:
        shutil.copy2(source / name, destination / name)
    for name in ['capture_config.json', 'case_config.json', 'code_manifest.json', 'README.md',
            'evaluation_finger_records.json', 'VISUAL_REVIEW_RECORD.json']:
        shutil.copy2(source / name, destination / name)
    for name in required:
        shutil.copytree(source / name, destination / name)
    (destination / 'source.json').write_text(json.dumps(dict(previous=str(source),
        purpose='Replay the marker-constrained palm and thumb stability pipeline with the same input baseline',
        raw_models_and_capture_external=True), ensure_ascii=False, indent=2) + '\n')
    print(destination, flush=True)


if __name__ == '__main__':
    main()

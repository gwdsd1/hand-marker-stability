import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'finalization.json').read_text())
    checks = {}
    for name, expected in manifest['sha256'].items():
        checks[name] = digest(root / name) == expected
    for name, expected in manifest['external_sha256'].items():
        checks[name] = digest(name) == expected
    for name in ['videos_palm_thumb_stability_20261022.zip', 'code_palm_thumb_stability_20261022.zip']:
        with zipfile.ZipFile(root / name) as archive:
            checks['ZIP:' + name] = archive.testzip() is None
    result = dict(passed=all(checks.values()), checks=len(checks), files=len(manifest['sha256']),
        dependencies=len(manifest['external_sha256']), failed=[name for name, valid in checks.items() if not valid],
        read_only=True, no_fitting_or_rendering_rerun=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

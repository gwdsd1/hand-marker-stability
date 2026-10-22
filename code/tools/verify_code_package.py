import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'package_manifest.json').read_text())
    failures = []
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append('missing_or_invalid:' + name)
        elif path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            failures.append('content:' + name)
    source_manifest = json.loads((root / 'code_manifest.json').read_text())
    expected_modules = {Path(name).stem for name in source_manifest['python_files']}
    local_modules = {path.stem for path in root.glob('*.py')}
    for name in source_manifest['python_files']:
        try:
            tree = ast.parse((root / name).read_text())
        except (OSError, SyntaxError) as error:
            failures.append('syntax:' + name + ':' + str(error))
            continue
        for node in ast.walk(tree):
            modules = [entry.name for entry in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                head = module.split('.')[0]
                if head in local_modules and head not in expected_modules:
                    failures.append('local_import:' + name + ':' + head)
    for entry in source_manifest['subprocess_entry_points']:
        if entry not in expected_modules:
            failures.append('entry_point:' + entry)
    provenance = json.loads((root / 'provenance.json').read_text())
    for name, record in provenance['copied_files'].items():
        if not (root / name).is_file() or digest(root / name) != record['source_sha256']:
            failures.append('provenance:' + name)
        if record['frozen_sha256'] != record['source_sha256']:
            failures.append('frozen_source:' + name)
    result = dict(passed=not failures, checked_files=len(manifest['files']),
        pipeline_python_files=len(source_manifest['python_files']),
        failures=failures, fitting_or_rendering_rerun=False, read_only=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

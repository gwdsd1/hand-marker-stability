import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
if not ROOT.is_relative_to(Path('/data2/wangzhengwei')):
    raise RuntimeError('All outputs and runtime files must remain under /data2/wangzhengwei')
for folder in ['tmp', 'cache', 'matplotlib', 'config', 'torch']:
    (ROOT / '.runtime' / folder).mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(ROOT / '.runtime/tmp'), XDG_CACHE_HOME=str(ROOT / '.runtime/cache'),
    XDG_CONFIG_HOME=str(ROOT / '.runtime/config'), MPLCONFIGDIR=str(ROOT / '.runtime/matplotlib'),
    TORCH_HOME=str(ROOT / '.runtime/torch'), PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
tempfile.tempdir = str(ROOT / '.runtime/tmp')
import numpy as np
import torch
torch.set_num_threads(2)


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def save(name, value):
    target = (ROOT / name).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('Output outside research copy')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), mean=float(values.mean()), p95=float(np.percentile(values, 95)), maximum=float(values.max())) if values.size else dict(count=0)


def assert_open():
    for folder in [ROOT,*ROOT.parents]:
        if (folder/'finalization.json').is_file():
            raise RuntimeError('Completed experiment or parent snapshot: create a new research copy instead of rerunning')
        if folder==Path('/data2/wangzhengwei'):break


def case_starts():
    path=ROOT/'case_config.json'
    return json.loads(path.read_text())['clip_starts_seconds'] if path.exists() else [19.,28.5,30.8]

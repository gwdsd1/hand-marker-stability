import hashlib
import json
from pathlib import Path
import numpy as np


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), median=float(np.median(values)), p95=float(np.quantile(values, .95)), maximum=float(values.max())) if values.size else dict(count=0)


def digest(path):
    checksum = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()

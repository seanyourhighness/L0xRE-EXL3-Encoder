"""Same-shape copy-and-patch only. Existing artifacts are never overwritten."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from gguf import GGUFReader


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()


def export(source, target, replacements, expected):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target.exists():
        raise ValueError('Export requires a new destination, separate from source')
    baseline_sha = sha(source)
    if baseline_sha != expected:
        raise ValueError('Source digest differs from pinned control')
    reader = GGUFReader(str(source))
    tensors = {t.name: t for t in reader.tensors}
    patches = {}
    for name, path in replacements.items():
        t = tensors[name]
        value = np.load(path, allow_pickle=False)
        if value.shape != t.data.shape or value.dtype != t.data.dtype or value.nbytes != t.n_bytes:
            raise ValueError('Replacement shape/dtype/size differs: ' + name)
        if value.dtype.kind == 'f' and not np.isfinite(value).all():
            raise ValueError('Nonfinite replacement: ' + name)
        if name.endswith('.lowgpu_zps') and not (value <= 7).all():
            raise ValueError('Illegal 3-bit zero points')
        if name.endswith('.lowgpu_scales') and not (value > 0).all():
            raise ValueError('Nonpositive vocabulary scale')
        patches[name] = value
    fd, temp = tempfile.mkstemp(prefix=target.name + '.', suffix='.partial', dir=target.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temp)
        with open(temp, 'r+b') as out:
            for name, value in patches.items():
                out.seek(tensors[name].data_offset)
                out.write(value.tobytes(order='C'))
            out.flush()
            os.fsync(out.fileno())
        changed = GGUFReader(temp)
        rows = []
        assert Path(temp).stat().st_size == source.stat().st_size
        assert len(changed.tensors) == len(reader.tensors)
        for old, new in zip(reader.tensors, changed.tensors):
            assert old.name == new.name and np.array_equal(old.shape, new.shape)
            assert old.tensor_type == new.tensor_type and old.data_offset == new.data_offset
            original, actual = tensor_sha(old), tensor_sha(new)
            if old.name in patches:
                assert actual == hashlib.sha256(patches[old.name].tobytes()).hexdigest()
            else:
                assert actual == original
            rows.append(dict(name=old.name, shape=old.shape.tolist(), type=int(old.tensor_type),
                             bytes=old.n_bytes, offset=old.data_offset, before_sha256=original,
                             after_sha256=actual, changed=actual != original))
        output_sha = sha(temp)
        if not patches:
            assert output_sha == baseline_sha
        # Link publishes atomically without replacing a competing destination.
        os.link(temp, target)
        return dict(source=str(source), source_sha256=baseline_sha, output=str(target),
                    sha256=output_sha, bytes=target.stat().st_size, tensors=rows,
                    changed_tensor_count=sum(t['changed'] for t in rows))
    finally:
        os.unlink(temp)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--replacements', type=Path)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    result = export(a.source, a.output, {} if a.replacements is None else json.loads(a.replacements.read_text()), a.source_sha256)
    a.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}), flush=True)

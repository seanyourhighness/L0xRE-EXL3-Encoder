"""EXL3 MCG -> existing Escha v1. CPU only; no inference dependencies."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

UPSTREAM_COMMIT = '8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0'
MODEL_BYTES = 8619127680
MCG = 0xCBAC1FED


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def escha_order():
    order = np.empty(256, np.int64)
    offsets = [(9,8),(8,8),(1,8),(0,8),(9,0),(8,0),(1,0),(0,0)]
    for m in range(32):
        for s, (dr, dc) in enumerate(offsets):
            r = 2 * (m & 3) + dr
            c = 2 * ((m >> 3) & 1) + 4 * ((m >> 4) & 1) + ((m >> 2) & 1) + dc
            order[(-8*m+s) % 256] = r*16+c
    return order


def exl_order():
    return np.array([((t%4)*2+(j&1)+(8 if j&2 else 0))*16+t//4+(8 if j&4 else 0)
                     for t in range(32) for j in range(8)])


def mcg_lut():
    x = (np.arange(65536, dtype=np.uint64) * MCG) & 0xffffffff
    x = (x & 0x8fff8fff) ^ 0x3b603b60
    lo = (x & 65535).astype('<u2').view('<f2').astype(np.float64)
    hi = (x >> 16).astype('<u2').view('<f2').astype(np.float64)
    return (lo + hi).astype('<f2')


def dependency(k):
    require(k in (2, 3), 'Only integer K2/K3 are admitted')
    q = (32-8*k+np.arange(256)[:,None]*k+np.arange(16)) % (256*k)
    physical = (-32*(q//32)+q%32) % (256*k)
    dep = np.empty((256, 16), np.int16)
    dep[escha_order()] = physical
    return dep


def exl_windows(packed, k):
    p = np.ascontiguousarray(packed)
    require(p.dtype in (np.dtype('<i2'), np.dtype('<u2')) and p.shape[-1] == 16*k,
            'EXL3 payload must be int16/uint16 with 16*K words')
    p = p.reshape(-1, 16*k).view('<u2').astype(np.uint32)
    words = p[:,0::2] | (p[:,1::2] << 16)
    bits = np.unpackbits(words.astype('>u4').view(np.uint8).reshape(len(p), -1), axis=1)
    pos = ((np.arange(256)+1)[:,None]*k-16+np.arange(16)) % (256*k)
    return (bits[:,pos].astype(np.uint32) * (1 << np.arange(15,-1,-1))).sum(-1).astype(np.uint16)


def escha_windows(packed, k):
    p = np.ascontiguousarray(packed).reshape(-1,16*k)
    bits = np.unpackbits(p.view(np.uint8), axis=1, bitorder='little')
    return (bits[:,dependency(k)].astype(np.uint32) * (1 << np.arange(16))).sum(-1).astype(np.uint16)


def pack_escha(row_major_windows, k):
    """Verify every overlapping window, including the circular boundary, before packing."""
    states = np.asarray(row_major_windows)
    require(states.dtype == np.uint16 and states.shape[-1] == 256, 'Expected uint16 row-major windows')
    states = states.reshape(-1,256)
    dep = dependency(k).reshape(-1)
    physical, first = np.unique(dep, return_index=True)
    require(np.array_equal(physical, np.arange(256*k)), 'Incomplete dependency coverage')
    bits = ((states[:,:,None] >> np.arange(16, dtype=np.uint16)) & 1).astype(np.uint8).reshape(len(states),-1)
    payload = bits[:,first]
    require(np.array_equal(payload[:,dep], bits), 'Incompatible overlapping or circular trellis windows')
    return np.ascontiguousarray(np.packbits(payload,axis=1,bitorder='little')).view('<i2')


def convert_tiles(packed, k, chunk=1024):
    """Repack only; preserve the exact 16-bit codeword at every tile coordinate."""
    require(k in (2,3), 'Only K2/K3')
    packed = np.asarray(packed)
    require(packed.shape[-1] == 16*k, 'Width differs from fixed allocation')
    flat = packed.reshape(-1,16*k)
    out = np.empty_like(flat, dtype='<i2')
    for start in range(0,len(flat),chunk):
        windows = exl_windows(flat[start:start+chunk],k)
        row_major = np.empty_like(windows)
        row_major[:,exl_order()] = windows
        result = pack_escha(row_major,k)
        require(np.array_equal(escha_windows(result,k),row_major), 'Reconstruction differs after packing')
        out[start:start+chunk] = result
    return out.reshape(packed.shape)


def validate_scales(suh, svh, ic, oc):
    scales = []
    for value, size in ((suh,ic),(svh,oc)):
        value = np.asarray(value)
        require(value.dtype == np.float16 and value.shape == (size,), 'Scale dtype/shape differs')
        require(np.isfinite(value).all() and (value != 0).all(), 'Zero or nonfinite folded scale')
        scales.append(value.copy())
    # Both codecs apply diag(suh) H128 Q H128 diag(svh); no extra normalization.
    return tuple(scales)


SOURCE_SUFFIX = {
    'attn_qkv':'linear_attn.in_proj_qkv', 'attn_gate':'linear_attn.in_proj_z',
    'ssm_out':'linear_attn.out_proj', 'attn_q':'self_attn.q_proj',
    'attn_k':'self_attn.k_proj', 'attn_v':'self_attn.v_proj',
    'attn_output':'self_attn.o_proj', 'ffn_gate':'mlp.gate_proj',
    'ffn_up':'mlp.up_proj', 'ffn_down':'mlp.down_proj',
}


def inventory(reader):
    tensors = {t.name:t for t in reader.tensors}
    require(np.array_equal(tensors['escha_lut'].data.reshape(-1).view('<u2'),mcg_lut().view('<u2')),
            'Template codebook differs from MCG')
    for k in (2,3):
        require(np.array_equal(tensors[f'escha_dep_k{k}'].data.reshape(256,16),dependency(k)),
                'Template dependencies differ')
    rows = []
    for t in reader.tensors:
        if not t.name.endswith('.escha_code'):
            continue
        name = t.name[:-len('.escha_code')]
        _, layer, suffix = name.split('.')
        ic = tensors[name+'.escha_rin'].data.size
        oc = tensors[name+'.escha_rout'].data.size
        k = int(t.shape[0])//16
        require(k in (2,3) and ic%128 == oc%128 == 0, 'Unsupported template geometry')
        require(t.data.shape == (ic//16,oc//16,16*k), 'Unexpected template tile layout')
        rows.append(dict(name=name,layer=int(layer),source_key=f'model.language_model.layers.{layer}.'+SOURCE_SUFFIX[suffix],
                         ic=ic,oc=oc,k=k,bytes=int(t.n_bytes)))
    require(len(rows)==400 and {r['layer'] for r in rows}==set(range(64)), 'Expected 400 projections in 64 layers')
    require(sum(r['k']==2 for r in rows)==272 and sum(r['k']==3 for r in rows)==128, 'Allocation map changed')
    return rows

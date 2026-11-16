"""Independent reference and stored-decoder checks. Never allocates CUDA memory."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np
from gguf import GGUFReader
from adapter import (UPSTREAM_COMMIT, atomic_json, convert_tiles, dependency, escha_order,
                     escha_windows, exl_order, inventory, mcg_lut, pack_escha, require, sha, validate_scales)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    upstream=Path(__file__).resolve().parents[2]
    source=upstream/'tests/testlib/trellis.py'
    spec=importlib.util.spec_from_file_location('exl3_independent_reference',source)
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    reader=GGUFReader(a.template);tensors={t.name:t for t in reader.tensors}
    rows=inventory(reader)
    lut=mcg_lut()
    require(np.array_equal(lut.view('u2'),ref.decode(np.arange(65536),'mcg').view('u2')), 'EXL3 reference LUT mismatch')
    require(np.array_equal(exl_order(),ref.tile_perm()), 'EXL3 tile permutation mismatch')
    rng=np.random.default_rng(20261115)
    checks=[]
    for k in (2,3):
        # All-zero, all-one, walking-bit and random rings exercise boundaries and ordering.
        walking=np.zeros((256*k,16*k),dtype=np.uint16)
        walking[np.arange(256*k),np.arange(256*k)//16] = (1 << (np.arange(256*k)%16)).astype(np.uint16)
        rings=np.concatenate([walking.view(np.int16),np.zeros((1,16*k),np.int16),
                              np.full((1,16*k),-1,np.int16),ref.random_packed(128,k,rng)])
        es=convert_tiles(rings,k)
        expected=np.empty((len(rings),256),np.uint16)
        expected[:,ref.tile_perm()]=ref.unpack(rings,k)
        require(np.array_equal(escha_windows(es,k),expected), 'Independent EXL3 windows differ')
        # Independent deployed decoder: GGUF dependency table, outside adapter's dependency implementation.
        bits=np.unpackbits(es.view(np.uint8),axis=1,bitorder='little')
        dep=tensors[f'escha_dep_k{k}'].data.reshape(256,16)
        ids=(bits[:,dep].astype(np.uint32)*(np.uint32(1)<<np.arange(16,dtype=np.uint32))).sum(-1)
        require(np.array_equal(lut[ids].view('u2'),ref.decode(expected,'mcg').view('u2')), 'Stored Escha decoder differs')
        # Real retained payloads survive native pack/decode roundtrip.
        candidates=[t for t in reader.tensors if t.name.endswith('.escha_code') and int(t.shape[0])==16*k]
        real=np.concatenate([t.data.reshape(-1,16*k)[:8] for t in (candidates[0],candidates[len(candidates)//2],candidates[-1])])
        require(np.array_equal(pack_escha(escha_windows(real,k),k),real), 'Real Escha payload roundtrip differs')
        broken=expected[:1].copy();broken[0,0]^=1
        try: pack_escha(broken,k)
        except ValueError: pass
        else: raise ValueError('Invalid trellis was admitted')
        checks.append(dict(k=k,synthetic_tiles=len(rings),real_tiles=len(real),bit_exact=True))
    # Full H128 transforms, nonuniform signed scales and original-basis matmul agree.
    h=np.ones((1,1),np.float64)
    for _ in range(7):h=np.block([[h,h],[h,-h]])
    h/=np.sqrt(128)
    transform=[]
    for k in (2,3):
        packed=ref.random_packed(64,k,rng).reshape(8,8,16*k)
        es=convert_tiles(packed,k)
        suh=(rng.choice([-1,1],128)*rng.uniform(.1,2,128)).astype(np.float16)
        svh=(rng.choice([-1,1],128)*rng.uniform(.1,2,128)).astype(np.float16)
        ri,ro=validate_scales(suh,svh,128,128)
        native=lut[escha_windows(es,k)].astype(np.float64).reshape(8,8,16,16).transpose(0,2,1,3).reshape(128,128)
        dense=ri[:,None].astype(np.float64)*(h@native@h)*ro[None,:].astype(np.float64)
        exdense=ref.dequant_rotated(packed,k,'mcg').astype(np.float64)
        expected=suh[:,None].astype(np.float64)*(h@exdense@h)*svh[None,:].astype(np.float64)
        require(np.array_equal(dense,expected), 'Transformed weights differ')
        x=rng.normal(size=(3,128))
        require(np.array_equal(x@dense,x@expected), 'Projection outputs differ')
        transform.append(dict(k=k,shape=[128,128],exact=True))
    result=dict(upstream_commit=UPSTREAM_COMMIT,reference_sha256=sha(source),template=str(Path(a.template).resolve()),
                template_sha256=sha(a.template),lut_entries=65536,lut_bit_exact=True,packing=checks,transforms=transform,
                projections=len(rows),width_counts={'2':272,'3':128},cpu_compatibility_passed=True,
                gpu_native_passed=False,projection_quantized=False,layer_quantized=False,
                limitations='CPU format/transform evidence only; no CUDA encoder, runtime, KLD or throughput qualification')
    atomic_json(a.output,result)
    print({key:value for key,value in result.items() if key not in ('packing','transforms')})


if __name__=='__main__':main()

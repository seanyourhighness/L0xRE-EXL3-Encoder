"""Resize GGUF code tensors without reserializing tokenizer/metadata or changing runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import numpy as np
from gguf import GGUFReader
from common import MODEL_BYTES,atomic_json,require,sha


def tensor_hash(t):return hashlib.sha256(memoryview(t.data).cast('B')).hexdigest()


def table_positions(path):
    """Keep metadata opaque; locate only fixed-width dimension/offset records."""
    with open(path,'rb') as f:
        def u32():return struct.unpack('<I',f.read(4))[0]
        def u64():return struct.unpack('<Q',f.read(8))[0]
        def string():
            n=u64();require(n<1<<30,'Invalid GGUF string');return f.read(n)
        sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
        def skip(kind):
            if kind in sizes:f.seek(sizes[kind],1)
            elif kind==8:f.seek(u64(),1)
            elif kind==9:
                element=u32();count=u64()
                if element in sizes:f.seek(count*sizes[element],1)
                else:
                    require(element==8,'Unsupported nested GGUF metadata array')
                    for _ in range(count):skip(element)
            else:raise ValueError('Unsupported GGUF metadata type')
        require(f.read(4)==b'GGUF' and u32()==3,'Require little-endian GGUFv3')
        tensors,fields=u64(),u64()
        for _ in range(fields):string();skip(u32())
        metadata_end=f.tell();positions={}
        for _ in range(tensors):
            name=string().decode('utf-8');n=u32();dims=f.tell();f.seek(8*n,1);kind=u32();offset=f.tell();u64()
            require(name not in positions,'Duplicate GGUF tensor')
            positions[name]=(dims,n,offset,kind)
        return metadata_end,positions


def export(source,target,replacements,expected,ceiling=MODEL_BYTES):
    source=Path(source).resolve();target=Path(target).resolve()
    require(source!=target and not target.exists(),'Use a new candidate destination')
    require(sha(source)==expected,'Template digest differs')
    reader=GGUFReader(str(source));by_name={t.name:t for t in reader.tensors}
    require(set(replacements)<=set(by_name),'Unknown replacement tensor')
    metadata_end,positions=table_positions(source)
    with source.open('rb') as f:header=bytearray(f.read(reader.data_offset))
    alignment=32
    if 'general.alignment' in reader.fields:
        field=reader.fields['general.alignment'];alignment=int(field.parts[field.data[0]][0])
    require(alignment>0 and alignment&(alignment-1)==0,'Invalid alignment')
    patches={};rows=[];cursor=0
    for old in reader.tensors:
        shape=tuple(int(x) for x in old.data.shape);nbytes=int(old.n_bytes)
        if old.name in replacements:
            value=np.load(replacements[old.name],mmap_mode='r',allow_pickle=False)
            require(value.dtype==old.data.dtype and value.flags.c_contiguous,'Replacement dtype/layout differs: '+old.name)
            if old.name.endswith('.escha_code'):
                require(value.ndim==3 and value.shape[:2]==shape[:2] and value.shape[2] in (32,48),'Only K2/K3 word width can change')
            else:
                require(old.name.endswith(('.escha_rin','.escha_rout')) and value.shape==shape,'Only body codes/scales may change')
                require(np.isfinite(value).all() and (value!=0).all(),'Invalid scales')
            patches[old.name]=value;shape=value.shape;nbytes=value.nbytes
        dims,n,offset,kind=positions[old.name]
        if old.name in patches:
            require(n==len(shape),'Rank change is unsupported')
            for i,d in enumerate(reversed(shape)):struct.pack_into('<Q',header,dims+8*i,d)
        struct.pack_into('<Q',header,offset,cursor)
        rows.append(dict(name=old.name,shape=list(shape),ggml_shape=list(reversed(shape)) if old.name in patches else old.shape.tolist(),
                         bytes=nbytes,offset=reader.data_offset+cursor,changed=old.name in patches))
        cursor+=(nbytes+alignment-1)//alignment*alignment
    total=int(reader.data_offset)+cursor
    require(total<=ceiling,'Complete aligned GGUF exceeds byte ceiling')
    fd,temp=tempfile.mkstemp(dir=target.parent,prefix=target.name+'.',suffix='.partial');os.close(fd)
    try:
        with open(temp,'wb') as out,source.open('rb') as inp:
            out.write(header)
            for old,row in zip(reader.tensors,rows):
                require(out.tell()==row['offset'],'Tensor offset mismatch')
                if old.name in patches:
                    raw=memoryview(patches[old.name]).cast('B')
                    for start in range(0,len(raw),4<<20):out.write(raw[start:start+(4<<20)])
                else:
                    inp.seek(old.data_offset);left=old.n_bytes
                    while left:
                        block=inp.read(min(left,4<<20));require(bool(block),'Truncated source tensor');out.write(block);left-=len(block)
                pad=(-row['bytes'])%alignment
                # Preserve existing padding for no-op byte identity; resize only as necessary.
                inp.seek(old.data_offset+old.n_bytes);oldpad=(-old.n_bytes)%alignment
                value=inp.read(min(pad,oldpad));out.write(value+b'\0'*(pad-len(value)))
            require(out.tell()==total,'Export length differs from layout')
            out.flush();os.fsync(out.fileno())
        actual=GGUFReader(temp)
        with open(temp,'rb') as f:require(f.read(metadata_end)==header[:metadata_end],'Metadata changed')
        require(len(actual.tensors)==len(reader.tensors),'Tensor count changed')
        for old,new,row in zip(reader.tensors,actual.tensors,rows):
            require(old.name==new.name and old.tensor_type==new.tensor_type,'Tensor identity changed')
            require(new.shape.tolist()==row['ggml_shape'],'GGUF dimensions changed unexpectedly')
            require(list(new.data.shape)==row['shape'] and int(new.n_bytes)==row['bytes'],'Resized tensor layout differs')
            expected_hash=(hashlib.sha256(memoryview(patches[old.name]).cast('B')).hexdigest()
                           if old.name in patches else tensor_hash(old))
            require(tensor_hash(new)==expected_hash,'Payload verification failed: '+old.name)
            row['sha256']=expected_hash
        result=dict(source_sha256=expected,sha256=sha(temp),bytes=total,ceiling=ceiling,unused_bytes=ceiling-total,
                    tensors=rows,metadata_byte_identical=True,all_unchanged_tensors_byte_identical=True,status='exported-unqualified')
        os.link(temp,target)
        directory=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
        atomic_json(str(target)+'.receipt.json',result)
        return result
    finally:
        if os.path.exists(temp):os.unlink(temp)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--source-sha256',required=True);p.add_argument('--replacements',required=True);a=p.parse_args()
    print(export(a.source,a.output,json.loads(Path(a.replacements).read_text()),a.source_sha256)['sha256'])

"""CPU-backed BF16 staging and saved-tensor offload for upstream YaQA autograd.

Numerical forward, Fisher sampling, factor accumulation and LDLQ remain upstream.
Only parameter residency and autograd's saved-tensor storage change.
"""
from contextlib import contextmanager
import re
import torch


class Streaming:
    def __init__(self,model,device='cuda:0'):
        self.model=model;self.device=torch.device(device);self.registry={};self.handles=[];self.saved_bytes=0
        blocks=[]
        for name,module in model.named_modules():
            if re.search(r'(?:^|\.)layers\.\d+$',name) and not any(s in name for s in ('visual','vision','mtp','audio')):
                blocks.append(module)
        selected=blocks+[model.get_input_embeddings(),model.get_output_embeddings()]
        unique=[]
        for module in selected:
            if module is not None and all(module is not x for x in unique):unique.append(module)
        # Small remaining parameters (final norms etc.) get their own residency boundary.
        owned={id(p) for module in unique for p in module.parameters()}
        for module in model.modules():
            if any(id(p) not in owned for p in module.parameters(recurse=False)):
                unique.append(module);owned.update(id(p) for p in module.parameters(recurse=False))
        self.cpu={id(p):p.detach().cpu() for p in model.parameters()}
        self.buffers={id(m):{n:b.detach().cpu() for n,b in m.named_buffers(recurse=False)} for m in model.modules()}
        def enter(module,args):
            ids=[];module._escha_saved_ids=ids
            for param in module.parameters():
                cpu=self.cpu[id(param)];gpu=cpu.to(self.device)
                param.data=gpu
                ptr=gpu.untyped_storage().data_ptr();self.registry[ptr]=(cpu,gpu.storage_offset());ids.append(ptr)
            for child in module.modules():
                for name,cpu in self.buffers[id(child)].items():child._buffers[name]=cpu.to(self.device)
        def leave(module,args,result):
            for param in module.parameters():param.data=self.cpu[id(param)]
            for child in module.modules():
                for name,cpu in self.buffers[id(child)].items():child._buffers[name]=cpu
            for ptr in getattr(module,'_escha_saved_ids',[]):self.registry.pop(ptr,None)
        for module in unique:
            self.handles.append(module.register_forward_pre_hook(enter))
            self.handles.append(module.register_forward_hook(leave,always_call=True))
        def move_inputs(module,args,kwargs):
            self.saved_bytes=0
            def move(x):return x.to(self.device) if isinstance(x,torch.Tensor) else x
            return tuple(move(x) for x in args),{k:move(v) for k,v in kwargs.items()}
        self.handles.append(model.register_forward_pre_hook(move_inputs,with_kwargs=True))

    def pack(self,value):
        entry=self.registry.get(value.untyped_storage().data_ptr()) if value.device==self.device else None
        if entry is not None:
            cpu,base_offset=entry
            if value.dtype!=cpu.dtype:raise ValueError('Saved parameter view dtype differs')
            backed=cpu.as_strided(value.shape,value.stride(),cpu.storage_offset()+value.storage_offset()-base_offset)
        else:
            self.saved_bytes+=value.numel()*value.element_size()
            if self.saved_bytes>24*1024**3:raise MemoryError('Saved activations exceed the24GiB offload budget; stop before a host OOM')
            backed=value.detach().to('cpu',copy=value.device.type!='cpu')
        return backed,value.device

    def unpack(self,saved):
        value,device=saved;return value.to(device)

    @contextmanager
    def saved_tensors(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack):yield

    def close(self):
        for handle in self.handles:handle.remove()
        for param in self.model.parameters():param.data=self.cpu[id(param)]
        self.registry.clear()


def cpu_model(args,devices):
    if devices!=[0]:raise ValueError('This adapter is qualified for one leased CUDA device')
    from transformers import AutoModelForCausalLM,AutoModelForImageTextToText
    options=dict(dtype=torch.bfloat16,device_map={'':'cpu'},local_files_only=True,trust_remote_code=False,
                 attn_implementation='sdpa')
    try:model=AutoModelForCausalLM.from_pretrained(args.model_dir,**options)
    except ValueError:model=AutoModelForImageTextToText.from_pretrained(args.model_dir,**options)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

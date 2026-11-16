"""CPU-backed BF16 staging and saved-tensor offload for upstream YaQA autograd.

Numerical forward, Fisher sampling, factor accumulation and LDLQ remain upstream.
Only parameter residency and autograd's saved-tensor storage change.
"""
from contextlib import contextmanager
from contextlib import nullcontext
import re
import torch
from torch.multiprocessing.reductions import StorageWeakRef


class Streaming:
    def __init__(self,model,device='cuda:0',activation_checkpoint=False):
        self.model=model;self.device=torch.device(device);self.registry={};self.handles=[];self.saved_bytes=0
        self.saved_cache={};self.cache_hits=0;self.current_module=None
        self.recomputing=0;self.original_forwards=[]
        names={id(module):name for name,module in model.named_modules()}
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
        self.parameters={id(m):dict(m._parameters) for m in model.modules()}
        self.buffers={id(m):{n:b.detach().cpu() for n,b in m.named_buffers(recurse=False)} for m in model.modules()}
        def enter(module,args):
            self.current_module=names.get(id(module),'unknown')
            ids=[];module._escha_saved_ids=ids
            # Never mutate .data on a Parameter saved by checkpoint's internal
            # hooks. Those hooks retain the Parameter itself during recompute;
            # moving it to CPU changes saved metadata before unpack validates it.
            # Restore module bindings instead, leaving each saved CUDA tensor intact.
            staged={}
            for child in module.modules():
                for name,param in self.parameters[id(child)].items():
                    if param is None:continue
                    if id(param) not in staged:
                        cpu=self.cpu[id(param)]
                        gpu=torch.nn.Parameter(cpu.to(self.device),requires_grad=param.requires_grad)
                        staged[id(param)]=gpu
                        ptr=gpu.untyped_storage().data_ptr();self.registry[ptr]=(cpu,gpu.storage_offset());ids.append(ptr)
                    child._parameters[name]=staged[id(param)]
                for name,cpu in self.buffers[id(child)].items():child._buffers[name]=cpu.to(self.device)
        def leave(module,args,result):
            for child in module.modules():
                child._parameters.update(self.parameters[id(child)])
                for name,cpu in self.buffers[id(child)].items():child._buffers[name]=cpu
            for ptr in getattr(module,'_escha_saved_ids',[]):self.registry.pop(ptr,None)
        for module in unique:
            self.handles.append(module.register_forward_pre_hook(enter))
            self.handles.append(module.register_forward_hook(leave,always_call=True))
        if activation_checkpoint:
            from torch.utils.checkpoint import checkpoint
            for block in blocks:
                original=block.forward;self.original_forwards.append((block,original))
                @contextmanager
                def recompute(block=block):
                    self.recomputing+=1
                    try:
                        enter(block,())
                        yield
                    finally:
                        leave(block,(),None);self.recomputing-=1
                def forward(*args,_original=original,_recompute=recompute,**kwargs):
                    if not torch.is_grad_enabled():return _original(*args,**kwargs)
                    return checkpoint(_original,*args,use_reentrant=False,
                            context_fn=lambda:(nullcontext(),_recompute()),**kwargs)
                block.forward=forward
        def move_inputs(module,args,kwargs):
            self.saved_bytes=0;self.saved_cache.clear();self.cache_hits=0
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
            # A StorageWeakRef identifies the actual C++ allocation without
            # retaining its GPU memory. data_ptr alone can be reused by CUDA.
            key=(StorageWeakRef(value.untyped_storage()),value.storage_offset(),tuple(value.shape),
                 tuple(value.stride()),value.dtype,value.device,value._version)
            if key in self.saved_cache:
                self.cache_hits+=1;return self.saved_cache[key],value.device
            self.saved_bytes+=value.numel()*value.element_size()
            if self.saved_bytes>24*1024**3:
                raise MemoryError(f'Saved unique activations exceed24GiB at {self.current_module}; duplicate hits={self.cache_hits}; stop before host OOM')
            backed=value.detach().to('cpu',copy=True)
            self.saved_cache[key]=backed
        return backed,value.device

    def unpack(self,saved):
        value,device=saved;return value.to(device)

    @contextmanager
    def saved_tensors(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack):yield

    def close(self):
        for module,forward in self.original_forwards:module.forward=forward
        for handle in self.handles:handle.remove()
        for module in self.model.modules():
            module._parameters.update(self.parameters[id(module)])
            module._buffers.update(self.buffers[id(module)])
        self.registry.clear()
        self.saved_cache.clear()


def guard_collector_recompute(collector,predicate):
    """Checkpoint recomputation must not count a second calibration forward."""
    original=collector.forward
    def forward(self,kind,x,leader):
        if predicate():return
        return original(self,kind,x,leader)
    collector.forward=forward


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

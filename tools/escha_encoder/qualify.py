"""Paired DFlash timing and canonical Club150 on the frozen SM89 runtime.

Runs only after a complete quantization and successful native inference. Uses
existing fixtures, sampler settings, drafter, sandbox and task catalog.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import numpy as np
import requests
from adapter import atomic_json, require, sha


@contextmanager
def server(runtime,model,draft,folder,host,mode='off'):
    folder.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind((host,0));port=sock.getsockname()[1]
    argv=runtime['argv'].copy()
    for flag,value in [('--reasoning',mode),('--reasoning-effort','low' if mode=='on' else 'none'),
                       ('--reasoning-budget','2048' if mode=='on' else '0')]:
        argv[argv.index(flag)+1]=value
    argv+=['--model',str(model),'--spec-draft-model',str(draft),'--host',host,'--port',str(port)]
    command=['numactl','--physcpubind=0-11,24-35','--preferred=0',*argv]
    atomic_json(folder/'launch.json',dict(argv=command,model_sha256=sha(model)))
    process=None;session=requests.Session();session.trust_env=False
    with (folder/'server.log').open('wb') as log:
        try:
            process=subprocess.Popen(command,env=dict(os.environ,**runtime['environment']),stdout=log,stderr=subprocess.STDOUT)
            endpoint=f'http://{host}:{port}';deadline=time.monotonic()+300
            while True:
                require(process.poll() is None,'Owned server exited')
                try:
                    if session.get(endpoint+'/health',timeout=2).status_code==200:break
                except requests.RequestException:pass
                require(time.monotonic()<deadline,'Owned server load deadline exceeded');time.sleep(.25)
            maps=Path(f'/proc/{process.pid}/maps').read_text()
            loaded={str(Path(line.split()[-1]).resolve()) for line in maps.splitlines() if 'libggml-cuda' in line}
            require(loaded=={str(Path(runtime['environment']['GGML_BACKEND_PATH']).resolve())},'Wrong loaded backend')
            (folder/'maps.txt').write_text(maps)
            yield endpoint,session
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            session.close()
    counts=re.findall(r'L0XRE_FASTPATH_SUMMARY fallthroughs=(\d+)',(folder/'server.log').read_text(errors='replace'))
    require(bool(counts) and all(int(v)==0 for v in counts),'Missing or nonzero frozen-runtime fallback counter')


def completion(endpoint,session,case,sampling):
    body=dict(prompt=case['tokens'],cache_prompt=True,seed=case['seed'],stream=True,**sampling)
    started=time.monotonic();first=None;events=[]
    with session.post(endpoint+'/completion',json=body,stream=True,timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: '):continue
            raw=line[6:]
            if raw==b'[DONE]':break
            event=json.loads(raw);events.append(event)
            if first is None and event.get('content'):first=time.monotonic()-started
    timings=next(e['timings'] for e in reversed(events) if 'timings' in e)
    require(timings['predicted_n']==128,'Timing output count changed')
    return dict(id=case['id'],first_content_seconds=first,elapsed_seconds=time.monotonic()-started,
                timings=timings,events=events)


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--work',required=True)
    a=p.parse_args();assets=Path(a.assets);root=Path(a.work);out=root/'qualification';out.mkdir(exist_ok=True)
    lease=json.loads((assets/'leases/4090/state.json').read_text())
    require(lease['state']=='running' and lease['child_pid']==os.getpid(),'Must be guardian child')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==lease['gpu'],'Lease GPU differs')
    gates=json.loads((root/'compatibility-gates.json').read_text())
    require(gates.get('full_model_exported') and gates['layer_runtime_passed'],'Full quantization is not complete')
    runtime_file=assets/'source/speed-parity-runtime-sm89-v2.json';runtime=json.loads(runtime_file.read_text())
    require(sha(runtime_file)==gates['runtime_manifest_sha256'],'Runtime manifest changed')
    require(sha(runtime['environment']['LD_PRELOAD'])==runtime['bridge_sha256'],'Bridge changed')
    require(sha(runtime['environment']['GGML_BACKEND_PATH'])==runtime['CUDA_backend_sha256'],'Backend changed')
    require(sha(runtime['sealed_settings']['path'])==runtime['sealed_settings']['sha256'],'Settings changed')
    models=dict(baseline=assets/'artifacts/stacking-loop/B02/A2/quant.gguf',candidate=root/'full-escha.gguf')
    require(sha(models['baseline'])=='ecdd8679d7cfe023cadb3eb8726fb09a869a6f71f69c183d71144973b0785bc7','Baseline changed')
    require(sha(models['candidate'])==gates['full_development']['identity']['model_sha256'],'Candidate changed')
    draft=Path('/mnt/ssd/ai-builds/l0xre-production-v011-20261031/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf')
    require(sha(draft)=='1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd','DFlash draft changed')
    fixture_file=assets/'PERFORMANCE-HISTORY-FIXTURE-v1.json';fixture=json.loads(fixture_file.read_text())
    require(len(fixture['cases'])==20 and fixture['sampling']['n_predict']==128,'Frozen timing fixture differs')
    os.sched_setaffinity(0,set(range(12))|set(range(24,36)))
    identity=dict(models={k:sha(v) for k,v in models.items()},runtime_sha256=sha(runtime_file),
                  fixture_sha256=sha(fixture_file),script_sha256=sha(__file__),
                  task_manifest_sha256=sha(assets/'TASK-MANIFEST.json'),
                  harness_sha256=sha(assets/'tools/benchlocal099_cli_v1.py'),
                  draft_sha256=sha(draft),order=['baseline','candidate','candidate','baseline'],
                  output_tokens=128,documents=20)
    protocol=out/'protocol.json'
    if protocol.exists():require(json.loads(protocol.read_text())==identity,'Qualification resume changed')
    else:atomic_json(protocol,identity)
    runs=[]
    for index,arm in enumerate(identity['order']):
        folder=out/'speed'/f'{index}-{arm}';receipt=folder/'result.json'
        if receipt.exists():
            run=json.loads(receipt.read_text());require(run['complete'] and run['identity']==identity,'Incomplete/changed timing receipt')
        else:
            run=dict(identity=identity,arm=arm,samples=[],complete=False)
            with server(runtime,models[arm],draft,folder,'127.0.0.1') as (endpoint,session):
                for _ in range(2):completion(endpoint,session,fixture['warmup'],fixture['sampling'])
                for case in fixture['cases']:
                    for phase in ('first','reuse'):
                        sample=completion(endpoint,session,case,fixture['sampling']);sample['phase']=phase
                        run['samples'].append(sample)
                    print('DFLASH_DOCUMENT',index,arm,case['id'],flush=True)
            run['complete']=True;atomic_json(receipt,run)
        runs.append(run)
    ratios=[]
    for case in fixture['cases']:
        logs={arm:[] for arm in models}
        for run in runs:
            for sample in run['samples']:
                if sample['id']==case['id']:
                    rate=sample['timings']['predicted_per_second'];require(rate>0 and np.isfinite(rate),'Invalid throughput')
                    logs[run['arm']].append(np.log(rate))
        ratios.append(float(np.mean(logs['candidate'])-np.mean(logs['baseline'])))
    rng=np.random.default_rng(20261115);ratios=np.array(ratios)
    boot=np.exp(ratios[rng.integers(0,len(ratios),(10000,len(ratios)))].mean(1))
    speed=dict(ratio=float(np.exp(ratios.mean())),paired_document_95CI=np.quantile(boot,[.025,.975]).tolist(),
               floor=.98,passed=bool(np.quantile(boot,.025)>=.98),protocol_sha256=sha(protocol),
               interpretation='One paired ABBA block; independently paired documents. No SM86/SM120 claim.')
    atomic_json(out/'speed.json',speed)
    networks=json.loads(subprocess.check_output(['docker','network','inspect','bridge'],text=True))
    host=networks[0]['IPAM']['Config'][0]['Gateway'];require(host.startswith(('172.','192.168.','10.')),'Expected local Docker bridge')
    club=[]
    for arm,model in models.items():
        for mode in ('off','on'):
            folder=out/'club150'/(arm+'-'+mode);receipt=folder/'measurement.json'
            if receipt.exists():
                result=json.loads(receipt.read_text());require(result['identity']==identity,'Club resume changed')
                club.append(result);continue
            with server(runtime,model,draft,folder,host,mode) as (endpoint,session):
                env=dict(os.environ,URL=endpoint,MODEL='L0xRE-27b-Low',QUALITY_RECORD='0',PYTHONUTF8='1',
                         NO_PROXY=host+',127.0.0.1,localhost',no_proxy=host+',127.0.0.1,localhost')
                for key in ('LD_PRELOAD','LD_LIBRARY_PATH','OPENAI_API_KEY','BENCHLOCAL_API_KEY','BENCHLOCAL_HISTORY_FILE'):env.pop(key,None)
                command=['/usr/bin/python3',str(assets/'tools/benchlocal099_cli_v1.py'),'run','--full','--endpoint',endpoint,
                         '--model','L0xRE-27b-Low','--output','markdown','--progress',
                         '--no-thinking' if mode=='off' else '--enable-thinking','--max-tokens','4096','--thinking-max-tokens','16384',
                         '--save-json',str(folder/'result.json'),'--incremental','--sandbox-log-dir',str(folder/'sandbox-logs'),
                         '--sandbox-image-tag','l0xre-low-v2-20261114','--no-retry','--extra-body',json.dumps(dict(seed=20261112))]
                with (folder/'harness.log').open('wb') as stream:
                    completed=subprocess.run(command,cwd='/home/sean/club-3090',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=10000)
                data=json.loads((folder/'result.json').read_text());packs=data.get('packs') or []
                require(len(packs)==8 and sum(int(x.get('total') or 0) for x in packs)==150,'Incomplete canonical Club150')
                result=dict(identity=identity,arm=arm,mode=mode,passed=sum(int(x.get('passed') or 0) for x in packs),
                            total=150,harness_exit=completed.returncode,result_sha256=sha(folder/'result.json'),
                            packs=packs,thinking_validity=data.get('thinking_validity'))
            atomic_json(receipt,result);club.append(result)
            print('CLUB_COMPLETE',arm,mode,result['passed'],flush=True)
    task_pass=True
    for mode in ('off','on'):
        b=next(x for x in club if x['arm']=='baseline' and x['mode']==mode)
        c=next(x for x in club if x['arm']=='candidate' and x['mode']==mode)
        task_pass=task_pass and c['passed']>=b['passed'] and c['harness_exit']==0 and b['harness_exit']==0
        def pack_scores(result):
            return {x.get('pack_id') or x.get('id') or x.get('name'):int(x.get('passed') or 0) for x in result['packs']}
        bs,cs=pack_scores(b),pack_scores(c)
        task_pass=task_pass and bs.keys()==cs.keys() and all(cs[k]>=v for k,v in bs.items())
    gates.update(dflash_qualified=speed['passed'],coding_qualified=bool(task_pass),
                 qualification_complete=True,release_qualified=False,
                 success_on_sm89=bool(gates['below_baseline'] and speed['passed'] and task_pass),
                 qualification_scope='Z840 SM89; development116, paired DFlash ABBA and canonical Club150 ON/OFF. Independent holdout and other hardware unqualified.')
    atomic_json(out/'result.json',dict(gates=gates,speed=speed,club=club))
    atomic_json(root/'compatibility-gates.json',gates)


if __name__=='__main__':main()

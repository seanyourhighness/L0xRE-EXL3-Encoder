"""Recover the existing 4090 guardian after a host reboot, preserving its history.

The user service runs this entry point at boot while the authorized reservation
is held. It delegates all locking, service handling and queue execution to the
retained guardian. It never changes the 3060 service or the Strata configuration.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--gpu',required=True);p.add_argument('--service',required=True)
    a=p.parse_args();root=Path(a.work);assets=Path(a.assets);lease=assets/'leases/4090'
    root.mkdir(parents=True,exist_ok=True)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    old=json.loads((lease/'state.json').read_text()) if (lease/'state.json').exists() else {}
    pid=old.get('pid');cmd=Path(f'/proc/{pid}/cmdline') if pid else None
    if cmd and cmd.exists() and b'lease_worker_goal_hold_v2.py' in cmd.read_bytes():
        raise RuntimeError('A live guardian already owns this reservation; do not replace it')
    # Bind the before/after snapshot to the service's current profile, including
    # a Strata version change performed outside this quantization task.
    start=subprocess.check_output(['systemctl','show',a.service,'-p','ExecStart','--value'],text=True)
    candidates=[Path(x) for x in re.findall(r'/[^\s;{}]+\.json',start)]
    candidates=[x for x in candidates if x.is_file()]
    if len(candidates)!=1:raise RuntimeError('Cannot uniquely identify the current borrowed service profile')
    archive=root/'reservation-history'/(boot+'-'+str(time.time_ns()))
    archive.mkdir(parents=True)
    for name in ('state.json','before.json','before-goal-hold.json'):
        if (lease/name).exists():shutil.copy2(lease/name,archive/name)
    checkpoint=root/'pipeline/full/conversion/ckpt/job.json'
    event=dict(time=time.time(),boot_id=boot,previous_state=old,service=a.service,
               service_profile=str(candidates[0]),checkpoint=None if not checkpoint.exists() else json.loads(checkpoint.read_text()),
               reason='Recover the authorized Z8404090 reservation after host reboot; queued conversion resumes upstream checkpoint',
               authorization='Sean: use the 4090 on the z840 for the work',prior_attempt_history_preserved=True)
    (archive/'recovery.json').write_text(json.dumps(event,indent=2)+'\n')
    with (root/'reservation-events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    # The frozen runtime's canonical helper materializes its settings in /tmp,
    # which is cleared at boot. Recreate the exact bytes, never change settings.
    runtime=json.loads((assets/'source/speed-parity-runtime-sm89-v2.json').read_text())
    preset=Path(runtime['sealed_settings']['path'])
    if not preset.exists():
        helper=assets/'runtime/sm89/L0xRE-v0.1.3.2-linux-sm89/tools/universal/internal_preset.py'
        if hashlib.sha256(helper.read_bytes()).hexdigest()!=runtime['helper_sha256']:
            raise RuntimeError('Frozen runtime preset helper changed')
        spec=importlib.util.spec_from_file_location('frozen_l0xre_preset',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rebuilt=module.prepare(runtime['sealed_settings']['values'])
        if rebuilt['sha256']!=runtime['sealed_settings']['sha256'] or Path(rebuilt['path'])!=preset:
            raise RuntimeError('Recreated runtime preset differs from the frozen identity')
    if hashlib.sha256(preset.read_bytes()).hexdigest()!=runtime['sealed_settings']['sha256']:
        raise RuntimeError('Frozen runtime preset is corrupt')
    worker=assets/'tools/lease_worker_goal_hold_v2.py'
    command=['/usr/bin/python3','-u',str(worker),'--root',str(lease),'--gpu',a.gpu,
             '--service',a.service,'--config',str(candidates[0]),'--health','http://127.0.0.1:8081/health']
    os.execv(command[0],command)


if __name__=='__main__':main()

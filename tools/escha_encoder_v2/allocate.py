"""Use upstream SC optimization, restricted by the existing Escha dispatch/budget."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import math
from pathlib import Path
from types import SimpleNamespace
import yaml
from common import TREE,atomic_json,read,require,sha,validate_recipe


def optimize(policy,measurement,rfn,output):
    rows={r['source_key']:r for r in policy['projections']}
    require(len(measurement['results'])>=len(rows),'Incomplete sensitivity measurement')
    selected={}
    for item in measurement['results']:
        if item['key'] not in rows:continue
        require(item['key'] not in selected and 'members' not in item,'Require individual dense projections, not grouped targets')
        r=rows[item['key']]
        require(item['numel']==r['ic']*r['oc'] and item['qbits_key']=='bits','Measurement geometry/role differs')
        require(item.get('shaped') is True,'Require actual LDLQ-shaped noise, not an iid fallback')
        require(len(item['levels'])>=2,'Require at least two measured noise levels')
        for level in item['levels']:
            require(math.isfinite(level['kld']) and level['kld']>=0 and math.isfinite(level['rfn_actual']) and level['rfn_actual']>0,
                    'Invalid sensitivity measurement')
        selected[item['key']]=item
    require(set(selected)==set(rows),'Missing per-tensor sensitivity results')
    anchors={x['key']:x for x in rfn['results']}
    require(set(rows)<=set(anchors),'Missing actual MCG reconstruction anchors')
    for key,r in rows.items():
        a=anchors[key]
        require(a['K']==r['k'] and a['numel']==r['ic']*r['oc'] and math.isfinite(a['rfn']) and a['rfn']>0,'Wrong MCG RFN anchor')
    fixed={key:r['allowed_k'][0] for key,r in rows.items() if len(r['allowed_k'])==1}
    flexible=[selected[key] for key,r in rows.items() if len(r['allowed_k'])==2]
    require(bool(flexible),'No supported precision reallocations')
    fixed_bits=sum(rows[key]['ic']*rows[key]['oc']*k for key,k in fixed.items())
    budget=policy['body_bits']-fixed_bits;numel=sum(r['numel'] for r in flexible)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    filtered=dict(measurement,results=flexible)
    atomic_json(out/'flexible-measurement.json',filtered);atomic_json(out/'mcg-rfn.json',rfn)
    source=TREE/'sc_optimize.py';spec=importlib.util.spec_from_file_location('pinned_sc_optimizer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=SimpleNamespace(measurement=str(out/'flexible-measurement.json'),rfn_ref=str(out/'mcg-rfn.json'),
                         bitrate=math.nextafter(budget/numel,math.inf),head_bits=16,alpha=None,alpha_low=None,
                         bit_ratio=1.96,anchor='2:0.292',tie='',min_k=2,max_k=3,k_step=1,
                         out=str(out/'upstream-recipe.yaml'),table=str(out/'predicted-kl.json'))
    with (out/'optimizer.log').open('w') as log,redirect_stdout(log):module.main(args)
    raw=yaml.safe_load((out/'upstream-recipe.yaml').read_text())
    recipe=dict(target_bpw=policy['body_bits']/sum(r['ic']*r['oc'] for r in rows.values()),head_bits=16,
                tensors={'*':16,**fixed,**raw['tensors']})
    audit=validate_recipe(policy,recipe)
    require(bool(audit['changed_projections']),'Measured optimizer retained V1 allocation; do not label it a redistributed candidate')
    recipe['achieved_bpw']=audit['body_bits']/sum(r['ic']*r['oc'] for r in rows.values())
    atomic_json(out/'recipe.yaml',recipe)
    table=read(out/'predicted-kl.json')['tensors']
    model={x['key']:x['kld'] for x in table}
    audit.update(upstream_optimizer_sha256=sha(source),sensitivity_estimate_not_measured_model_KLD=True,
                 mcg_anchor_provenance='V1 MCG RFN probe; per-bit amplitude ratio remains upstream assumption1.96',
                 predicted_flexible_baseline=sum(model[k][str(rows[k]['k'])] for k in model),
                 predicted_flexible_candidate=sum(model[k][str(recipe['tensors'][k])] for k in model))
    require(audit['predicted_flexible_candidate']<audit['predicted_flexible_baseline'],
            'SC estimate does not improve the existing map; preserve evidence instead of forcing a worse allocation')
    atomic_json(out/'allocation-audit.json',audit)
    return recipe,audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--measurement',required=True)
    p.add_argument('--rfn',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    _,audit=optimize(read(a.contract),read(a.measurement),read(a.rfn),a.out);print(audit)

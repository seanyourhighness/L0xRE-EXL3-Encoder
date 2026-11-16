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


def optimize(policy,measurement,rfn,output,allow_unchanged=False):
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
    if not allow_unchanged:require(bool(audit['changed_projections']),'Measured optimizer retained V1 allocation; do not label it a redistributed candidate')
    recipe['achieved_bpw']=audit['body_bits']/sum(r['ic']*r['oc'] for r in rows.values())
    atomic_json(out/'recipe.yaml',recipe)
    table=read(out/'predicted-kl.json')['tensors']
    model={x['key']:x['kld'] for x in table}
    audit.update(upstream_optimizer_sha256=sha(source),sensitivity_estimate_not_measured_model_KLD=True,
                 mcg_anchor_provenance='V1 MCG RFN probe; per-bit amplitude ratio remains upstream assumption1.96',
                 predicted_flexible_baseline=sum(model[k][str(rows[k]['k'])] for k in model),
                 predicted_flexible_candidate=sum(model[k][str(recipe['tensors'][k])] for k in model))
    if not allow_unchanged:
        require(audit['predicted_flexible_candidate']<audit['predicted_flexible_baseline'],
                'SC estimate does not improve the existing map; preserve evidence instead of forcing a worse allocation')
    audit['allocation_changed']=bool(audit['changed_projections'])
    atomic_json(out/'allocation-audit.json',audit)
    return recipe,audit


def robust(policy,measurements,rfn,output,min_rank=.75,max_disagreement=.25):
    import numpy as np
    from exact_allocation import solve
    out=Path(output);out.mkdir(parents=True,exist_ok=True);tables=[];maps=[];audits=[]
    for i,measurement in enumerate(measurements):
        _,upstream=optimize(policy,measurement,rfn,out/f'replicate-{i}',allow_unchanged=True)
        table=read(out/f'replicate-{i}/predicted-kl.json');recipe,audit=solve(policy,table)
        tables.append({r['key']:r for r in table['tensors']});maps.append(recipe);audits.append(audit)
        atomic_json(out/f'replicate-{i}/exact-audit.json',dict(audit,upstream_prediction=upstream['predicted_flexible_candidate']))
    require(len(tables)==2,'Require two independent replicates')
    flexible=[r for r in policy['projections'] if len(r['allowed_k'])==2]
    def ranks(values):
        order=np.argsort(values,kind='stable');result=np.empty(len(values),float);start=0
        while start<len(values):
            end=start+1
            while end<len(values) and values[order[end]]==values[order[start]]:end+=1
            result[order[start:end]]=(start+end-1)/2;start=end
        return result
    densities=[]
    for table in tables:densities.append(np.array([(table[r['source_key']]['kld']['2']-table[r['source_key']]['kld']['3'])/(r['ic']*r['oc']) for r in flexible]))
    ra,rb=map(ranks,densities);correlation=float(np.corrcoef(ra,rb)[0,1])
    disagreement=sum(r['ic']*r['oc'] for r in flexible if maps[0]['tensors'][r['source_key']]!=maps[1]['tensors'][r['source_key']])
    denominator=sum(r['ic']*r['oc'] for r in flexible);fraction=disagreement/denominator
    stability=dict(rank_spearman=correlation,width_disagreement_weight_fraction=fraction,min_rank=min_rank,max_disagreement=max_disagreement,
                   passed=bool(np.isfinite(correlation) and correlation>=min_rank and fraction<=max_disagreement))
    atomic_json(out/'stability.json',stability)
    require(stability['passed'],'Sensitivity ranking is unstable; expand/review measurements before full factors')
    merged=dict(tensors=[])
    for key,item in tables[0].items():
        merged['tensors'].append(dict(item,kld={str(k):sum(t[key]['kld'][str(k)] for t in tables)/2 for k in (2,3)}))
    exact,audit=solve(policy,merged);check=validate_recipe(policy,exact)
    exact.update(target_bpw=policy['body_bits']/sum(r['ic']*r['oc'] for r in policy['projections']),
                 achieved_bpw=check['body_bits']/sum(r['ic']*r['oc'] for r in policy['projections']))
    for table in tables:
        base=sum(table[r['source_key']]['kld'][str(r['k'])] for r in flexible)
        candidate=sum(table[r['source_key']]['kld'][str(exact['tensors'][r['source_key']])] for r in flexible)
        require(candidate<=base+1e-12,'Mean recipe regresses one independent sensitivity replicate')
    fixed=dict(target_bpw=exact['target_bpw'],head_bits=16,tensors={'*':16,**{r['source_key']:r['k'] for r in policy['projections']}})
    atomic_json(out/'predicted-kl-mean.json',merged);atomic_json(out/'recipe.yaml',exact);atomic_json(out/'fixed-recipe.yaml',fixed)
    atomic_json(out/'allocation-audit.json',dict(audit,stability=stability,budget_check=check,solver='exact discrete additive-cost DP',
       allocation_changed=bool(check['changed_projections']),unchanged_control_permitted=True,not_measured_model_KLD=True))
    return exact,audit


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--measurement',required=True)
    p.add_argument('--rfn',required=True);p.add_argument('--out',required=True)
    p.add_argument('--replicate');a=p.parse_args()
    if a.replicate:_,audit=robust(read(a.contract),[read(a.measurement),read(a.replicate)],read(a.rfn),a.out)
    else:_,audit=optimize(read(a.contract),read(a.measurement),read(a.rfn),a.out,allow_unchanged=True)
    print(audit)

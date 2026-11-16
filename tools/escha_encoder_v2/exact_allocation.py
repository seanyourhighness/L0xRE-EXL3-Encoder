"""Offline exact K2/K3 budget audit of upstream predicted-kl.json (stdlib only).

Optimality is only for the supplied additive predicted costs, not model quality.
Does not modify the live plan, upstream optimizer, or queue.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def solve(policy, table):
    rows = policy['projections']
    keys = [r['source_key'] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate contract keys')
    costs = {}
    for item in table['tensors']:
        key = item['key']
        if key in costs:
            raise ValueError('Duplicate cost keys')
        costs[key] = item
    flexible = []
    assigned = {}
    minimum_bits = 0
    for r in rows:
        key = r['source_key']
        n = r['ic'] * r['oc']
        ks = r['allowed_k']
        if n <= 0 or ks not in ([2], [3], [2, 3]) or r['k'] not in ks:
            raise ValueError('Invalid contract geometry or widths')
        assigned[key] = min(ks)
        minimum_bits += n * min(ks)
        if len(ks) == 2:
            item = costs[key]
            if item['numel'] != n:
                raise ValueError('Cost geometry differs from contract')
            c2, c3 = (float(item['kld'][str(k)]) for k in (2, 3))
            if not all(math.isfinite(c) and c >= 0 for c in (c2, c3)):
                raise ValueError('Invalid predicted costs')
            flexible.append((key, n, c2 - c3))
    remaining = policy['body_bits'] - minimum_bits
    if remaining < 0:
        raise ValueError('Budget below minimum supported widths')
    unit = math.gcd(*(n for _, n, _ in flexible)) if flexible else 1
    capacity = min(remaining // unit, sum(n // unit for _, n, _ in flexible))
    if (capacity + 1) * max(1, len(flexible)) > 20_000_000:
        raise ValueError('DP state limit exceeded; use a separately bounded solver')
    gains = [0.0] * (capacity + 1)
    choices = [0] * (capacity + 1)
    for i, (_, n, gain) in enumerate(flexible):
        weight = n // unit
        for b in range(capacity, weight - 1, -1):
            candidate = gains[b - weight] + gain
            if candidate > gains[b]:
                gains[b] = candidate
                choices[b] = choices[b - weight] | (1 << i)
    for i, (key, _, _) in enumerate(flexible):
        if choices[capacity] & (1 << i):
            assigned[key] = 3
    spent = sum(r['ic'] * r['oc'] * assigned[r['source_key']] for r in rows)
    baseline = sum(float(costs[r['source_key']]['kld'][str(r['k'])])
                   for r in rows if len(r['allowed_k']) == 2)
    predicted = sum(float(costs[k]['kld'][str(assigned[k])]) for k, _, _ in flexible)
    return dict(head_bits=16, tensors={'*': 16, **assigned}), dict(
        objective='Exact discrete allocation for supplied additive predictions; NOT measured KLD',
        body_bits=spent, unused_body_bytes=(policy['body_bits'] - spent) // 8,
        predicted_flexible_baseline=baseline, predicted_flexible_candidate=predicted,
        changed_projections=[k for k, r in zip(keys, rows) if assigned[k] != r['k']],
        capacity_units=capacity, unit_bits=unit, flexible_count=len(flexible))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--table', required=True, help='allocation/predicted-kl.json')
    p.add_argument('--out', required=True, help='New audit directory; never overwrite a run')
    a = p.parse_args()
    policy = json.loads(Path(a.contract).read_text())
    table = json.loads(Path(a.table).read_text())
    recipe, audit = solve(policy, table)
    audit['inputs'] = {str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                       for f in (a.contract, a.table)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    # JSON is valid YAML, matching the existing recipe writer.
    (out / 'recipe.yaml').write_text(json.dumps(recipe, indent=2) + '\n')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

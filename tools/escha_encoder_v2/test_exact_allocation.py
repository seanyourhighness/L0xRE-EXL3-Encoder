import itertools
import random
import unittest
from exact_allocation import solve


class ExactAllocationTests(unittest.TestCase):
    def fixture(self, sizes, gains, baseline):
        policy = dict(body_bits=sum(n*k for n,k in zip(sizes,baseline)), projections=[
            dict(source_key=str(i), ic=n, oc=1, k=k, allowed_k=[2,3])
            for i,(n,k) in enumerate(zip(sizes,baseline))])
        table = dict(tensors=[dict(key=str(i), numel=n, kld={'2':100., '3':100.-g})
                              for i,(n,g) in enumerate(zip(sizes,gains))])
        return policy, table

    def test_beats_density_greedy_and_single_promotion_repair(self):
        # Greedy chooses the size-6 item (gain 12); two size-5 items gain 18.
        # Neither single promotion improves on 12, so local repair stalls.
        p,t = self.fixture([6,5,5],[12,9,9],[2,3,3])
        recipe,audit = solve(p,t)
        self.assertEqual([recipe['tensors'][str(i)] for i in range(3)],[2,3,3])
        self.assertEqual(audit['unused_body_bytes'],0)

    def test_matches_exhaustive_search(self):
        rng=random.Random(41)
        for _ in range(60):
            sizes=[rng.randint(1,17) for _ in range(8)]
            p,t=self.fixture(sizes,[rng.uniform(-3,30) for _ in sizes],
                             [rng.choice([2,3]) for _ in sizes])
            recipe,audit=solve(p,t)
            optimum=min(sum(t['tensors'][i]['kld'][str(k)] for i,k in enumerate(ks))
                        for ks in itertools.product((2,3),repeat=8)
                        if sum(n*k for n,k in zip(sizes,ks))<=p['body_bits'])
            self.assertAlmostEqual(audit['predicted_flexible_candidate'],optimum,places=10)
            self.assertLessEqual(audit['body_bits'],p['body_bits'])

    def test_fixed_width_and_bad_measurement(self):
        p,t=self.fixture([6,5,5],[12,9,9],[2,3,3])
        p['projections'][0]['allowed_k']=[2]
        self.assertEqual(solve(p,t)[0]['tensors']['0'],2)
        t['tensors'][1]['kld']['3']=float('nan')
        with self.assertRaises(ValueError):solve(p,t)


if __name__=='__main__':unittest.main()

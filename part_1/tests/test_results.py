import json
import hashlib
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import poisson

from part_1.stationary_monopoly.reproduce import load_factor, optimal_load_factor
from part_1.time_varying_demand.reproduce import posted_prices, gap_closed
from part_1.duopoly.reproduce import load, metrics

ROOT = Path(__file__).resolve().parents[1]


class PublishedResults(unittest.TestCase):
    def test_exponential_mapping(self):
        for n in (1, 20, 50, 100):
            for x in (.1, 5, 30):
                expected = x/n * poisson.cdf(n-1,x) / poisson.cdf(n,x)
                self.assertAlmostEqual(load_factor(n,x), expected, places=12)
            for observed in (.01,.2,.8,.99):
                self.assertGreater(optimal_load_factor(n,observed), observed)
        self.assertAlmostEqual(100*(1-optimal_load_factor(50,.8)), 1.54167288, places=7)
        self.assertEqual(optimal_load_factor(50,0), 0)
        self.assertEqual(optimal_load_factor(50,1), 1)

    def test_conditional_price_weighting(self):
        out = dict(prices=np.array([[np.inf,np.inf],[2.,4.],[8.,10.]]),
                   prob=np.array([[.5,.2],[.3,.5],[.2,.3]]))
        np.testing.assert_allclose(posted_prices(out), [4.4,6.25])
        out['prob'] = np.array([[1.,1.],[0.,0.],[0.,0.]])
        self.assertTrue(np.isnan(posted_prices(out)).all())

    def test_refunds(self):
        for model, target in [('exponential',79.1),('ces',82.3)]:
            data = loadmat(ROOT/'time_varying_demand'/'data'/f'{model}.mat', simplify_cells=True)
            self.assertAlmostEqual(gap_closed(data), target, delta=.05)
            for key in ('out_base_mu','out_opt','out_plan'):
                self.assertTrue(np.isfinite(posted_prices(data[key])).all())
                np.testing.assert_allclose(data[key]['prob'].sum(axis=0),1,atol=1e-10)
            self.assertNotIn('out_opt_mu0',data)

    def test_duopoly_table(self):
        targets = {'exponential':[20.,5.7,98.3,150.,77.4,76.2],
                   'logit':[20.,6.,98.2,176.,90.7,75.]}
        for model, values in targets.items():
            rows = load(model)
            self.assertEqual(len(rows),34)
            self.assertEqual({r['mode'] for r in rows},{'complete'})
            result = metrics(rows)
            for i,(actual,target) in enumerate(zip(result.values(),values)):
                self.assertAlmostEqual(actual,target,delta=.5 if i==3 else .05)

    def test_distributed_inputs(self):
        entries = json.loads((ROOT/'data_checksums.json').read_text())
        for name, expected in entries.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)


if __name__ == '__main__':
    unittest.main()

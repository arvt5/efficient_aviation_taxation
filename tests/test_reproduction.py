import tempfile
from pathlib import Path
import unittest

from aviation_taxation.reproduce import reproduce
from aviation_taxation.accounting import endpoint_metrics
from aviation_taxation.reproduce import ROOT, read_json


class ReproductionTests(unittest.TestCase):
    def test_published_values_and_interpolated_roots(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = reproduce(Path(folder))
        self.assertTrue(audit['passed'])
        self.assertEqual(len(audit['checks']), 42)

    def test_fiscal_identity_and_frequency_cancellation(self):
        config = read_json(ROOT / 'config/paper.json')
        baseline = read_json(ROOT / 'data/archived/status_quo.json')
        record = read_json(ROOT / 'data/archived/passenger_neutral.json')
        metrics = endpoint_metrics(record, baseline, config, carbon=True)
        self.assertAlmostEqual(metrics['fiscal_dividend_busd'], 67.4985988252662)
        emissions_bn_tonnes = config['status_quo_emissions_mt'] / 1000 * record['M'] / baseline['M']
        self.assertAlmostEqual(metrics['carbon_revenue_busd'] / emissions_bn_tonnes,
                               metrics['carbon_tax_usd_per_tonne'])

    def test_negative_carbon_instrument_is_preserved(self):
        config = read_json(ROOT / 'config/paper.json')
        baseline = read_json(ROOT / 'data/archived/status_quo.json')
        record = read_json(ROOT / 'data/archived/passenger_neutral.json')
        record['implied_tau_c_model_per_flight'] = -0.5
        metrics = endpoint_metrics(record, baseline, config, carbon=True)
        self.assertLess(metrics['carbon_tax_usd_per_tonne'], 0)
        self.assertLess(metrics['carbon_revenue_busd'], 0)


if __name__ == '__main__':
    unittest.main()

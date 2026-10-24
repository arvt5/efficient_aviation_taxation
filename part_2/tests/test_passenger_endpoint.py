import unittest

from aviation_taxation.passenger_endpoint import summarize, validate_parameters
from aviation_taxation.reproduce import ROOT, read_json


class PassengerEndpointTests(unittest.TestCase):
    def test_accounting_derives_instrument_from_policy_profits(self):
        raw = read_json(ROOT / 'data/archived/passenger_neutral_raw.json')
        params = read_json(ROOT / 'config/passenger_training.json')['params']
        report = summarize(raw, params)
        metrics = report['global_results']
        self.assertAlmostEqual(metrics['carbon_tax_usd_per_tonne'], 139.6082538882, places=8)
        self.assertAlmostEqual(metrics['fiscal_dividend_busd'], 67.4985988253, places=8)
        self.assertAlmostEqual(metrics['welfare_gain_busd'], 53.6177880453, places=8)
        self.assertTrue(report['policy_conclusion']['net_fiscal_dividend_positive'])
        self.assertTrue(report['policy_conclusion']['welfare_gain_positive'])
        self.assertLess(report['policy_conclusion']['passenger_change_percent'], 0)
        for difference in report['difference_from_selected_archive'].values():
            self.assertAlmostEqual(difference, 0, places=10)
        changed = summarize({**raw, 'firm_operating_profit_before_fixed': raw['firm_operating_profit_before_fixed'] + .1}, params)
        self.assertGreater(changed['global_results']['carbon_tax_usd_per_tonne'], metrics['carbon_tax_usd_per_tonne'])
        self.assertEqual(changed['global_results']['welfare_gain_busd'], metrics['welfare_gain_busd'])

    def test_wrong_checkpoint_parameters_are_rejected(self):
        expected = read_json(ROOT / 'config/passenger_training.json')['params']
        validate_parameters(expected, expected)
        for key in expected:
            with self.assertRaises(ValueError):
                validate_parameters({**expected, key: expected[key] + .01}, expected)
        with self.assertRaises(ValueError):
            validate_parameters({}, expected)


if __name__ == '__main__':
    unittest.main()

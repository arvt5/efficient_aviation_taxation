import unittest

from aviation_taxation.eu_policy_brief import calculate


class EUPolicyBriefTests(unittest.TestCase):
    def test_policy_versions_and_brief_numbers(self):
        result = calculate()
        self.assertEqual(result['assumptions']['effective_scale'], .05)
        expected = [
            (50.6561234623, 67.4985988253),
            (55.0800292598, 53.6177880453),
            (129.5442947334, 129.5442947334),
            (91.3524317852, 89.8901905707),
            (14.8975938943, 14.8975938943),
            (4.8838199115, 4.8838199115),
        ]
        for row, (refined, unrefined) in zip(result['rows'], expected):
            self.assertAlmostEqual(row['refined_global_busd'], refined, places=8)
            self.assertAlmostEqual(row['unrefined_global_busd'], unrefined, places=8)
            self.assertAlmostEqual(row['refined_eu_beur'], .05 * refined, places=8)
            self.assertAlmostEqual(row['unrefined_eu_beur'], .05 * unrefined, places=8)
        policies = result['policies']
        self.assertEqual(policies['refined']['M'], policies['unrefined']['M'])
        self.assertAlmostEqual(policies['refined']['vacancy_percent'], 2.66275196, places=7)
        self.assertAlmostEqual(policies['unrefined']['vacancy_percent'], 3.52364643, places=7)
        self.assertAlmostEqual(policies['unrefined']['passenger_change_percent'], -.67213, places=5)

    def test_revenue_rebasing(self):
        standard, doubled = calculate(), calculate(100)
        for a, b in zip(standard['rows'], doubled['rows']):
            for policy in ('refined', 'unrefined'):
                self.assertEqual(a[f'{policy}_global_busd'], b[f'{policy}_global_busd'])
                self.assertEqual(2 * a[f'{policy}_eu_beur'], b[f'{policy}_eu_beur'])
        self.assertEqual(standard['policies'], doubled['policies'])
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                calculate(invalid)


if __name__ == '__main__':
    unittest.main()

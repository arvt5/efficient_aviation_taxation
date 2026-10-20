import json
import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np

from aviation_taxation.welfare_plot import (
    DEFAULT_SUMMARY, DEFAULT_INPUT, DEFAULT_FITS, load_data, quadratic_peak,
)


class WelfareTests(unittest.TestCase):
    def test_implementing_carbon_prices_match_paper(self):
        summary = json.loads(DEFAULT_SUMMARY.read_text())
        m, conversion, _, panels = load_data(DEFAULT_INPUT, summary)
        self.assertEqual(len(m), 15)
        fit = json.loads(DEFAULT_FITS.read_text())['implied_tau_c_model_per_flight']
        coefficients = fit['candidates'][str(fit['selected_degree'])]['coefficients']
        for (_, welfare, _), expected in zip(panels, (-299, -104, 87, 206)):
            _, optimum, _ = quadratic_peak(m, welfare)
            tax = float(np.polyval(coefficients, optimum)) * conversion / 0.9
            self.assertEqual(round(tax), expected)


if __name__ == '__main__':
    unittest.main()

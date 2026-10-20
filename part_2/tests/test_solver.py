import unittest
from importlib.util import find_spec
import numpy as np

HAS_TORCH = find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from aviation_taxation.equilibrium import loss_fn
    from aviation_taxation.model import FractionalParams, demand_M3, brute_force_demand
    from aviation_taxation.network import FractionalDEQN, demand_torch, sample_interior
    from aviation_taxation.tables import build_tables
    from aviation_taxation.evaluation import policy_eval_full


@unittest.skipUnless(HAS_TORCH, 'Install part_2[solver] for numerical solver tests')
class SolverTests(unittest.TestCase):
    def test_demand_matches_independent_quadrature(self):
        params = FractionalParams(T=3, d=3/2.6, N=2, kappa=0.28)
        for phi in (0.2, 0.7):
            expected = brute_force_demand(.3, .5, .6, phi, 2, 1, 2, params, n_grid=20001)
            exact = demand_M3(.3, .5, .6, phi, 2, 1, 2, params)
            np.testing.assert_allclose(exact, expected, atol=1e-4)
            actual = demand_torch(torch.tensor([[.3, .5, .6]], dtype=torch.float64),
                                  torch.tensor([phi], dtype=torch.float64),
                                  torch.tensor([[2., 1., 2.]], dtype=torch.float64), params)
            np.testing.assert_allclose(actual.numpy()[0], exact, atol=1e-10)

    def test_residual_gradient_and_small_policy_evaluation(self):
        torch.manual_seed(12)
        params = FractionalParams(T=3, d=3/2.6, N=2, lambda_bar=1, kappa=.28)
        net = FractionalDEQN(hidden=16, activation='relu')
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        for _ in range(2):
            optimizer.zero_grad()
            batch = sample_interior(16, params, .01, device=torch.device('cpu'))
            loss, _ = loss_fn(net, batch, params, .01)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
            optimizer.step()
        phi = np.linspace(0, 1, 11)
        prices, demand, _ = build_tables(net, params, phi)
        values, history = policy_eval_full(prices, demand, params, phi)
        self.assertTrue(np.isfinite(values).all())
        self.assertLess(history[-1], 1e-8)


if __name__ == '__main__':
    unittest.main()

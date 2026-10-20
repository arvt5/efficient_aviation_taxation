function G = calib_2moments(x, N, T, vr_target, C_target, dh, deh, sh, c_fixed, pf)
% calib_2moments  Two-moment calibration residual for lsqnonlin.
%   Matches vacancy rate and relative marginal cost at tau=0.
%
%   x       : Parameter vector.
%             If c_fixed is empty: x = [marginal_cost, demand_params...]
%             If c_fixed is set:   x = [arrival_rate_L, demand_params...]
%   c_fixed : (Optional) Fixed marginal cost value.
%   pf      : (Optional) Price formula handle.
%
%   G  : [VR residual; RMC residual]

    if nargin < 9,  c_fixed = []; end
    if nargin < 10, pf = [];      end

    out  = solve_bellman_monopol(x, N, T, 0, dh, deh, sh, c_fixed, pf);
    G(1) = out.vacancy_rate          - vr_target;
    G(2) = out.relative_marginal_costs - C_target;
end

function out = solve_bellman_planner(z, N, T, demand_handle, surplus_handle, c_fixed)
% solve_bellman_planner  Social planner's backward induction.
%
%   The planner maximises PS + CS each period. The FOC reduces to:
%       p_W = c + delta_V   (explicit — no solver needed)
%   because S'(p) = -D(p) cancels the D(p) term from the revenue FOC.
%   The value update is therefore:
%       V(n,t) = L * S(p_W) + V(n, t+1)
%
%   No vacancy tax enters the planner's problem: tau is a pure transfer
%   and cancels in PS + CS + tau*(N-sales).
%
%   z              : Parameter vector.
%                    If c_fixed empty: z = [marginal_cost, demand_params...]
%                    If c_fixed set:   z = [arrival_rate_L, demand_params...]
%   c_fixed        : (Optional) Fixed marginal cost value.
%   demand_handle  : @(p, t, params) -> D(p)
%   surplus_handle : @(p, t, params) -> S(p) = integral_p^inf D(q) dq
%
%   Output struct matches solve_bellman_monopol for compatibility.

    if nargin >= 6 && ~isempty(c_fixed)
        marginal_cost = c_fixed;
        L             = z(1);
        params        = z(2:end);
    else
        marginal_cost = z(1);
        L             = N / T;
        params        = z(2:end);
    end

    fprintf('Solving Planner model (N=%d, T=%d, L=%.4f, c=%.4f)...\n', ...
        N, T, L, marginal_cost);

    V             = zeros(N+1, T+1);
    prices        = zeros(N+1, T);
    S_arr         = zeros(N+1, T);
    Prob_sale_arr = zeros(N+1, T);

    % --- Backward Induction ---
    for t = T:-1:1
        for n = 1:N+1
            if n == 1
                prices(n, t) = inf;
                continue;
            end

            delta_V    = V(n, t+1) - V(n-1, t+1);
            p_W        = marginal_cost + delta_V;

            prices(n, t)        = p_W;
            S_arr(n, t)         = surplus_handle(p_W, t, params);
            Prob_sale_arr(n, t) = L * demand_handle(p_W, t, params);

            V(n, t) = L * S_arr(n, t) + V(n, t+1);
        end
    end

    % --- Forward Simulation ---
    prob = zeros(N+1, T);
    prob(N+1, 1) = 1;

    for t = 1:T-1
        for n = 1:N+1
            if prob(n, t) == 0, continue; end
            p_sale = Prob_sale_arr(n, t);
            prob(n, t+1) = prob(n, t+1) + prob(n, t) * (1 - p_sale);
            if n > 1
                prob(n-1, t+1) = prob(n-1, t+1) + prob(n, t) * p_sale;
            end
        end
    end

    % --- Summary Statistics ---
    mask        = isfinite(prices);
    real_prices = prob .* Prob_sale_arr .* prices;
    real_prices(~mask) = 0;

    expected_sales_per_period   = sum(prob .* Prob_sale_arr, 1);
    expected_revenue_per_period = sum(real_prices, 1);

    out.expected_price_per_period = expected_revenue_per_period ./ expected_sales_per_period;
    out.expected_price_per_period(expected_sales_per_period < 1e-10) = NaN;

    expected_revenue_total = sum(real_prices, 'all');
    expected_sales_total   = sum(prob .* Prob_sale_arr, 'all');
    expected_cs_total      = sum(prob .* L .* S_arr, 'all');
    tickets_unsold         = N - expected_sales_total;

    out.expected_revenue = expected_revenue_total;
    out.expected_sales   = expected_sales_total;
    out.vacancy_rate     = tickets_unsold / N;
    out.av_price         = expected_revenue_total / expected_sales_total;
    out.relative_marginal_costs = marginal_cost / out.av_price;

    % PS has no vacancy tax term (tau is a transfer, not a social cost)
    out.ps            = expected_revenue_total - expected_sales_total * marginal_cost;
    out.cs            = expected_cs_total;
    out.total_welfare = out.ps + out.cs;

    out.valuation = V;
    out.prices    = prices;
    out.prob      = prob;
    out.expected_sales_per_period = expected_sales_per_period;

    fprintf('  Planner | W=%.4f | PS=%.4f | CS=%.4f | RMC=%.3f | VR=%.2f%%\n', ...
        out.total_welfare, out.ps, out.cs, out.relative_marginal_costs, out.vacancy_rate*100);
end

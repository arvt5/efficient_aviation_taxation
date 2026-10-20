function out = solve_bellman_return(z, N, T, tau, xi, mu_bar, demand_handle, deriv_handle, surplus_handle, psat_handle, c_fixed, price_formula_xi0)
% solve_bellman_return  Generic engine for value + welfare Bellman under return policy.
%   Corrected to enforce the Bernoulli constraint: P_sale + P_return <= 1.
%   Summary statistics use survival-weighted Net Sales and Net Revenue.
%   Also reports expected sales split across the first and last halves of
%   the sales horizon.

    c      = c_fixed;
    mu     = mu_bar / T;
    params = z;
    returns_active = xi > 0;

    V = zeros(N+1, T+1);
    for n = 1:N+1
        V(n, T+1) = -tau * (n - 1);
    end
    prices = zeros(N+1, T);
    D_arr  = zeros(N+1, T);
    S_arr  = zeros(N+1, T);

    % -----------------------------------------------------------------------
    % Backward Induction: Value Bellman
    % -----------------------------------------------------------------------
    for t = T:-1:1
        for n = 1:N+1
            if n == 1
                prices(n, t) = inf;
                continue;
            end

            n_sold  = N - (n - 1);
            delta_V = V(n, t+1) - V(n-1, t+1);
            c_eff   = c + delta_V;

            % PHYSICAL CONSTRAINT: L*D(p) + n_sold*mu <= 1
            return_prob = returns_active * n_sold * mu;
            target_D = max(0, 1 - return_prob);
            p_sat    = psat_handle(t, params, target_D);

            foc = @(p) (deriv_handle(p, t, params) * (p - c_eff) + demand_handle(p, t, params)) - (n_sold * mu * xi);

            if foc(p_sat) <= 0
                p_star = p_sat;
            else
                p_max = max(100, price_formula_xi0(c_eff, t, params) * 2);
                if ~isfinite(p_max), p_max = 1000; end

                if foc(p_sat) * foc(p_max) < 0
                    p_star = fzero(foc, [p_sat, p_max]);
                else
                    p_star = fzero(foc, price_formula_xi0(c_eff, t, params));
                end
            end

            if ~isfinite(p_star) || p_star < p_sat
                p_star = p_sat;
            end

            D_val = demand_handle(p_star, t, params);
            S_val = surplus_handle(p_star, t, params);

            prices(n, t) = p_star;
            D_arr(n, t)  = D_val;
            S_arr(n, t)  = S_val;

            DV_ret = 0;
            if returns_active && n < N+1
                DV_ret = V(n+1, t+1) - V(n, t+1);
            end

            % Match decay always affects welfare via survival, but only an
            % active return policy releases seats back into inventory.
            V(n, t) = D_val*(p_star - c_eff) - n_sold*mu*xi*p_star ...
                      + return_prob*DV_ret + V(n, t+1);
        end
    end

    % -----------------------------------------------------------------------
    % Backward Induction: Welfare Bellman
    % -----------------------------------------------------------------------
    W = zeros(N+1, T+1);

    for t = T:-1:1
        survival = exp(-mu_bar * (1 - t/T));
        for n = 1:N+1
            if n == 1
                n_sold      = N;
                return_prob = returns_active * n_sold * mu;
                DW_ret      = W(2, t+1) - W(1, t+1);
                W(n, t)     = return_prob * DW_ret + W(n, t+1);
                continue;
            end

            n_sold      = N - (n - 1);
            return_prob = returns_active * n_sold * mu;
            p           = prices(n, t);
            DW_n        = W(n, t+1) - W(n-1, t+1);
            DW_ret      = 0;
            if returns_active && n < N+1
                DW_ret = W(n+1, t+1) - W(n, t+1);
            end

            if ~isfinite(p)
                W(n, t) = return_prob * DW_ret + W(n, t+1);
                continue;
            end

            D_val        = D_arr(n, t);
            S_val        = S_arr(n, t);
            welfare_flow = (S_val + D_val * (p - c)) * survival;

            W(n, t) = welfare_flow - D_val * DW_n + return_prob * DW_ret + W(n, t+1);
        end
    end

    % -----------------------------------------------------------------------
    % Forward Simulation
    % -----------------------------------------------------------------------
    prob = zeros(N+1, T+1); % Extended to T+1 for departure state
    prob(N+1, 1) = 1;

    for t = 1:T
        for n = 1:N+1
            if prob(n, t) == 0, continue; end

            p_sale      = D_arr(n, t);
            n_sold      = N - (n - 1);
            return_prob = returns_active * n_sold * mu;

            % Constraints from Bellman ensure sum <= 1
            p_neither = max(0, 1 - p_sale - return_prob);

            prob(n, t+1) = prob(n, t+1) + prob(n, t) * p_neither;
            if n > 1
                prob(n-1, t+1) = prob(n-1, t+1) + prob(n, t) * p_sale;
            end
            if n < N+1 && return_prob > 0
                prob(n+1, t+1) = prob(n+1, t+1) + prob(n, t) * return_prob;
            end
        end
    end

    % -----------------------------------------------------------------------
    % Summary Statistics (Survival-weighted Net Sales and Net Revenue)
    % -----------------------------------------------------------------------
    t_vec        = (1:T);
    survival_vec = exp(-mu_bar * (1 - t_vec/T));
    n_sold_vec   = (N:-1:0)';

    % Expected transactions per period
    % Use safe_prices to avoid 0 * inf = NaN in sold-out states
    safe_prices = prices;
    safe_prices(~isfinite(safe_prices)) = 0;

    exp_sales_per_period = sum(prob(:, 1:T) .* D_arr, 1);
    exp_gross_rev_per_period = sum(prob(:, 1:T) .* D_arr .* safe_prices, 1);

    first_half_end = floor(T / 2);
    first_half_idx = 1:first_half_end;
    last_half_idx  = (first_half_end + 1):T;

    % Net Sales: Expected passengers who survive to departure
    net_sales_total = sum(exp_sales_per_period .* survival_vec);
    net_sales_first_half = sum(exp_sales_per_period(first_half_idx) .* survival_vec(first_half_idx));
    net_sales_last_half  = sum(exp_sales_per_period(last_half_idx)  .* survival_vec(last_half_idx));

    % Gross sales: Expected ticket transactions by calendar half
    gross_sales_first_half = sum(exp_sales_per_period(first_half_idx));
    gross_sales_last_half  = sum(exp_sales_per_period(last_half_idx));

    % Net Revenue: Gross Sales Revenue - Expected Refund Payments
    exp_refunds_per_period = sum(prob(:, 1:T) .* n_sold_vec .* (returns_active * mu) .* xi .* safe_prices, 1);

    net_rev_total = sum(exp_gross_rev_per_period) - sum(exp_refunds_per_period);

    tickets_unsold = N - net_sales_total;

    out.expected_revenue = net_rev_total;
    out.expected_sales   = net_sales_total;
    out.expected_sales_first_half = gross_sales_first_half;
    out.expected_sales_last_half  = gross_sales_last_half;
    out.expected_net_sales_first_half = net_sales_first_half;
    out.expected_net_sales_last_half  = net_sales_last_half;
    out.vacancy_rate     = tickets_unsold / N;
    out.av_price         = net_rev_total / net_sales_total;
    out.relative_marginal_costs = c / out.av_price;

    % Welfare Components
    out.ps = (net_rev_total - net_sales_total * c) - tickets_unsold * tau;
    out.cs = sum(sum(prob(:, 1:T) .* S_arr, 1) .* survival_vec);
    out.total_welfare    = W(N+1, 1);

    out.valuation = V;
    out.welfare   = W;
    out.prices    = prices;
    out.prob      = prob;
    out.expected_sales_per_period = exp_sales_per_period;
end

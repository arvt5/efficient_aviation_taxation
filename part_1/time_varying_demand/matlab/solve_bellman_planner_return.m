function out = solve_bellman_planner_return(z, N, T, mu_bar, demand_handle, surplus_handle, c_fixed)
% solve_bellman_planner_return  Generic social planner's Bellman for return models.
%   Corrected to enforce the Bernoulli constraint and use survival-weighted Net Sales.
%   Also reports expected sales split across the first and last halves of
%   the sales horizon.

    c      = c_fixed;
    mu     = mu_bar / T;
    params = z;

    W      = zeros(N+1, T+1);
    prices = zeros(N+1, T);
    D_arr  = zeros(N+1, T);
    S_arr  = zeros(N+1, T);

    % -----------------------------------------------------------------------
    % Backward Induction: Planner Welfare Bellman
    % -----------------------------------------------------------------------
    for t = T:-1:1
        survival = exp(-mu_bar * (1 - t/T));
        amp      = exp( mu_bar * (1 - t/T));

        for n = 1:N+1
            if n == 1
                n_sold      = N;
                return_prob = n_sold * mu;
                DW_ret      = W(2, t+1) - W(1, t+1);

                W(n, t)     = return_prob * DW_ret + W(n, t+1);
                prices(n,t) = inf;
                continue;
            end

            n_sold      = N - (n - 1);
            return_prob = n_sold * mu;
            DW_n        = W(n, t+1) - W(n-1, t+1);
            DW_ret      = 0;
            if n < N+1
                DW_ret = W(n+1, t+1) - W(n, t+1);
            end

            % Planner's explicit price formula: p = c + DW_n * amp
            p_wm = c + DW_n * amp;
            if p_wm <= 0, p_wm = c; end

            D_val = demand_handle(p_wm, t, params);
            S_val = surplus_handle(p_wm, t, params);

            prices(n, t) = p_wm;
            D_arr(n, t)  = D_val;
            S_arr(n, t)  = S_val;

            % Simplified update: L*S(p^wm)*survival + returns
            W(n, t) = S_val * survival + return_prob * DW_ret + W(n, t+1);
        end
    end

    % -----------------------------------------------------------------------
    % Forward Simulation
    % -----------------------------------------------------------------------
    prob = zeros(N+1, T+1);
    prob(N+1, 1) = 1;

    for t = 1:T
        for n = 1:N+1
            if prob(n, t) == 0, continue; end
            p_sale      = D_arr(n, t);
            n_sold      = N - (n - 1);
            return_prob = n_sold * mu;
            p_neither   = max(0, 1 - p_sale - return_prob);

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
    % Summary Statistics (Survival-weighted Net Sales)
    % -----------------------------------------------------------------------
    t_vec        = (1:T);
    survival_vec = exp(-mu_bar * (1 - t_vec/T));

    % Use safe_prices to avoid 0 * inf = NaN in sold-out states
    safe_prices = prices;
    safe_prices(~isfinite(safe_prices)) = 0;

    exp_sales_per_period = sum(prob(:, 1:T) .* D_arr, 1);
    exp_rev_per_period   = sum(prob(:, 1:T) .* D_arr .* safe_prices, 1);

    first_half_end = floor(T / 2);
    first_half_idx = 1:first_half_end;
    last_half_idx  = (first_half_end + 1):T;

    net_sales_total = sum(exp_sales_per_period .* survival_vec);
    net_sales_first_half = sum(exp_sales_per_period(first_half_idx) .* survival_vec(first_half_idx));
    net_sales_last_half  = sum(exp_sales_per_period(last_half_idx)  .* survival_vec(last_half_idx));
    gross_sales_first_half = sum(exp_sales_per_period(first_half_idx));
    gross_sales_last_half  = sum(exp_sales_per_period(last_half_idx));
    net_rev_total   = sum(exp_rev_per_period); % Note: Planner ignores refunds

    tickets_unsold  = N - net_sales_total;

    out.expected_revenue = net_rev_total;
    out.expected_sales   = net_sales_total;
    out.expected_sales_first_half = gross_sales_first_half;
    out.expected_sales_last_half  = gross_sales_last_half;
    out.expected_net_sales_first_half = net_sales_first_half;
    out.expected_net_sales_last_half  = net_sales_last_half;
    out.vacancy_rate     = tickets_unsold / N;
    out.av_price         = net_rev_total / net_sales_total;
    out.relative_marginal_costs = c / out.av_price;
    out.total_welfare    = W(N+1, 1);

    out.welfare  = W;
    out.prices   = prices;
    out.prob     = prob;
    out.expected_sales_per_period = exp_sales_per_period;
end

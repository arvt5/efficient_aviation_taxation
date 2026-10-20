function [out] = solve_bellman_monopol(z, N, T, tau, demand_handle, deriv_handle, surplus_handle, c_fixed, price_formula, ticket_tax)
    % solve_bellman_monopol: Robust engine using exponential transition probabilities.
    %
    %   z             : Parameter vector.
    %                   If c_fixed is empty: z = [marginal_cost, demand_params...]
    %                   If c_fixed is set:   z = [arrival_rate_L, demand_params...]
    %   c_fixed       : (Optional) Fixed marginal cost value.
    %   price_formula : (Optional) @(c_eff, t, params) -> p_star.
    %                   When supplied, bypasses fzero entirely; deriv_handle unused.
    %                   e.g. exponential: @(c,t,p) c + 1./a_t(t,p)
    %                        CES:         @(c,t,p) eps_t(t,p).*c./(eps_t(t,p)-1)
    %   ticket_tax    : (Optional) per-ticket excise wedge. Demand sees the
    %                   gross posted price p; the firm receives p-ticket_tax.
    %                   The existing tau argument remains the vacancy tax.
    if nargin < 10 || isempty(ticket_tax)
        ticket_tax = 0;
    end

    if nargin >= 8 && ~isempty(c_fixed)
        marginal_cost = c_fixed;
        L             = z(1);
        params        = z(2:end);
    else
        marginal_cost = z(1);
        L             = N / T;
        params        = z(2:end);
    end

    fprintf('Solving Monopoly model (N=%d, T=%d, vacancy_tax=%.2f, ticket_tax=%.2f, L=%.4f, c=%.4f)...\n', ...
        N, T, tau, ticket_tax, L, marginal_cost);

    % Storage
    V = zeros(N + 1, T + 1);
    prices = zeros(N + 1, T);
    Prob_sale_arr = zeros(N + 1, T);
    S_arr = zeros(N + 1, T);

    % --- Backward Induction ---
    for t = T:-1:1
        for n = 1:N+1
            if n == 1
                prices(n, t) = inf;
                continue;
            end

            delta_V = V(n, t+1) - V(n-1, t+1);
            c_eff = marginal_cost + delta_V + ticket_tax - tau;

            if isnan(c_eff), error('solve_bellman_monopol: c_eff is NaN at n=%d, t=%d.', n, t); end

            % If tau exceeds the effective cost, the price formula is undefined.
            % Set price to infinity (firm does not sell) — outside the model's valid range.
            p0 = c_eff + 1.0;
            if t < T && isfinite(prices(n, t+1))
                p0 = prices(n, t+1);
            end

            if nargin >= 9 && ~isempty(price_formula)
                p_star = price_formula(c_eff, t, params);
            else
                [p_star, ~] = solve_period_monopol(demand_handle, deriv_handle, surplus_handle, params, c_eff, p0, t);
            end

            if p_star <= 0
                if c_eff > 0
                    % eps < 1: revenue is increasing in price — monopolist
                    % prices to infinity (does not sell).
                    p_star = inf;
                else
                    % c_eff <= 0: the interior optimum wants a negative
                    % price. If the model admits a saturation price where
                    % L*D(p)=1, use that corner. Otherwise fall back to the
                    % nonnegative boundary price. This matters for
                    % exponential demand, where max_p_sale = L < 1 when
                    % p=0, so no certainty-sale root exists.
                    lower_bound = 1e-8;
                    sat_fn = @(p) L * demand_handle(p, t, params) - 1;

                    if sat_fn(lower_bound) >= 0
                        p_star = fzero(sat_fn, [lower_bound, 1e6]);
                    else
                        p_star = lower_bound;
                    end
                end
            end

            % Enforce the Bernoulli feasibility constraint whenever the
            % candidate price implies a sale probability above one. This is
            % essential for CES demand, where the closed-form markup can be
            % positive yet still lie below the physical saturation floor.
            if isfinite(p_star)
                prob_sale_candidate = L * demand_handle(p_star, t, params);
                if prob_sale_candidate > 1
                    p_star = enforce_saturation_floor(L, demand_handle, params, t, p_star);
                end
            end

            prices(n, t) = p_star;
            prob_sale = L*demand_handle(p_star, t, params);
            Prob_sale_arr(n, t) = prob_sale;
            S_arr(n, t) = surplus_handle(p_star, t, params);

            V(n, t) = prob_sale* (p_star - c_eff) + V(n, t+1);
        end
    end

    % --- Forward Simulation ---
    prob = zeros(N + 1, T);
    prob(N + 1, 1) = 1;

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
    mask = isfinite(prices);
    real_prices = prob.*Prob_sale_arr.*prices;
    real_prices(~mask) = 0;

    % Expected price per period (probability weighted)
    % sum over n: prob(n,t) * prob_sale(n,t) * price(n,t) / sum over n: prob(n,t) * prob_sale(n,t)
    expected_sales_per_period = sum(prob.*Prob_sale_arr, 1);
    expected_revenue_per_period = sum(real_prices, 1);

    out.expected_price_per_period = expected_revenue_per_period ./ expected_sales_per_period;
    % Handle periods with zero sales
    out.expected_price_per_period(expected_sales_per_period < 1e-10) = NaN;

    expected_price = sum(real_prices, 'all');
    expected_sales_total = sum(prob.*Prob_sale_arr, 'all');
    expected_revenue_total = expected_price;

    expected_cs_total = sum(prob.*L.*S_arr, 'all');

    tickets_unsold = N - expected_sales_total;
    ticket_tax_revenue = expected_sales_total * ticket_tax;
    vacancy_tax_revenue = tickets_unsold * tau;

    out.expected_revenue = expected_revenue_total;
    out.expected_sales   = expected_sales_total;
    out.expected_unsold  = tickets_unsold;
    out.vacancy_rate     = tickets_unsold / N;
    out.av_price         = expected_revenue_total / expected_sales_total;
    out.relative_marginal_costs = marginal_cost / out.av_price;
    out.ticket_tax       = ticket_tax;
    out.vacancy_tax      = tau;
    out.ticket_tax_revenue = ticket_tax_revenue;
    out.vacancy_tax_revenue = vacancy_tax_revenue;

    out.ps = expected_revenue_total ...
        - expected_sales_total * marginal_cost ...
        - ticket_tax_revenue ...
        - vacancy_tax_revenue;
    out.cs = expected_cs_total;
    out.total_welfare = out.ps + out.cs + ticket_tax_revenue + vacancy_tax_revenue;

    out.valuation = V;
    out.prices    = prices;
    out.prob      = prob;
    out.expected_sales_per_period = expected_sales_per_period;

    fprintf('  VacancyTax=%.2f | TicketTax=%.2f | W=%.3f | PS=%.3f | CS=%.3f | RMC=%.3f | VR=%.1f%%\n', ...
        tau, ticket_tax, out.total_welfare, out.ps, out.cs, out.relative_marginal_costs, out.vacancy_rate*100);
end


function p_sat = enforce_saturation_floor(L, demand_handle, params, t, p_candidate)
% enforce_saturation_floor  Raise price to the lowest level satisfying
% L*D(p) <= 1, if such a finite saturation price exists.

    lower_bound = 1e-8;
    sat_fn = @(p) L * demand_handle(p, t, params) - 1;

    if sat_fn(lower_bound) < 0
        % No certainty-sale root exists. The nonnegative boundary is the
        % tightest feasible floor.
        p_sat = lower_bound;
        return;
    end

    upper_bound = max(1, p_candidate);
    max_upper   = 1e8;
    while sat_fn(upper_bound) > 0 && upper_bound < max_upper
        upper_bound = upper_bound * 2;
    end

    if sat_fn(upper_bound) > 0
        error('solve_bellman_monopol: could not bracket saturation price at t=%d.', t);
    end

    p_sat = fzero(sat_fn, [lower_bound, upper_bound]);
end

function [out] = solve_bellman_logit(z, NA, NB, T, vacancy_tax, mode, ticket_tax)
    %
    % Backward-induction Bellman solver for the duopoly logit demand model.
    %
    % mode: 'incomplete' (default) — firms ignore rival inventory externality
    %       'complete'             — firms internalize rival inventory effect
    %       'monopoly'             — joint two-product monopolist
    %
    if nargin < 6, mode = 'incomplete'; end
    if nargin < 7 || isempty(ticket_tax), ticket_tax = 0; end
    is_joint_monopoly = strcmp(mode, 'monopoly');
    complete_info     = strcmp(mode, 'complete');
    if ~(strcmp(mode, 'incomplete') || complete_info || is_joint_monopoly)
        error('solve_bellman_logit: mode "%s" is not implemented.', mode);
    end

    alpha         = z(1);
    s             = z(2);
    true_marginal_cost = z(3);
    marginal_cost = true_marginal_cost + ticket_tax - vacancy_tax;
    L             = z(4) / T;

    prices_A     = zeros(NA + 1, NB + 1, T);
    prices_B     = zeros(NA + 1, NB + 1, T);
    DA_arr       = zeros(NA + 1, NB + 1, T);
    DB_arr       = zeros(NA + 1, NB + 1, T);
    costs_A      = zeros(NA + 1, NB + 1, T);
    costs_B      = zeros(NA + 1, NB + 1, T);
    valuations_A = zeros(NA + 1, NB + 1, T);
    valuations_B = zeros(NA + 1, NB + 1, T);
    wA           = zeros(NA + 1, NB + 1, T);
    wB           = zeros(NA + 1, NB + 1, T);
    dr           = zeros(NA + 1, NB + 1, T);
    crosselast_A = zeros(NA + 1, NB + 1, T);
    crosselast_B = zeros(NA + 1, NB + 1, T);

    fprintf('Running model (mode=%s)...\n', mode);

    % =========================================================================
    % Terminal period (t = T): no continuation value
    % =========================================================================
    for nA = NA+1:-1:1
        for nB = NB+1:-1:1
            if nA < nB
                prices_A(nA, nB, T)     = prices_B(nB, nA, T);
                prices_B(nA, nB, T)     = prices_A(nB, nA, T);
                valuations_A(nA, nB, T) = valuations_B(nB, nA, T);
                valuations_B(nA, nB, T) = valuations_A(nB, nA, T);
                DA_arr(nA, nB, T)       = DB_arr(nB, nA, T);
                DB_arr(nA, nB, T)       = DA_arr(nB, nA, T);
                wA(nA, nB, T)           = wB(nB, nA, T);
                wB(nA, nB, T)           = wA(nB, nA, T);
                dr(nA, nB, T)           = dr(nB, nA, T);
                crosselast_A(nA, nB, T) = crosselast_B(nB, nA, T);
                crosselast_B(nA, nB, T) = crosselast_A(nB, nA, T);
                continue;
            end

            if nA > 1
                costs_A(nA, nB, T) = marginal_cost;
            else
                costs_A(nA, nB, T) = inf;
            end

            if nB > 1
                costs_B(nA, nB, T) = marginal_cost;
            else
                costs_B(nA, nB, T) = inf;
            end

            if nA == 1 && nB == 1
                prices_A(nA, nB, T)     = inf;
                prices_B(nA, nB, T)     = inf;
                DA_arr(nA, nB, T)       = 0;
                DB_arr(nA, nB, T)       = 0;
                valuations_A(nA, nB, T) = 0;
                valuations_B(nA, nB, T) = 0;
                wA(nA, nB, T)           = 0;
                wB(nA, nB, T)           = 0;
                continue;
            end

            p0 = [max(costs_A(nA, nB, T), 0.1), max(costs_B(nA, nB, T), 0.1)];
            [pA, pB, sol] = solve_period_logit(alpha, costs_A(nA, nB, T), costs_B(nA, nB, T), s, p0, mode, 0, 0, 0, 0, marginal_cost);

            if nA > 1 && ~isinf(pA)
                valuations_A(nA, nB, T) = L * sol.DA * (pA - costs_A(nA, nB, T));
                wA(nA, nB, T)           = sol.WA;
                DA_arr(nA, nB, T)       = sol.DA;
                prices_A(nA, nB, T)     = pA;
                dr(nA, nB, T)           = sol.diversion_ratio;
            end

            if nB > 1 && ~isinf(pB)
                valuations_B(nA, nB, T) = L * sol.DB * (pB - costs_B(nA, nB, T));
                if isnan(valuations_B(nA, nB, T))
                    error('solve_bellman_logit: NaN valuation_B at t=%d, nA=%d, nB=%d.', T, nA, nB);
                end
                wB(nA, nB, T)       = sol.WB;
                DB_arr(nA, nB, T)   = sol.DB;
                prices_B(nA, nB, T) = pB;
            end

            if isfinite(pA) && isfinite(pB)
                if nA > 1 && sol.DA > 1e-10
                    crosselast_A(nA, nB, T) = sol.dA_B * pB / sol.DA;
                end
                if nB > 1 && sol.DB > 1e-10
                    crosselast_B(nA, nB, T) = sol.dB_A * pA / sol.DB;
                end
            end
        end
    end

    % =========================================================================
    % Backward induction: t = T-1 down to 1
    % =========================================================================
    for t = T-1:-1:1
        for nA = NA+1:-1:1
            for nB = NB+1:-1:1
                if nA < nB
                    prices_A(nA, nB, t)     = prices_B(nB, nA, t);
                    prices_B(nA, nB, t)     = prices_A(nB, nA, t);
                    valuations_A(nA, nB, t) = valuations_B(nB, nA, t);
                    valuations_B(nA, nB, t) = valuations_A(nB, nA, t);
                    DA_arr(nA, nB, t)       = DB_arr(nB, nA, t);
                    DB_arr(nA, nB, t)       = DA_arr(nB, nA, t);
                    wA(nA, nB, t)           = wB(nB, nA, t);
                    wB(nA, nB, t)           = wA(nB, nA, t);
                    dr(nA, nB, t)           = dr(nB, nA, t);
                    crosselast_A(nA, nB, t) = crosselast_B(nB, nA, t);
                    crosselast_B(nA, nB, t) = crosselast_A(nB, nA, t);
                    continue;
                end

                costs_A(nA, nB, t) = marginal_cost;
                costs_B(nA, nB, t) = marginal_cost;

            if nA > 1
                cAA = valuations_A(nA, nB, t+1) - valuations_A(nA-1, nB, t+1);
                cBA = valuations_B(nA, nB, t+1) - valuations_B(nA-1, nB, t+1);
                costs_A(nA, nB, t) = costs_A(nA, nB, t) + cAA;
            else
                cAA = inf;
                cBA = 0;
                costs_A(nA, nB, t) = inf;
            end

            if nB > 1
                cBB = valuations_B(nA, nB, t+1) - valuations_B(nA, nB-1, t+1);
                cAB = valuations_A(nA, nB, t+1) - valuations_A(nA, nB-1, t+1);
                costs_B(nA, nB, t) = costs_B(nA, nB, t) + cBB;
            else
                cBB = inf;
                cAB = 0;
                costs_B(nA, nB, t) = inf;
            end

            % Incomplete-info: firms ignore cross-inventory externality
            if ~complete_info && ~is_joint_monopoly
                cAB = 0;
                cBA = 0;
            end

            cA = costs_A(nA, nB, t);
            cB = costs_B(nA, nB, t);

                if nA == 1 && nB == 1
                    prices_A(nA, nB, t)     = inf;
                    prices_B(nA, nB, t)     = inf;
                    DA_arr(nA, nB, t)       = 0;
                    DB_arr(nA, nB, t)       = 0;
                    valuations_A(nA, nB, t) = 0;
                    valuations_B(nA, nB, t) = 0;
                    wA(nA, nB, t)           = 0;
                    wB(nA, nB, t)           = 0;
                    continue;
                end

                if isnan(cA) || isnan(cB)
                    error('solve_bellman_logit: NaN effective cost at nA=%d, nB=%d, t=%d (cA=%.4f, cB=%.4f).', nA, nB, t, cA, cB);
                end

                p0 = [max(prices_A(nA, nB, t+1), 0.1), max(prices_B(nA, nB, t+1), 0.1)];
                [pA, pB, sol] = solve_period_logit(alpha, cA, cB, s, p0, mode, cAA, cAB, cBB, cBA, marginal_cost);

                if ~is_joint_monopoly && (pA < costs_A(nA, nB, t) || pB < costs_B(nA, nB, t))
                    warning('solve_bellman_logit: price below effective cost at nA=%d, nB=%d, t=%d (pA=%.4f, cA=%.4f, pB=%.4f, cB=%.4f).', ...
                        nA, nB, t, pA, costs_A(nA, nB, t), pB, costs_B(nA, nB, t));
                end
                if sol.SOC_AA > 0 || sol.SOC_BB > 0
                    warning('solve_bellman_logit: SOC violated at nA=%d, nB=%d, t=%d (SOC_AA=%.4f, SOC_BB=%.4f).', ...
                        nA, nB, t, sol.SOC_AA, sol.SOC_BB);
                end
                foc_resid = max(abs([sol.FOC_A, sol.FOC_B]), [], 'omitnan');
                if ~isempty(foc_resid) && isfinite(foc_resid) && foc_resid > 1e-4
                    warning('solve_bellman_logit: FOC residual too large at nA=%d, nB=%d, t=%d (FOC_A=%.2e, FOC_B=%.2e).', ...
                        nA, nB, t, sol.FOC_A, sol.FOC_B);
                end

                if nA > 1 && ~isinf(pA)
                    valuations_A(nA, nB, t) = valuations_A(nA, nB, t+1) + L * (sol.DA * (pA - costs_A(nA, nB, t)) - sol.DB * cAB);
                    if isnan(valuations_A(nA, nB, t))
                        error('solve_bellman_logit: NaN valuation_A at t=%d, nA=%d, nB=%d (V_next=%.4f, DA=%.4f, pA=%.4f, cA=%.4f).', ...
                            t, nA, nB, valuations_A(nA, nB, t+1), sol.DA, pA, costs_A(nA, nB, t));
                    end
                    wA(nA, nB, t)       = sol.WA;
                    DA_arr(nA, nB, t)   = sol.DA;
                    prices_A(nA, nB, t) = pA;
                    dr(nA, nB, t)       = sol.diversion_ratio;
                end

                if nB > 1 && ~isinf(pB)
                    valuations_B(nA, nB, t) = valuations_B(nA, nB, t+1) + L * (sol.DB * (pB - costs_B(nA, nB, t)) - sol.DA * cBA);
                    if isnan(valuations_B(nA, nB, t))
                        error('solve_bellman_logit: NaN valuation_B at t=%d, nA=%d, nB=%d (V_next=%.4f, DB=%.4f, pB=%.4f, cB=%.4f).', ...
                            t, nA, nB, valuations_B(nA, nB, t+1), sol.DB, pB, costs_B(nA, nB, t));
                    end
                    wB(nA, nB, t)       = sol.WB;
                    DB_arr(nA, nB, t)   = sol.DB;
                    prices_B(nA, nB, t) = pB;
                end

                if isfinite(pA) && isfinite(pB)
                    if nA > 1 && sol.DA > 1e-10
                        crosselast_A(nA, nB, t) = sol.dA_B * pB / sol.DA;
                    end
                    if nB > 1 && sol.DB > 1e-10
                        crosselast_B(nA, nB, t) = sol.dB_A * pA / sol.DB;
                    end
                end
            end
        end
    end

    % =========================================================================
    % Forward probability simulation
    % =========================================================================
    prob = zeros(NA + 1, NB + 1, T);
    prob(NA + 1, NB + 1, 1) = 1;
    qA = zeros(NA + 1, NB + 1, T);
    qB = zeros(NA + 1, NB + 1, T);
    qA(:, :, 1) = L * DA_arr(:, :, 1);
    qB(:, :, 1) = L * DB_arr(:, :, 1);

    for t = 2:T
        for nA = NA+1:-1:1
            for nB = NB+1:-1:1
                qA(nA, nB, t) = L * DA_arr(nA, nB, t);
                qB(nA, nB, t) = L * DB_arr(nA, nB, t);

                stay = prob(nA, nB, t-1) * (1 - L * DA_arr(nA, nB, t-1) - L * DB_arr(nA, nB, t-1));

                fromA = 0;
                if nA < NA + 1
                    fromA = prob(nA+1, nB, t-1) * L * DA_arr(nA+1, nB, t-1);
                end

                fromB = 0;
                if nB < NB + 1
                    fromB = prob(nA, nB+1, t-1) * L * DB_arr(nA, nB+1, t-1);
                end

                prob(nA, nB, t) = fromA + fromB + stay;
            end
        end
    end

    if any(prob(:, :, T) > 1, 'all') || any(qA(:, :, T) > 1, 'all')
        error('Probability > 1 detected at t=%d.', T);
    end

    % =========================================================================
    % Summary statistics
    % =========================================================================
    vacant_seats_A = dot(0:NA, sum(prob(:, :, T), 2));
    vacancy_rate   = vacant_seats_A / NA;
    tickets_sold   = sum(prob .* (qA + qB), 'all');
    tickets_unsold = 2 * NA - tickets_sold;

    maskA = isfinite(prices_A) & isfinite(costs_A);
    maskB = isfinite(prices_B) & isfinite(costs_B);

    termA = prob .* qA .* (prices_A - true_marginal_cost - ticket_tax);
    termB = prob .* qB .* (prices_B - true_marginal_cost - ticket_tax);
    termA(~maskA) = 0;
    termB(~maskB) = 0;
    ticket_tax_revenue = ticket_tax * tickets_sold;
    vacancy_tax_revenue = vacancy_tax * tickets_unsold;
    ps = sum(termA + termB, 'all') - vacancy_tax_revenue;

    cs = L * sum(prob .* (wA + wB), 'all');
    total_welfare = ps + cs + ticket_tax_revenue + vacancy_tax_revenue;

    termA_rev = prob .* qA .* prices_A;
    termB_rev = prob .* qB .* prices_B;
    termA_rev(~maskA) = 0;
    termB_rev(~maskB) = 0;

    revenue_A = sum(termA_rev, 'all');
    revenue_B = sum(termB_rev, 'all');
    total_revenue = revenue_A + revenue_B;
    av_price  = total_revenue / tickets_sold;
    relative_marginal_costs = true_marginal_cost / av_price;

    denom_qA = sum(prob .* qA, 'all');

    out.mode                    = mode;
    out.vacancy_tax             = vacancy_tax;
    out.ticket_tax              = ticket_tax;
    out.ps                      = ps;
    out.cs                      = cs;
    out.valuation               = ps;
    out.revenue                 = total_revenue;
    out.gross_revenue           = total_revenue;
    out.revenue_A               = revenue_A;
    out.revenue_B               = revenue_B;
    out.av_price                = av_price;
    out.median_price            = median(prices_B(maskB));
    out.total_welfare           = total_welfare;
    out.vacancy_rate            = vacancy_rate * 100;
    out.prices_A                = prices_A;
    out.prices_B                = prices_B;
    out.prob                    = prob;
    out.diversion_ratio         = sum(prob .* qA .* dr, 'all') / denom_qA;
    out.cross_elasticity        = sum(prob .* qA .* crosselast_A, 'all') / denom_qA;
    out.relative_marginal_costs = relative_marginal_costs;
    out.ticket_tax_revenue      = ticket_tax_revenue;
    out.vacancy_tax_revenue     = vacancy_tax_revenue;
    out.expected_sales          = tickets_sold;
    out.expected_unsold         = tickets_unsold;
    out.entry_value_A           = valuations_A(NA + 1, NB + 1, 1);
    out.entry_value_B           = valuations_B(NA + 1, NB + 1, 1);

    fprintf(['  VacancyTax=%.2f | TicketTax=%.2f | %s | W=%.3f | PS=%.3f | CS=%.3f' ...
        ' | RMC=%.3f | VR=%.1f%% | DR=%.2f%%\n'], vacancy_tax, ticket_tax, mode, total_welfare, ...
        ps, cs, relative_marginal_costs, vacancy_rate * 100, out.diversion_ratio * 100);
end

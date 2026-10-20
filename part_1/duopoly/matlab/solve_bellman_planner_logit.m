function out = solve_bellman_planner_logit(z, NA, NB, T)
% solve_bellman_planner_logit  Social planner Bellman solver for duopoly logit demand.
%
%   z = [alpha, a, c, lambda]
%
%   The planner chooses both prices jointly to maximize expected welfare:
%       CS(pA,pB) + sA(pA,pB) * (pA - c - dWA) + sB(pA,pB) * (pB - c - dWB)
%   where CS is expected consumer surplus per arrival and dWi are welfare
%   shadow values of inventory.
%
%   No vacancy tax enters the planner problem because it is a transfer.

    alpha         = z(1);
    a             = z(2);
    marginal_cost = z(3);
    L             = z(4) / T;

    fprintf('Running planner model (logit)...\n');

    W          = zeros(NA + 1, NB + 1, T + 1);
    prices_A   = inf(NA + 1, NB + 1, T);
    prices_B   = inf(NA + 1, NB + 1, T);
    DA_arr     = zeros(NA + 1, NB + 1, T);
    DB_arr     = zeros(NA + 1, NB + 1, T);
    CS_arr     = zeros(NA + 1, NB + 1, T);
    dr         = zeros(NA + 1, NB + 1, T);
    foc_tol    = 1e-8;

    % =========================================================================
    % Backward induction
    % =========================================================================
    for t = T:-1:1
        for nA = 1:NA + 1
            for nB = 1:NB + 1
                if nA == 1 && nB == 1
                    continue;
                end

                if nA == 1
                    dWB = W(nA, nB, t + 1) - W(nA, nB - 1, t + 1);
                    pB = marginal_cost + dWB;
                    sB = single_share(pB, alpha, a);

                    cs = single_cs(pB, alpha, a);

                    prices_B(nA, nB, t) = pB;
                    DA_arr(nA, nB, t)   = 0;
                    DB_arr(nA, nB, t)   = sB;
                    CS_arr(nA, nB, t)   = cs;
                    W(nA, nB, t)        = W(nA, nB, t + 1) + L * cs;
                    continue;
                end

                if nB == 1
                    dWA = W(nA, nB, t + 1) - W(nA - 1, nB, t + 1);
                    pA = marginal_cost + dWA;
                    sA = single_share(pA, alpha, a);

                    cs = single_cs(pA, alpha, a);

                    prices_A(nA, nB, t) = pA;
                    DA_arr(nA, nB, t)   = sA;
                    DB_arr(nA, nB, t)   = 0;
                    CS_arr(nA, nB, t)   = cs;
                    W(nA, nB, t)        = W(nA, nB, t + 1) + L * cs;
                    continue;
                end

                dWA = W(nA, nB, t + 1) - W(nA - 1, nB, t + 1);
                dWB = W(nA, nB, t + 1) - W(nA, nB - 1, t + 1);

                [pA, pB, sol] = solve_period_planner_logit(alpha, a, marginal_cost, dWA, dWB);

                foc_resid = max(abs(sol.foc));
                if ~isfinite(foc_resid) || foc_resid > foc_tol
                    error('solve_bellman_planner_logit: FOC residual too large at nA=%d, nB=%d, t=%d.', nA, nB, t);
                end

                prices_A(nA, nB, t) = pA;
                prices_B(nA, nB, t) = pB;
                DA_arr(nA, nB, t)   = sol.sA;
                DB_arr(nA, nB, t)   = sol.sB;
                CS_arr(nA, nB, t)   = sol.cs;
                dr(nA, nB, t)       = sol.diversion_ratio;

                W(nA, nB, t) = W(nA, nB, t + 1) + ...
                    L * (sol.cs + sol.sA * (pA - marginal_cost - dWA) + sol.sB * (pB - marginal_cost - dWB));
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
        for nA = NA + 1:-1:1
            for nB = NB + 1:-1:1
                qA(nA, nB, t) = L * DA_arr(nA, nB, t);
                qB(nA, nB, t) = L * DB_arr(nA, nB, t);

                stay = prob(nA, nB, t - 1) * (1 - qA(nA, nB, t - 1) - qB(nA, nB, t - 1));

                fromA = 0;
                if nA < NA + 1
                    fromA = prob(nA + 1, nB, t - 1) * qA(nA + 1, nB, t - 1);
                end

                fromB = 0;
                if nB < NB + 1
                    fromB = prob(nA, nB + 1, t - 1) * qB(nA, nB + 1, t - 1);
                end

                prob(nA, nB, t) = fromA + fromB + stay;
            end
        end
    end

    if any(prob(:, :, T) > 1 + 1e-10, 'all') || any(prob < -1e-12, 'all')
        error('solve_bellman_planner_logit: invalid state probabilities.');
    end
    if any(qA + qB > 1 + 1e-10, 'all') || any(qA < -1e-12, 'all') || any(qB < -1e-12, 'all')
        error('solve_bellman_planner_logit: invalid transition probabilities.');
    end

    % =========================================================================
    % Summary statistics
    % =========================================================================
    vacant_seats_A = dot(0:NA, sum(prob(:, :, T), 2));
    vacancy_rate   = vacant_seats_A / NA;
    tickets_sold   = sum(prob .* (qA + qB), 'all');

    maskA = isfinite(prices_A);
    maskB = isfinite(prices_B);

    termA_rev = prob .* qA .* prices_A;
    termB_rev = prob .* qB .* prices_B;
    termA_rev(~maskA) = 0;
    termB_rev(~maskB) = 0;

    revenue_A       = sum(termA_rev, 'all');
    revenue_B       = sum(termB_rev, 'all');
    total_revenue   = revenue_A + revenue_B;
    total_cs        = L * sum(prob .* CS_arr, 'all');
    total_ps        = total_revenue - tickets_sold * marginal_cost;
    total_welfare   = W(NA + 1, NB + 1, 1);

    av_price = revenue_A / (NA - vacant_seats_A);
    relative_marginal_costs = marginal_cost / av_price;
    denom_qA = sum(prob .* qA, 'all');

    out.mode                    = 'planner';
    out.ps                      = total_ps;
    out.cs                      = total_cs;
    out.revenue                 = total_revenue;
    out.revenue_A               = revenue_A;
    out.revenue_B               = revenue_B;
    out.av_price                = av_price;
    out.median_price            = median(prices_A(maskA));
    out.total_welfare           = total_welfare;
    out.vacancy_rate            = vacancy_rate * 100;
    out.prices_A                = prices_A;
    out.prices_B                = prices_B;
    out.prob                    = prob;
    out.welfare                 = W;
    out.valuation               = W;
    out.expected_sales_per_period_A = squeeze(sum(prob .* qA, [1 2]));
    out.expected_sales_per_period_B = squeeze(sum(prob .* qB, [1 2]));
    out.relative_marginal_costs = relative_marginal_costs;

    if denom_qA > 1e-12
        out.diversion_ratio = sum(prob .* qA .* dr, 'all') / denom_qA;
    else
        out.diversion_ratio = NaN;
    end

    fprintf(['  Planner | W=%.3f | PS=%.3f | CS=%.3f' ...
        ' | RMC=%.3f | VR=%.1f%% | DR=%.2f%%\n'], total_welfare, ...
        total_ps, total_cs, relative_marginal_costs, vacancy_rate * 100, out.diversion_ratio * 100);
end

function [pA, pB, out] = solve_period_planner_logit(alpha, a, c, dWA, dWB)
    pA = c + dWA;
    pB = c + dWB;

    sA = share_A(pA, pB, alpha, a);
    sB = share_B(pA, pB, alpha, a);

    out.sA = sA;
    out.sB = sB;
    out.cs = cs_logit(pA, pB, alpha, a);
    out.foc = planner_foc([pA; pB], alpha, a, c, dWA, dWB);

    if isfinite(pA) && isfinite(pB) && sA > 1e-10
        out.diversion_ratio = sB / (1 - sA);
    else
        out.diversion_ratio = NaN;
    end
end

function G = planner_foc(z, alpha, a, c, dWA, dWB)
    pA = z(1);
    pB = z(2);

    sA = share_A(pA, pB, alpha, a);
    sB = share_B(pA, pB, alpha, a);

    dA_A = -alpha * sA * (1 - sA);
    dB_B = -alpha * sB * (1 - sB);
    dA_B = alpha * sA * sB;
    dB_A = dA_B;

    G = zeros(2, 1);
    G(1) = dA_A * (pA - c - dWA) + dB_A * (pB - c - dWB);
    G(2) = dB_B * (pB - c - dWB) + dA_B * (pA - c - dWA);
end

function s = share_A(pA, pB, alpha, a)
    uA = exp(a - alpha * pA);
    uB = exp(a - alpha * pB);
    s = uA / (1 + uA + uB);
end

function s = share_B(pA, pB, alpha, a)
    uA = exp(a - alpha * pA);
    uB = exp(a - alpha * pB);
    s = uB / (1 + uA + uB);
end

function s = single_share(p, alpha, a)
    u = exp(a - alpha * p);
    s = u / (1 + u);
end

function cs = cs_logit(pA, pB, alpha, a)
    cs = (1 / alpha) * log(1 + exp(a - alpha * pA) + exp(a - alpha * pB));
end

function cs = single_cs(p, alpha, a)
    cs = (1 / alpha) * log(1 + exp(a - alpha * p));
end

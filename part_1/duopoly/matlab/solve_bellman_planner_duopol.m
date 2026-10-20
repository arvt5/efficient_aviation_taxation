function out = solve_bellman_planner_duopol(z, NA, NB, T)
% solve_bellman_planner_duopol  Social planner Bellman solver for custom duopoly demand.
%
%   z = [a, c, lambda, gamma]
%
%   The planner chooses both prices jointly to maximize expected welfare:
%       CS(pA,pB) + DA(pA,pB) * (pA - c - dWA) + DB(pA,pB) * (pB - c - dWB)
%   where CS = CSA(pA,pB) + CSA(pB,pA) is expected consumer surplus per arrival.
%
%   Symmetry is imposed explicitly by copying mirrored states, so reported
%   market statistics are exact under the symmetric primitives.

    a             = z(1);
    marginal_cost = z(2);
    L             = z(3) / T;
    gamma         = z(4);

    fprintf('Running planner model (custom duo)...\n');

    prices_A = inf(NA + 1, NB + 1, T);
    prices_B = inf(NA + 1, NB + 1, T);
    DA_arr   = zeros(NA + 1, NB + 1, T);
    DB_arr   = zeros(NA + 1, NB + 1, T);
    CS_arr   = zeros(NA + 1, NB + 1, T);
    W        = zeros(NA + 1, NB + 1, T + 1);
    dr       = zeros(NA + 1, NB + 1, T);

    foc_tol = 1e-4;

    % =========================================================================
    % Backward induction
    % =========================================================================
    for t = T:-1:1
        for nA = NA + 1:-1:1
            for nB = NB + 1:-1:1
                if nA < nB
                    prices_A(nA, nB, t) = prices_B(nB, nA, t);
                    prices_B(nA, nB, t) = prices_A(nB, nA, t);
                    DA_arr(nA, nB, t)   = DB_arr(nB, nA, t);
                    DB_arr(nA, nB, t)   = DA_arr(nB, nA, t);
                    CS_arr(nA, nB, t)   = CS_arr(nB, nA, t);
                    W(nA, nB, t)        = W(nB, nA, t);
                    dr(nA, nB, t)       = dr(nB, nA, t);
                    continue;
                end

                if nA == 1 && nB == 1
                    continue;
                end

                if nA == 1
                    dWB = W(nA, nB, t + 1) - W(nA, nB - 1, t + 1);
                    pB  = marginal_cost + dWB;
                    pA  = inf;

                    dB  = DA(pB, inf, a, gamma);
                    cs  = CSA(pB, inf, a, gamma);

                    prices_B(nA, nB, t) = pB;
                    DB_arr(nA, nB, t)   = dB;
                    CS_arr(nA, nB, t)   = cs;
                    W(nA, nB, t)        = W(nA, nB, t + 1) + L * cs;
                    continue;
                end

                if nB == 1
                    dWA = W(nA, nB, t + 1) - W(nA - 1, nB, t + 1);
                    pA  = marginal_cost + dWA;
                    pB  = inf;

                    dA  = DA(pA, inf, a, gamma);
                    cs  = CSA(pA, inf, a, gamma);

                    prices_A(nA, nB, t) = pA;
                    DA_arr(nA, nB, t)   = dA;
                    CS_arr(nA, nB, t)   = cs;
                    W(nA, nB, t)        = W(nA, nB, t + 1) + L * cs;
                    continue;
                end

                dWA = W(nA, nB, t + 1) - W(nA - 1, nB, t + 1);
                dWB = W(nA, nB, t + 1) - W(nA, nB - 1, t + 1);

                if nA == nB && abs(dWA - dWB) < 1e-10
                    pA = marginal_cost + dWA;
                    pB = pA;

                    sol.DA  = DA(pA, pB, a, gamma);
                    sol.DB  = sol.DA;
                    sol.CS  = CSA(pA, pB, a, gamma) + CSA(pB, pA, a, gamma);
                    sol.foc = [0; 0];
                    if sol.DA > 1e-12
                        sol.diversion_ratio = DA_B(pA, pB, a, gamma) / abs(DA_A(pA, pB, a, gamma));
                    else
                        sol.diversion_ratio = NaN;
                    end
                else

                    p0 = [max(marginal_cost + dWA, 0.1), max(marginal_cost + dWB, 0.1)];
                    if t < T
                        p0 = [finite_or(prices_A(nA, nB, t + 1), p0(1)), ...
                              finite_or(prices_B(nA, nB, t + 1), p0(2))];
                    end

                    [pA, pB, sol] = solve_period_planner_duopol(a, gamma, marginal_cost, dWA, dWB, p0);
                end

                foc_resid = max(abs(sol.foc));
                if ~isfinite(foc_resid) || foc_resid > foc_tol
                    error('solve_bellman_planner_duopol: FOC residual too large at nA=%d, nB=%d, t=%d.', nA, nB, t);
                end
                if nA == nB && abs(pA - pB) > 1e-8
                    error('solve_bellman_planner_duopol: asymmetry at symmetric state n=%d, t=%d (pA=%.10f, pB=%.10f).', ...
                        nA, t, pA, pB);
                end

                prices_A(nA, nB, t) = pA;
                prices_B(nA, nB, t) = pB;
                DA_arr(nA, nB, t)   = sol.DA;
                DB_arr(nA, nB, t)   = sol.DB;
                CS_arr(nA, nB, t)   = sol.CS;
                dr(nA, nB, t)       = sol.diversion_ratio;

                W(nA, nB, t) = W(nA, nB, t + 1) + ...
                    L * (sol.CS + sol.DA * (pA - marginal_cost - dWA) + sol.DB * (pB - marginal_cost - dWB));
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
        error('solve_bellman_planner_duopol: invalid state probabilities.');
    end
    if any(qA + qB > 1 + 1e-10, 'all') || any(qA < -1e-12, 'all') || any(qB < -1e-12, 'all')
        error('solve_bellman_planner_duopol: invalid transition probabilities.');
    end

    % =========================================================================
    % Summary statistics
    % =========================================================================
    vacant_seats_A = dot(0:NA, sum(prob(:, :, T), 2));
    vacancy_rate   = vacant_seats_A / NA;
    tickets_sold   = sum(prob .* (qA + qB), 'all');

    maskA = isfinite(prices_A);
    maskB = isfinite(prices_B);

    termA_rev = prob .* qA .* prices_A; termA_rev(~maskA) = 0;
    termB_rev = prob .* qB .* prices_B; termB_rev(~maskB) = 0;

    revenue_A     = sum(termA_rev, 'all');
    revenue_B     = sum(termB_rev, 'all');
    total_revenue = revenue_A + revenue_B;
    total_cs      = L * sum(prob .* CS_arr, 'all');
    total_ps      = total_revenue - tickets_sold * marginal_cost;
    total_welfare = W(NA + 1, NB + 1, 1);

    av_price = total_revenue / tickets_sold;
    relative_marginal_costs = marginal_cost / av_price;
    denom_qA = sum(prob .* qA, 'all');

    out.mode                    = 'planner';
    out.ps                      = total_ps;
    out.cs                      = total_cs;
    out.revenue                 = total_revenue;
    out.revenue_total           = total_revenue;
    out.revenue_A               = revenue_A;
    out.revenue_B               = revenue_B;
    out.av_price                = av_price;
    out.total_welfare           = total_welfare;
    out.vacancy_rate            = vacancy_rate * 100;
    out.prices_A                = prices_A;
    out.prices_B                = prices_B;
    out.prob                    = prob;
    out.welfare                 = W;
    out.valuation               = W;
    out.relative_marginal_costs = relative_marginal_costs;
    out.symmetry_gap_prices     = max(abs(prices_A - permute(prices_B, [2 1 3])), [], 'all', 'omitnan');
    out.symmetry_gap_revenue    = abs(revenue_A - revenue_B);

    if denom_qA > 1e-12
        out.diversion_ratio = sum(prob .* qA .* dr, 'all') / denom_qA;
    else
        out.diversion_ratio = NaN;
    end

    fprintf(['  Planner | W=%.3f | PS=%.3f | CS=%.3f' ...
        ' | RMC=%.3f | VR=%.1f%% | DR=%.2f%%\n'], total_welfare, ...
        total_ps, total_cs, relative_marginal_costs, vacancy_rate * 100, out.diversion_ratio * 100);
end

function [pA, pB, out] = solve_period_planner_duopol(a, gamma, c, dWA, dWB, p0)
    opts = optimoptions('fsolve', 'Display', 'off');
    sys = @(z) planner_foc(z, a, gamma, c, dWA, dWB);

    starts = [
        p0;
        max([c + dWA, c + dWB], 0.1);
        max([c + dWA, c + dWB], 0.1) + [1, 0];
        max([c + dWA, c + dWB], 0.1) + [0, 1]
    ];

    best_z   = [NaN; NaN];
    best_res = Inf;
    best_ef  = -Inf;

    for k = 1:size(starts, 1)
        start_k = starts(k, :).';
        [zc_k, ~, exitflag_k] = fsolve(sys, start_k, opts);
        if any(~isfinite(zc_k)), continue; end
        resid_k = norm(sys(zc_k), inf);
        if resid_k < best_res || (abs(resid_k - best_res) < 1e-10 && exitflag_k > best_ef)
            best_z = zc_k;
            best_res = resid_k;
            best_ef = exitflag_k;
        end
    end

    if any(~isfinite(best_z))
        error('solve_bellman_planner_duopol: planner fsolve failed for all starts.');
    end

    pA = best_z(1);
    pB = best_z(2);

    out.DA  = DA(pA, pB, a, gamma);
    out.DB  = DA(pB, pA, a, gamma);
    out.CS  = CSA(pA, pB, a, gamma) + CSA(pB, pA, a, gamma);
    out.foc = planner_foc([pA; pB], a, gamma, c, dWA, dWB);
    if out.DA > 1e-12
        out.diversion_ratio = DA_B(pA, pB, a, gamma) / abs(DA_A(pA, pB, a, gamma));
    else
        out.diversion_ratio = NaN;
    end
end

function G = planner_foc(z, a, gamma, c, dWA, dWB)
    pA = z(1);
    pB = z(2);

    G = zeros(2, 1);
    G(1) = DA_A(pA, pB, a, gamma) * (pA - c - dWA) + ...
           DA_B(pB, pA, a, gamma) * (pB - c - dWB);
    G(2) = DA_A(pB, pA, a, gamma) * (pB - c - dWB) + ...
           DA_B(pA, pB, a, gamma) * (pA - c - dWA);
end

function val = finite_or(candidate, fallback)
    if isfinite(candidate)
        val = candidate;
    else
        val = fallback;
    end
end

function val = DA(pA, pB, a, gamma)
    tau = tau_f(pA, pB, gamma);
    term_exp = exp(-a * pA + (a * gamma)^2 / 2);
    term_brackets = normcdf(pA / gamma - a * gamma) - normcdf(tau - a * gamma);
    term_last = normcdf(-pA / gamma);
    val = term_exp * term_brackets + term_last;
end

function val = DA_A(pA, pB, a, gamma)
    tau = tau_f(pA, pB, gamma);
    arg_pA = pA / gamma - a * gamma;
    arg_tau = tau - a * gamma;
    term_exp = -exp(-a * pA + (a * gamma)^2 / 2);
    term_bracket = a * (normcdf(arg_pA) - normcdf(arg_tau)) + ...
                   (1 / (2 * gamma)) * normpdf(arg_tau);
    val = term_exp * term_bracket;
end

function val = DA_B(pA, pB, a, gamma)
    tau = tau_f(pA, pB, gamma);
    z = tau - a * gamma;
    val = 1 / (2 * gamma) .* exp(-a .* pA + (a * gamma).^2 / 2) .* normpdf(z);
end

function F = CSA(pA, pB, a, gamma)
    F = 1 / a * DA(pA, pB, a, gamma) - pA * normcdf(-pA / gamma) + gamma * normpdf(pA / gamma);
end

function tau = tau_f(pA, pB, gamma)
    tau = (pA - pB) / (2 .* gamma);
end

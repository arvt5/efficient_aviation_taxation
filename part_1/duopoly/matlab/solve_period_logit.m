function [pA, pB, out] = solve_period_logit(alpha, cA, cB, a, p0, mode, cAA, cAB, cBB, cBA, c_phys)
    if nargin < 6 || isempty(mode), mode = 'incomplete'; end
    if nargin < 7 || isempty(cAA), cAA = 0; end
    if nargin < 8 || isempty(cAB), cAB = 0; end
    if nargin < 9 || isempty(cBB), cBB = 0; end
    if nargin < 10 || isempty(cBA), cBA = 0; end
    if nargin < 11 || isempty(c_phys), c_phys = min(cA, cB); end

    is_joint_monopoly = strcmp(mode, 'monopoly');
    complete_info     = strcmp(mode, 'complete');

    if ~(strcmp(mode, 'incomplete') || complete_info || is_joint_monopoly)
        error('solve_period_logit: unknown mode "%s".', mode);
    end

    if Di(cA, [cA, cB], alpha, a) < 1e-7
        pA = inf;
        pB = solve_single_firm_price(cB, alpha, a);
    elseif Di(cB, [cA, cB], alpha, a) < 1e-7
        pB = inf;
        pA = solve_single_firm_price(cA, alpha, a);
    else
        opts = optimoptions('fsolve', 'Display', 'off');
        if is_joint_monopoly
            sys = @(z) joint_monopoly_foc(z, alpha, a, c_phys, cAA, cAB, cBB, cBA);
        elseif complete_info
            sys = @(z) complete_info_foc(z, alpha, a, c_phys, cAA, cAB, cBB, cBA);
        else
            sys = @(z) incomplete_info_foc(z, cA, cB, alpha, a);
        end
        [zc, ~, exitflag, output] = fsolve(sys, p0, opts);
        if exitflag <= 0
            fprintf('fsolve did not converge: exitflag = %d\n', exitflag);
            disp(output.message);
        end
        pA = zc(1);
        pB = zc(2);
    end

    p = [pA, pB];
    sA = Di(pA, p, alpha, a);
    sB = Di(pB, p, alpha, a);
    out.DA = sA;
    out.DB = sB;
    out.WA = W(p, alpha, a);
    out.WB = 0;
    out.W  = W(p, alpha, a);

    if ~isfinite(pA) && isfinite(pB)
        out.FOC_A = 0;
        out.FOC_B = single_firm_foc(pB, cB, alpha, a);
    elseif isfinite(pA) && ~isfinite(pB)
        out.FOC_A = single_firm_foc(pA, cA, alpha, a);
        out.FOC_B = 0;
    elseif is_joint_monopoly
        g = joint_monopoly_foc([pA; pB], alpha, a, c_phys, cAA, cAB, cBB, cBA);
        out.FOC_A = g(1);
        out.FOC_B = g(2);
    elseif complete_info
        g = complete_info_foc([pA; pB], alpha, a, c_phys, cAA, cAB, cBB, cBA);
        out.FOC_A = g(1);
        out.FOC_B = g(2);
    elseif isfinite(pA) && sA > 1e-10
        out.FOC_A = cA + 1 / (alpha * (1 - sA)) - pA;
        out.FOC_B = cB + 1 / (alpha * (1 - sB)) - pB;
    else
        out.FOC_A = 0;
        out.FOC_B = 0;
    end

    if isfinite(pA) && sA > 1e-10
        out.SOC_AA = 2 * Di_i(pA, p, alpha, a) / sA;
    else
        out.SOC_AA = -1;
    end

    if isfinite(pB) && sB > 1e-10
        out.SOC_BB = 2 * Di_i(pB, p, alpha, a) / sB;
    else
        out.SOC_BB = -1;
    end

    if isfinite(pA) && isfinite(pB) && sA > 1e-10 && sB > 1e-10
        out.diversion_ratio = sB / (1 - sA);
        out.dA_B = Di_j(pA, p, alpha, a);
        out.dB_A = Di_j(pB, p, alpha, a);
    else
        out.diversion_ratio = 0;
        out.dA_B = 0;
        out.dB_A = 0;
    end
end

function G = incomplete_info_foc(z, cA, cB, alpha, a)
    G = zeros(2, 1);
    sA = Di(z(1), z, alpha, a);
    sB = Di(z(2), z, alpha, a);
    G(1) = cA + 1 / (alpha * (1 - sA)) - z(1);
    G(2) = cB + 1 / (alpha * (1 - sB)) - z(2);
end

function G = complete_info_foc(z, alpha, a, c, cAA, cAB, cBB, cBA)
    pA = z(1);
    pB = z(2);
    DA_val = Di(pA, z, alpha, a);
    DB_val = Di(pB, z, alpha, a);
    dA_A = Di_i(pA, z, alpha, a);
    dB_B = Di_i(pB, z, alpha, a);
    dA_B = Di_j(pA, z, alpha, a);
    dB_A = Di_j(pB, z, alpha, a);

    termA = DA_val + dA_A * (pA - (c + cAA)) - dB_A * cAB;
    termB = DB_val + dB_B * (pB - (c + cBB)) - dA_B * cBA;

    G = [termA / max(DA_val, 1e-12); termB / max(DB_val, 1e-12)];
end

function G = joint_monopoly_foc(z, alpha, a, c, cAA, cAB, cBB, cBA)
    pA = z(1);
    pB = z(2);
    DA_val = Di(pA, z, alpha, a);
    DB_val = Di(pB, z, alpha, a);
    dA_A = Di_i(pA, z, alpha, a);
    dB_B = Di_i(pB, z, alpha, a);
    dA_B = Di_j(pA, z, alpha, a);
    dB_A = Di_j(pB, z, alpha, a);

    cA_eff = c + cAA + cBA;
    cB_eff = c + cBB + cAB;

    termA = DA_val + dA_A * (pA - cA_eff) + dB_A * (pB - cB_eff);
    termB = DB_val + dB_B * (pB - cB_eff) + dA_B * (pA - cA_eff);

    G = [termA / max(DA_val, 1e-12); termB / max(DB_val, 1e-12)];
end

function val = h(z, c, alpha, a)
    val = c + 1 / (alpha * (1 - Di(z, [inf, z], alpha, a))) - z;
end

function p = solve_single_firm_price(c, alpha, a)
    x0 = max(c, 0) + 1 / alpha;
    opts = optimoptions('fsolve', 'Display', 'off');
    [zc, ~, exitflag] = fsolve(@(z) h(z, c, alpha, a), x0, opts);
    if exitflag <= 0 || ~isfinite(zc)
        error('solve_period_logit: single-firm fallback failed (exitflag=%d, c=%.4f).', exitflag, c);
    end
    p = zc;
end

function F = single_firm_foc(p, c, alpha, a)
    s = Di(p, [inf, p], alpha, a);
    F = c + 1 / (alpha * (1 - s)) - p;
end

function f = Di(pi, p, alpha, a)
    f = exp(a - alpha * pi) / (1 + exp(a - alpha * p(1)) + exp(a - alpha * p(2)));
end

function f = Di_i(pi, p, alpha, a)
    f = -alpha * Di(pi, p, alpha, a) * (1 - Di(pi, p, alpha, a));
end

function f = Di_j(~, p, alpha, a)
    f = alpha * Di(p(1), p, alpha, a) * Di(p(2), p, alpha, a);
end

function w = W(z, alpha, a)
    w = 1 / alpha * log(1 + exp(a - alpha * z(1)) + exp(a - alpha * z(2)));
end

function [pA, pB, out] = solve_exponential_model(a, gamma, c, cAA, cAB, cBB, cBA, p0, complete_info, is_joint_monopoly)
   cA = c + cAA;
   cB = c + cBB;

   if isinf(cA) && isinf(cB) % Both firms are out of stock
        pA = inf;
        pB = inf;
   elseif abs(DA(cB, cA, a, gamma)) < 1e-7 % B out of stock
         opts = optimset('Display', 'off', 'TolX', 1e-6);
         p0 = max(cA, 0.1) + 1/a;
         pA = fzero(@(z) FOC_A(z, Inf, a, gamma, cA), p0, opts);
         pB = inf;
   elseif abs(DA(cA, cB, a, gamma)) < 1e-7 % A out of stock
         opts = optimset('Display', 'off', 'TolX', 1e-6);
         p0 = max(cB, 0.1) + 1/a;
         pB = fzero(@(z) FOC_A(z, Inf, a, gamma, cB), p0, opts);
         pA = inf;
   elseif ~complete_info && ~is_joint_monopoly % Incomplete-info duopoly
        opts = optimoptions('fsolve', 'Display', 'off');

        [zc, ~, exitflag] = fsolve(@(z) best_response(z, a, gamma, cA, cB), p0, opts);
        if exitflag <= 0
            p0_ = [max(cA, 0.1); max(cB, 0.1)];
            [zc, ~, exitflag] = fsolve(@(z) best_response(z, a, gamma, cA, cB), p0_, opts);
        end

        if exitflag <= 0 || any(~isfinite(zc))
            error('solve_exponential_model: incomplete-info fsolve failed (exitflag=%d, cA=%.4f, cB=%.4f).', exitflag, cA, cB);
        end

        pA = zc(1);
        pB = zc(2);
   elseif complete_info && ~is_joint_monopoly % Complete-info duopoly
        opts = optimoptions('fsolve', 'Display', 'off');

        [zc, ~, exitflag] = fsolve(@(z) best_response_pi(z, a, gamma, c, cAA, cAB, cBB, cBA), p0, opts);
        if exitflag <= 0
            p0_ = [max(cA, 0.1); max(cB, 0.1)];
            [zc, ~, exitflag] = fsolve(@(z) best_response_pi(z, a, gamma, c, cAA, cAB, cBB, cBA), p0_, opts);
        end

        if exitflag <= 0 || any(~isfinite(zc))
            error('solve_exponential_model: complete-info fsolve failed (exitflag=%d, cA=%.4f, cB=%.4f).', exitflag, cA, cB);
        end

        pA = zc(1);
        pB = zc(2);
   elseif is_joint_monopoly % Two-product monopolist
        opts = optimoptions('fsolve', 'Display', 'off');

        [zc, ~, exitflag] = fsolve(@(z) joint_monopoly_foc(z, a, gamma, c, cAA, cAB, cBB, cBA), p0, opts);
        if exitflag <= 0 || any(~isfinite(zc))
            error('solve_exponential_model: joint-monopoly fsolve failed (exitflag=%d, cA=%.4f, cB=%.4f).', exitflag, cA, cB);
        end

        pA = zc(1);
        pB = zc(2);
   end

   opts1 = optimset('Display', 'off', 'TolX', 1e-8);
   if pA < 0 || pB < 0
       if pA < 0 && pB < 0
            pA = 0;
            pB = 0;
       elseif pA < 0
            pA = 0;
            % B's best response given pA = 0: swap args to use FOC_A by symmetry
            pB = max(fzero(@(z) FOC_A(z, 0, a, gamma, cB), max(cB, 0.1), opts1), 0);
       elseif pB < 0
            pB = 0;
            pA = max(fzero(@(z) FOC_A(z, 0, a, gamma, cA), max(cA, 0.1), opts1), 0);
       end
   end

    out.DA = DA(pA, pB, a, gamma);
    out.DB = DA(pB, pA, a, gamma);
    if is_joint_monopoly
        g = joint_monopoly_foc([pA; pB], a, gamma, c, cAA, cAB, cBB, cBA);
        out.FOC_A = g(1);
        out.FOC_B = g(2);
    elseif complete_info
        out.FOC_A = FOC_A_pi(pA, pB, a, gamma, cAA, cAB, c);
        out.FOC_B = FOC_A_pi(pB, pA, a, gamma, cBB, cBA, c);
    else
        out.FOC_A = FOC_A(pA, pB, a, gamma, cA);
        out.FOC_B = FOC_A(pB, pA, a, gamma, cB);
    end
    out.WA = CSA(pA, pB, a, gamma);
    out.WB = CSA(pB, pA, a, gamma);

    out.SOC_AA = -1;
    out.SOC_BB = -1;
    out.diversion_ratio = DA_B(pA, pB, a, gamma) / abs(DA_A(pA, pB, a, gamma));

    % ================================================================
    % Check Assumption 19: Strategic complements & Equilibrium stability
    % ================================================================
    if isfinite(pA) && isfinite(pB)
        dA_val  = DA(pA, pB, a, gamma);
        dB_val  = DA(pB, pA, a, gamma);
        dA_A    = DA_A(pA, pB, a, gamma);
        dA_B    = DA_B(pA, pB, a, gamma);
        dB_A    = DA_B(pB, pA, a, gamma);
        dB_B    = DA_A(pB, pA, a, gamma);
        dA_AA   = DA_AA(pA, pB, a, gamma);
        dA_AB   = DA_AB(pA, pB, a, gamma);
        dB_BB   = DA_AA(pB, pA, a, gamma);
        dB_AB   = DA_AB(pB, pA, a, gamma);

        % Jacobian of the FOC system at equilibrium (FOC_i = 0).
        % dF_A/dp_A = [2*D_{A,A} + D_{A,AA}*(p_A - c_A)] / D_A
        % dF_A/dp_B = [D_{A,B}   + D_{A,AB}*(p_A - c_A)] / D_A
        J11 = (2*dA_A  + dA_AA * (pA - cA)) / dA_val;
        J12 = (dA_B    + dA_AB * (pA - cA)) / dA_val;
        J21 = (dB_A    + dB_AB * (pB - cB)) / dB_val;
        J22 = (2*dB_B  + dB_BB * (pB - cB)) / dB_val;

        out.J = [J11, J12; J21, J22];

        out.SOC_AA = J11;
        out.SOC_BB = J22;
        out.SOC_holds = (J11 < 0) && (J22 < 0);
        out.dA_A = dA_A;
        out.dA_B = dA_B;
        out.dB_A = dB_A;

        out.BR_slope_A = -J12 / J11;
        out.BR_slope_B = -J21 / J22;
        out.strategic_complements = (J12 > 0) && (J21 > 0);

        out.det_J = J11 * J22 - J12 * J21;
        out.stable_equilibrium = (out.det_J > 0);

        out.BR_slope_product = out.BR_slope_A * out.BR_slope_B;
        out.contraction = (abs(out.BR_slope_product) < 1);

        out.assumption_19 = out.SOC_holds && out.stable_equilibrium && out.strategic_complements;

        if ~out.assumption_19 && ~is_joint_monopoly
            warning('solve_exponential_model:Assumption19', ...
                'solve_exponential_model: Assumption 19 violated at pA=%.4f, pB=%.4f (a=%.4f, gamma=%.4f).', ...
                pA, pB, a, gamma);
        end
    else
        out.SOC_AA = NaN;
        out.SOC_BB = NaN;
        out.SOC_holds = NaN;
        out.J = NaN(2);
        out.det_J = NaN;
        out.strategic_complements = NaN;
        out.stable_equilibrium = NaN;
        out.BR_slope_A = NaN;
        out.BR_slope_B = NaN;
        out.BR_slope_product = NaN;
        out.contraction = NaN;
        out.assumption_19 = NaN;
        out.dA_A = NaN;
        out.dA_B = NaN;
        out.dB_A = NaN;
    end

end

function G = best_response(z, a, gamma, cA, cB)
    pA = z(1);
    pB = z(2);
    F1 = FOC_A(pA, pB, a, gamma, cA);
    F2 = FOC_A(pB, pA, a, gamma, cB);
    G = [F1; F2];
end

function G = best_response_pi(z, a, gamma, c, cAA, cAB, cBB, cBA)
    pA = z(1);
    pB = z(2);
    F1 = FOC_A_pi(pA, pB, a, gamma, cAA, cAB, c);
    F2 = FOC_A_pi(pB, pA, a, gamma, cBB, cBA, c);
    G = [F1; F2];
end

function G = joint_monopoly_foc(z, a, gamma, c, cAA, cAB, cBB, cBA)
    pA = z(1);
    pB = z(2);

    DA_val = DA(pA, pB, a, gamma);
    DB_val = DA(pB, pA, a, gamma);

    cA_eff = c + cAA + cBA;
    cB_eff = c + cBB + cAB;

    termA = DA_val + DA_A(pA, pB, a, gamma)*(pA - cA_eff) + DA_B(pB, pA, a, gamma)*(pB - cB_eff);
    F1 = termA / DA_val;

    termB = DB_val + DA_A(pB, pA, a, gamma)*(pB - cB_eff) + DA_B(pA, pB, a, gamma)*(pA - cA_eff);
    F2 = termB / DB_val;

    G = [F1; F2];
end

% ---- Profit FOCs
function F = FOC_A(pA, pB, a, gamma, cA)
    term1 = DA(pA, pB, a, gamma) + DA_A(pA, pB, a, gamma).*(pA-cA);
    F = term1/DA(pA, pB, a, gamma);
end

function F = FOC_A_pi(pA, pB, a, gamma, cAA, cAB, c)
    % Complete-info FOC: includes cross-externality term cAB * D_{A,B}(pB, pA)
    term1 = DA(pA, pB, a, gamma) + DA_A(pA, pB, a, gamma).*(pA-(c+cAA)) - DA_B(pB, pA, a, gamma)*cAB;
    F = term1/DA(pA, pB, a, gamma);
end

% ---- Demand
function val = DA(pA, pB, a, gamma)
    tau = tau_f(pA, pB, gamma);
    term_exp = exp(-a * pA + (a * gamma)^2 / 2);
    term_brackets = normcdf(pA / gamma - a * gamma) - normcdf(tau - a * gamma);
    term_last = normcdf(-pA / gamma);
    val = term_exp * term_brackets + term_last;
end

% ---- First order partials demand

function val = DA_A(pA, pB, a, gamma)
    tau = tau_f(pA, pB, gamma);
    arg_pA = pA / gamma - a * gamma;
    arg_tau = tau - a * gamma;
    term_exp = -exp(-a * pA + (a * gamma)^2 / 2);
    term_bracket = a * (normcdf(arg_pA) - normcdf(arg_tau)) + ...
                   (1 / (2 * gamma)) * normpdf(arg_tau);
    val = term_exp * term_bracket;
end

function F = DA_B(pA, pB, a, gamma)
    tau = tau_f(pA, pB, gamma);
    z = tau - a*gamma;
    F = 1/(2*gamma) .* exp(-a.*pA + (a*gamma).^2/2) * normpdf(z);
end

% ---- Second order partials demand

function result = DA_AA(pA, pB, a, gamma)
    term1_inner = (pA ./ gamma) - (a .* gamma);
    term2_inner = ((pA - pB) ./ (2 .* gamma)) - (a .* gamma);

    outer_exp = exp(-a .* pA + ((a .* gamma).^2) ./ 2);

    part1 = (a.^2) .* (normcdf(term1_inner) - normcdf(term2_inner));
    part2 = (a ./ gamma) .* normpdf(term1_inner);

    numerator = 6 .* a .* (gamma.^2) + pA - pB;
    denominator = 8 .* (gamma.^3);
    part3 = (numerator ./ denominator) .* normpdf(term2_inner);

    result = outer_exp .* (part1 - part2 + part3);
end

function result = DA_AB(pA, pB, a, gamma)
    inner_term = ((pA - pB) ./ (2 .* gamma)) - (a .* gamma);

    numerator = 2 .* a .* (gamma.^2) + pA - pB;
    denominator = 8 .* (gamma.^3);
    fraction_part = -(numerator ./ denominator);

    exp_part = exp(-a .* pA + ((a .* gamma).^2) ./ 2);
    phi_part = normpdf(inner_term);

    result = fraction_part .* exp_part .* phi_part;
end

% ---- Consumer surplus

function F = CSA(pA, pB, a, gamma)
    F = 1/a * DA(pA, pB, a, gamma) - pA*normcdf(-pA/gamma) + gamma*normpdf(pA/gamma);
end

% ---- Helper functions

function tau = tau_f(pA, pB, gamma)
    tau = (pA - pB)/(2.*gamma);
end

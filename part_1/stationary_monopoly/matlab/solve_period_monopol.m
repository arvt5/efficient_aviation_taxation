function [p_star, out] = solve_period_monopol(demand_handle, deriv_handle, surplus_handle, params, c_eff, p0, t)
    % solve_period_monopol: Finds the optimal price using exponential transitions.
    % t: periods remaining (passed through to demand handles for time-varying specs)

    foc = @(p) foc_objective(p, demand_handle, deriv_handle, params, c_eff, t);

    opts = optimset('Display', 'off', 'TolX', 1e-8);

    p_min = max(c_eff, 1e-6);   % lower bracket: p* > c_eff, avoids overflow for large elasticities
    p_max = 1e6;

    try
        f_min = foc(p_min);
        f_max = foc(p_max);
        if f_min * f_max < 0
            p_star = fzero(foc, [p_min, p_max], opts);
        else
            p_star = fzero(foc, p0, opts);
        end
        if isnan(p_star)
            error('solve_period_monopol: fzero returned NaN (c_eff=%.4f, p0=%.4f, t=%d).', c_eff, p0, t);
        end
    catch e
        error('solve_period_monopol: fzero failed (c_eff=%.4f, p0=%.4f, t=%d): %s', c_eff, p0, t, e.message);
    end

    out.D = demand_handle(p_star, t, params);
    out.S = surplus_handle(p_star, t, params);
end

function val = foc_objective(p, dh, deh, params, c_eff, t)
    D = dh(p, t, params);
    Dp = deh(p, t, params);
    val = Dp * (p - c_eff) + D;
end

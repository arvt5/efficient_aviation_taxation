function [ar, p_posted_t] = compute_ar(out, T)
% compute_ar  Price ratio: mean posted price near departure / far from departure.
%   Weights prices by the probability of being in each inventory state.
%   Excludes the sold-out state (where price is infinite).
%
%   out        : output struct from solve_bellman_monopol or solve_bellman_planner
%   T          : horizon length
%   p_posted_t : [1 x T] probability-weighted average posted price per period

    t_early = 1 : max(1, floor(0.1 * T));
    t_late  = min(T, ceil(0.9 * T)) : T;

    p    = out.prices;
    % Use only the first T columns of prob to match prices dimension
    prob = out.prob(:, 1:T);

    % Filter to finite prices (excludes n=1 sold-out state)
    mask = isfinite(p);

    % Calculate expected posted price per period
    % Only include states where a price is actually posted (mask)
    p_finite = p;
    p_finite(~mask) = 0;

    weighted_p = sum(prob .* p_finite, 1);
    weights    = sum(prob .* mask, 1);

    p_posted_t = weighted_p ./ weights;

    % Handle periods with zero probability of being open (all seats sold)
    p_posted_t(weights < 1e-12) = NaN;

    % Calculate AR using mean over windows (ignoring NaNs if horizon is sold out)
    mean_early = mean(p_posted_t(t_early), 'omitnan');
    mean_late  = mean(p_posted_t(t_late), 'omitnan');

    ar = mean_late / mean_early;
end

function results = recompute_duopoly(model, refit)
% Full-information, fixed-flight ticket-tax replacement experiment.
% False: evaluate archived calibration and sweep. True: also refit the moments.
    if nargin < 2, refit = false; end
    assert(ismember(model, {'exponential','logit'}), 'Unknown demand model');
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    rows = readtable(fullfile(here,'..','data',[model,'.csv']), 'TextType','string');
    base = rows(rows.policy_case == "current_ticket_tax",:);
    N = 10; T = 500;
    ticket = base.ticket_tax_status_quo;
    if strcmp(model,'exponential')
        z = [base.a_fit,base.c_fit,base.lambda_fit,base.gamma_fit];
        unpack = @(x) [x(1),1,x(2),x(3)];
        initial = z([1,3,4]);
        lower = [1e-3,1e-3,1e-3];
        solver = @solve_bellman_duopol;
        planner = @solve_bellman_planner_duopol;
    else
        z = [base.alpha_fit,base.a_fit,base.c_fit,base.lambda_fit];
        unpack = @(x) [x(1),x(2),1,x(3)];
        initial = z([1,2,4]);
        lower = [1e-4,1e-4,1];
        solver = @solve_bellman_logit;
        planner = @solve_bellman_planner_logit;
    end
    if refit
        options = optimoptions('lsqnonlin','Display','iter', ...
            'FunctionTolerance',1e-5,'StepTolerance',1e-6);
        [fit,~,~,flag] = lsqnonlin(@(x) moments(solver(unpack(x),N,N,T,0,'complete',ticket)), ...
            initial,lower,[],options);
        assert(flag>0, 'Calibration did not converge');
        z = unpack(fit);
    end
    solve = @(tau,tt) solver(z,N,N,T,tau,'complete',tt);
    results.z = z;
    results.current_ticket_tax = solve(0,ticket);
    results.undistorted_baseline = solve(0,0);
    results.planner = planner(z,N,N,T);
    taus = rows.tau(rows.policy_case == "vacancy_tax_sweep");
    W = zeros(size(taus));
    VR = zeros(size(taus));
    for j = 1:numel(taus)
        point = solve(taus(j),0);
        W(j) = point.total_welfare; VR(j) = point.vacancy_rate;
    end
    [~,j] = max(W);
    lo = max(0,taus(j)-.25); hi = min(6,taus(j)+.25);
    tau = fminbnd(@(t) welfare_objective(solve,t),lo,hi);
    results.tau_star = tau;
    results.optimum = solve(tau,0);
    results.sweep = table(taus,W,VR);
    folder = fullfile(here,'..','outputs');
    if ~exist(folder,'dir'), mkdir(folder); end
    save(fullfile(folder,[model,'_recomputed.mat']),'-struct','results');
    if ~refit
        assert(abs(results.current_ticket_tax.total_welfare-base.total_welfare) < 1e-6, ...
            'Archived baseline welfare mismatch');
        archived = rows(rows.policy_case == "vacancy_tax_sweep",:);
        assert(max(abs(W-archived.total_welfare)) < 1e-6, 'Archived sweep mismatch');
    end
end

function G = moments(out)
    G = [out.vacancy_rate/100-.2, out.relative_marginal_costs-.2, out.diversion_ratio-.3];
end

function value = welfare_objective(solve,tau)
    out = solve(tau,0);
    value = -out.total_welfare;
end

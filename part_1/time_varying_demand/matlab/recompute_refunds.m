function results = recompute_refunds(model, refit)
% Re-evaluate the archived calibration; optionally recalibrate and optimize.
% Run here: recompute_refunds('ces') or recompute_refunds('exponential', true).
    if nargin < 2, refit = false; end
    assert(ismember(model, {'ces', 'exponential'}), 'Unknown demand model');
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    data = load(fullfile(here, '..', 'data', [model, '.mat']));
    N = size(data.out_base.prices,1)-1;
    T = size(data.out_base.prices,2);
    mu = data.mu_bar;
    if strcmp(model, 'ces')
        c = @(z) z(1);
        elasticity = @(t,z) z(2) + z(3)*(1-t/T);
        dh = @(p,t,z) N/T*p.^(-elasticity(t,z));
        deh = @(p,t,z) -N/T*elasticity(t,z).*p.^(-elasticity(t,z)-1);
        sh = @(p,t,z) N/T/(elasticity(t,z)-1)*p.^(1-elasticity(t,z));
        psh = @(t,z,d) (N/T/max(1e-10,d))^(1/elasticity(t,z));
        pf = @(eff,t,z) elasticity(t,z).*eff./(elasticity(t,z)-1);
        lower = [1e-3, 1+1e-3, 1e-3];
    else
        c = @(z) 1;
        slope = @(t,z) z(2) + z(3)*(1-t/T)^2;
        dh = @(p,t,z) z(1)*exp(-slope(t,z)*p);
        deh = @(p,t,z) -z(1)*slope(t,z)*exp(-slope(t,z)*p);
        sh = @(p,t,z) z(1)/slope(t,z)*exp(-slope(t,z)*p);
        psh = @(t,z,d) max(0,-log(max(1e-10,d/z(1)))/slope(t,z));
        pf = @(eff,t,z) eff + 1/slope(t,z);
        lower = [1e-3, 1e-3, 1e-3];
    end
    solve = @(z,tau,xi,hazard) solve_bellman_return(z,N,T,tau,xi,hazard, ...
        dh,deh,sh,psh,c(z),pf);
    z = data.z_fit;
    xi = double(data.xi_opt); tau = double(data.tau_opt); tau_only = double(data.tau_opt_mu);
    if refit
        options = optimoptions('lsqnonlin','Display','iter','FunctionTolerance',1e-6);
        [z,~,~,flag] = lsqnonlin(@(z) moments(solve(z,0,0,0),T), z, lower, [], options);
        assert(flag>0, 'Calibration did not converge');
        options = optimoptions('fmincon','Display','iter','Algorithm','sqp', ...
            'StepTolerance',1e-5,'FunctionTolerance',1e-5);
        [policy,~,flag] = fmincon(@(p) objective(solve(z,p(2),p(1),mu)), ...
            [data.xi_opt,data.tau_opt],[],[],[],[],[0,0],[1,5],[],options);
        assert(flag>0, 'Policy optimization did not converge');
        xi = policy(1); tau = policy(2);
        tau_only = fminbnd(@(t) objective(solve(z,t,0,mu)),0,5);
    end
    results.z_fit = z;
    results.mu_bar = mu;
    results.xi_opt = xi; results.tau_opt = tau; results.tau_opt_mu = tau_only;
    results.out_base = solve(z,0,0,0);
    results.out_base_mu = solve(z,0,0,mu);
    results.out_tau_mu = solve(z,tau_only,0,mu);
    results.out_opt = solve(z,tau,xi,mu);
    results.out_plan = solve_bellman_planner_return(z,N,T,mu,dh,sh,c(z));
    results.gap_closed = (results.out_opt.total_welfare-results.out_base_mu.total_welfare) ...
        /(results.out_plan.total_welfare-results.out_base_mu.total_welfare);
    folder = fullfile(here,'..','outputs');
    if ~exist(folder,'dir'), mkdir(folder); end
    save(fullfile(folder,[model,'_recomputed.mat']),'-struct','results');
    if ~refit
        for key = {'out_base','out_base_mu','out_tau_mu','out_opt','out_plan'}
            actual = results.(key{1}).total_welfare;
            expected = data.(key{1}).total_welfare;
            assert(abs(actual-expected) < 1e-7*max(1,abs(expected)), ...
                'Archived welfare mismatch for %s',key{1});
        end
    end
    fprintf('Gap closed: %.4f%%\n',100*results.gap_closed);
end

function residual = moments(out,T)
    residual = [out.vacancy_rate-.2, out.relative_marginal_costs-.2, compute_ar(out,T)-2];
end

function value = objective(out)
    if out.vacancy_rate > .99 || ~isfinite(out.total_welfare)
        value = 1e6;
    else
        value = -out.total_welfare;
    end
end

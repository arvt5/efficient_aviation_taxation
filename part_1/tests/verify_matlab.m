function verify_matlab()
% Re-solve archived policies, without fitting parameters or selecting new taxes.
    root = fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'time_varying_demand','matlab'));
    addpath(fullfile(root,'duopoly','matlab'));
    addpath(fullfile(root,'stationary_monopoly','matlab'));
    z = [.2,1.2];
    demand = @(p,t,z) p.^(-z(1));
    surplus = @(p,t,z) p.^(1-z(1))/(z(1)-1);
    price = @(c,t,z) z(1)*c/(z(1)-1);
    taxed = solve_bellman_monopol(z,3,200,z(1)/z(2),demand,[],surplus,[],price);
    planner = solve_bellman_planner(z,3,200,demand,surplus,[]);
    active = isfinite(taxed.prices) & isfinite(planner.prices);
    assert(max(abs(taxed.prices(active)-planner.prices(active)))<1e-8, ...
        'CES policy alignment failed');
    for model = {'ces','exponential'}
        recompute_refunds(model{1});
    end
    for model = {'exponential','logit'}
        rows = readtable(fullfile(root,'duopoly','data',[model{1},'.csv']));
        base = rows(strcmp(rows.policy_case,'current_ticket_tax'),:);
        if strcmp(model{1},'exponential')
            z = [base.a_fit,base.c_fit,base.lambda_fit,base.gamma_fit];
            solver = @solve_bellman_duopol;
        else
            z = [base.alpha_fit,base.a_fit,base.c_fit,base.lambda_fit];
            solver = @solve_bellman_logit;
        end
        out = solver(z,10,10,500,0,'complete',base.ticket_tax_status_quo);
        assert(abs(out.total_welfare-base.total_welfare)<1e-6, 'Welfare mismatch');
        assert(abs(out.vacancy_rate-base.vacancy_rate)<1e-6, 'Vacancy mismatch');
        assert(abs(out.av_price-base.av_price)<1e-6, 'Price mismatch');
    end
    disp('Archived refund policies and duopoly baselines verified.');
end

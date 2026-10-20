function results = recompute_ces(N,T)
% Calibrate stationary CES demand at the chosen capacity and time resolution.
% This is the recovered numerical method, not a recovered published calibration.
    if nargin < 1, N = 100; end
    if nargin < 2, T = 20000; end
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    dh = @(p,t,z) p.^(-z(1));
    sh = @(p,t,z) p.^(1-z(1))/(z(1)-1);
    pf = @(c,t,z) z(1)*c/(z(1)-1);
    options = optimoptions('lsqnonlin','Display','iter', ...
        'FunctionTolerance',1e-10,'StepTolerance',1e-10);
    [z,~,~,flag] = lsqnonlin(@(z) calib_2moments(z,N,T,.2,.2,dh,[],sh,[],pf), ...
        [.2,1.2],[1e-3,1+1e-3],[],options);
    assert(flag>0, 'Calibration did not converge');
    tau = z(1)/z(2);
    results.N = N; results.T = T; results.z = z; results.tau = tau;
    results.baseline = solve_bellman_monopol(z,N,T,0,dh,[],sh,[],pf);
    results.optimum = solve_bellman_monopol(z,N,T,tau,dh,[],sh,[],pf);
    results.planner = solve_bellman_planner(z,N,T,dh,sh,[]);
    results.gain_relative_to_welfare = ...
        (results.optimum.total_welfare-results.baseline.total_welfare)/results.baseline.total_welfare;
    results.gain_relative_to_revenue = ...
        (results.optimum.total_welfare-results.baseline.total_welfare)/results.baseline.expected_revenue;
    folder = fullfile(here,'..','outputs');
    if ~exist(folder,'dir'), mkdir(folder); end
    save(fullfile(folder,'ces_recomputed.mat'),'-struct','results');
end

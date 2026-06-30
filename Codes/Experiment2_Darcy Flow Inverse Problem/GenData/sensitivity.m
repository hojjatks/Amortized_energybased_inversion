clc,clear all,close all

%% constants
Nsol=64;
Nobs=8;
alpha = 2;
tau = 3;
F = ones(Nsol,1);
eps=1e-5;
%% Loading the true parameters:
data = load('./../../../Data/Experiment3/darcy_data1D_64.mat');

x_true = data.x;
y_notnoisy = data.y;
x_true = x_true(1,:); % we have used the first iteration to study
plot(x_true)
%% Building eigenfunctions.

[x_spatial, Phi, lambdas] = build_phi_lambda(Nsol, tau, alpha);
%%
Nmodes = Nsol-1; 
sens = zeros(Nmodes, Nobs);   % or length(observations) if you sub-sample
sens_norm= zeros(Nmodes,1);

for i = 1:Nmodes

    phi_i = Phi(:, i)';          % Phi is (64 × 63), x_true is (1 × 64)
                                 % so transpose phi_i to match row shape

    % construct perturbed fields (remember to exponentiate)
    coef_plus  = exp( x_true + eps * phi_i );
    coef_minus = exp( x_true - eps * phi_i );

    % forward evaluations
    y_plus  = solve_gwf_1D(coef_plus,  F);
    y_minus = solve_gwf_1D(coef_minus, F);

    % sensitivity
    sens(i, :) = (y_plus(4:8:end) - y_minus(4:8:end)) / (2*eps);
    sens_norm(i) = norm(sens(i, :), 2);
end

%% plotting
plot(1:Nmodes, sens_norm, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;

xlabel('Eigenmode index i', 'Interpreter', 'latex', 'FontSize', 14);

ylabel(['$\left\| s_i \right\|_2 = \left\| \frac{ G(u_{\mathrm{true}} + \epsilon \phi_i )' ...
        ' - G(u_{\mathrm{true}} - \epsilon \phi_i ) }{2 \epsilon} \right\|_2 $'], ...
        'Interpreter', 'latex', 'FontSize', 16);

title('Sensitivity of Observations to Each Eigenmode', ...
      'Interpreter', 'latex', 'FontSize', 16);

set(gca, 'FontSize', 14);
saveas(gcf, 'sensitivity_plot.png');







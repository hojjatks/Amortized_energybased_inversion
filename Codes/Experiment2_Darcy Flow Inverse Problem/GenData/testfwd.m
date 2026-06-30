% --------------------------------------------------------------
% TEST 5: High-contrast diffusion + oscillatory solution
% --------------------------------------------------------------
fprintf('\nTest 5: high-contrast a(x) and oscillatory p(x)\n');

% Define exact p(x) and a(x)
syms xs
a_sym  = 1 + 99*xs^2;
p_sym  = sin(5*pi*xs) * xs * (1-xs);
pprime = diff(p_sym, xs);
app    = a_sym * pprime;
f_sym  = -diff(app, xs);

% Convert to MATLAB functions
a_fun = matlabFunction(a_sym,  'Vars', xs);
p_fun = matlabFunction(p_sym,  'Vars', xs);
f_fun = matlabFunction(f_sym,  'Vars', xs);

% choose resolution
K = 500;
x_cc = (1/(2*K) : 1/K : (2*K-1)/(2*K))';

% generate input coef and F on cell centers
coef = a_fun(x_cc);
F    = f_fun(x_cc);
p_ex = p_fun(x_cc);

% numerical solution
p_num = solve_gwf_1D(coef, F);

% error
err = max(abs(p_num - p_ex));
fprintf('  max error = %.3e\n', err);

% plot
figure; plot(x_cc, p_ex, 'k-', 'LineWidth', 2); hold on;
plot(x_cc, p_num, 'ro-');
title('Test 5: high-contrast diffusion + oscillatory solution');
legend('Exact','Numerical');

function test_solver()

    % Choose resolution
    K = 50;
    x = (1/(2*K) : 1/K : (2*K-1)/(2*K))';   % cell-centered input grid

    % ------------------------------
    % TEST 1: a = 1, p = x(1-x)
    % ------------------------------
    a_fun = @(x) 1 + 0*x;
    p_fun = @(x) x.*(1-x);
    f_fun = @(x) 2 + 0*x;   % constant RHS

    run_test('Test 1', a_fun, p_fun, f_fun, K);

    % ------------------------------
    % TEST 2: a = 1+x, p = sin(pi x)
    % ------------------------------
    a_fun = @(x) 1 + x;
    p_fun = @(x) sin(pi*x);
    f_fun = @(x) -pi*cos(pi*x) + pi^2*(1+x).*sin(pi*x);

    run_test('Test 2', a_fun, p_fun, f_fun, K);

    % ------------------------------
    % TEST 3: a = exp(3x), p = x^2(1-x)
    % ------------------------------
    a_fun = @(x) exp(3*x);
    p_fun = @(x) x.^2 .* (1-x);
    f_fun = @(x) (9*x.^2 - 2) .* exp(3*x);

    run_test('Test 3', a_fun, p_fun, f_fun, K);

    % ------------------------------
    % TEST 4: a = 0.1 + x^2, p = sin(2pi x)
    % ------------------------------
    a_fun = @(x) 0.1 + x.^2;
    p_fun = @(x) sin(2*pi*x);
    f_fun = @(x) -2*pi*((0.1+x.^2).*(-2*pi*sin(2*pi*x)) + 2*x.*cos(2*pi*x));

    run_test('Test 4', a_fun, p_fun, f_fun, K);
end


function run_test(name, a_fun, p_fun, f_fun, K)

    fprintf('\n%s\n', name);

    % Cell-centered grid (input grid)
    x_cc = (1/(2*K) : 1/K : (2*K-1)/(2*K))';

    % Generate exact data
    coef = a_fun(x_cc);
    F    = f_fun(x_cc);
    p_ex = p_fun(x_cc);

    % Solve
    p_num = solve_gwf_1D(coef, F);

    % Error
    err = max(abs(p_num - p_ex));
    fprintf('  max error = %.3e\n', err);

    % Plot
    figure; plot(x_cc, p_ex, 'k-', 'LineWidth',2); hold on;
    plot(x_cc, p_num, 'ro-');
    title(name); legend('Exact','Numerical');
end

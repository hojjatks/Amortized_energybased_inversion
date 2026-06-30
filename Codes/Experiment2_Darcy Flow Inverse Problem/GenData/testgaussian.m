    % =====================================================
    % Verify gaussrnd(alpha, tau, N) against theoretical
    % KL covariance for C = (-Delta + tau^2)^(-alpha)
    % with Neumann BC, evaluated at cell centers.
    %
    % Your sampler:
    %   xi_k ~ N(0,1)
    %   coef_k = (pi^2 k^2 + tau^2)^(-alpha/2)
    %   L_k = sqrt(N) * coef_k * xi_k
    %   L(1) = 0  (remove constant mode)
    %   U = idct(L, 'Type', 2)
    %
    % Theoretical KL covariance:
    %   x_i = (i - 0.5)/N
    %   phi_0(x) = 1
    %   phi_k(x) = sqrt(2) cos(k*pi*x),  k>=1
    %   lambda_k = (pi^2 k^2 + tau^2)^(-alpha), lambda_0 = 0
    %   C_th(i,j) = sum_k lambda_k phi_k(x_i) phi_k(x_j)
    % =====================================================

    clear; clc;

    alpha = 2;
    tau   = 5;
    N     = 64;
    Nsamp = 50000;   % number of Monte Carlo samples

    % -----------------------------------------------------
    % 1. Grid: cell centers x_i = (i - 0.5)/N
    % -----------------------------------------------------
    i = 1:N;
    x = (i - 0.5) / N;      % 1×N

    % -----------------------------------------------------
    % 2. Generate samples using your gaussrnd
    % -----------------------------------------------------
    U_samples = zeros(N, Nsamp);
    for n = 1:Nsamp
        U_samples(:, n) = gaussernd(alpha, tau, N).';
    end

    % Empirical covariance (N×N)
    C_emp = cov(U_samples.');

    % -----------------------------------------------------
    % 3. Theoretical covariance C_th (KL formula)
    % -----------------------------------------------------

    % Eigenvalues lambda_k = (pi^2 k^2 + tau^2)^(-alpha)
    k = 0:N-1;
    lambda = (pi^2 * k.^2 + tau^2).^(-alpha);

    % Remove constant mode to match L(1) = 0 in gaussrnd
    lambda(1) = 0;

    % Build Phi(i,k) = phi_k(x_i)
    Phi = zeros(N, N);

    % k = 0 mode: phi_0(x) = 1
    Phi(:, 1) = 1;

    % k >= 1 modes: phi_k(x) = sqrt(2) cos(k*pi*x)
    for kk = 2:N
        Phi(:, kk) = sqrt(2) * cos((kk-1) * pi * x.');
    end

    % THEORETICAL covariance:
    %   C_th(i,j) = sum_k lambda_k phi_k(x_i) phi_k(x_j)
    C_th = Phi * diag(lambda) * Phi.';   % <-- NO /N HERE

    % -----------------------------------------------------
    % 4. Compare empirical vs theoretical
    % -----------------------------------------------------
    figure;
    subplot(1,3,1)
    imagesc(C_emp)
    axis equal tight
    title('Empirical Covariance')
    colorbar

    subplot(1,3,2)
    imagesc(C_th)
    axis equal tight
    title('Theoretical C\_th (KL)')
    colorbar

    subplot(1,3,3)
    imagesc(C_emp - C_th)
    axis equal tight
    title('Difference: Emp - C\_th')
    colorbar

    % Relative Frobenius error
    err = norm(C_emp - C_th, 'fro') / norm(C_th, 'fro');
    fprintf("Relative Frobenius error (Emp vs C_th) = %.3e\n", err);

    % Check overall scale agreement
    s = trace(C_emp) / trace(C_th);
    fprintf("Trace ratio trace(C_emp)/trace(C_th) = %.6f (should be ~1)\n", s);

    % A single variance comparison
    idx = 10;
    fprintf("Var at i=%d: Emp = %.4e, Th = %.4e\n", ...
        idx, C_emp(idx,idx), C_th(idx,idx));

    % Mean sanity check
    fprintf("Mean of u(%d): %.3e (should be ~0)\n", idx, mean(U_samples(idx,:)));
function [x_spatial, Phi, lambdas] = build_phi_lambda(Nsol, tau, alpha)

    % grid points x_j = (2j - 1)/(2 Nsol), j = 1..Nsol
    j = (1:Nsol);
    x_spatial = (2*j - 1) / (2*Nsol);     % 1 × Nsol

    % mode indices i = 1..Nsol
    i = (1:Nsol);

    % Phi(j,i) = sqrt(2) * cos(i * pi * x_j)
    % Use outer product (Nsol × Nsol)
    Phi_full = sqrt(2) * cos(pi * (x_spatial(:)) * i);

    % lambda_i = ((i*pi)^2 + tau^2)^(-alpha)
    lambdas_full = ((i*pi).^2 + tau^2).^(-alpha);

    % drop the last mode exactly as Python: [:, :-1]
    Phi = Phi_full(:, 1:end-1);          % Nsol × (Nsol-1)
    lambdas = lambdas_full(1:end-1);     % 1 × (Nsol-1)

end

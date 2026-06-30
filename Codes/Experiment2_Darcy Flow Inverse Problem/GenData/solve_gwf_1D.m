function P = solve_gwf_1D(coef, F)

    % ----------------------------------------------------------
    % 1. INPUT GRID: cell-centered, same as 2D code
    % ----------------------------------------------------------
    % coef and F are length K, defined at:
    % x1(i) = (i - 0.5)/K,  i = 1,...,K   (cell centers)
    %
    % Example: 1/(2K), 3/(2K), ..., (2K-1)/(2K)
    % ----------------------------------------------------------

    K = length(coef);

    x1 = (1/(2*K) : 1/K : (2*K-1)/(2*K))';   % cell-centered grid

    % ----------------------------------------------------------
    % 2. SOLVER GRID: node-centered, same as 2D code
    % ----------------------------------------------------------
    % x2(j) = (j-1)/(K-1), j = 1,...,K
    %
    % Example: 0, 1/(K-1), 2/(K-1), ..., 1
    % ----------------------------------------------------------

    x2 = (0 : 1/(K-1) : 1)';                  % node-centered grid

    % ----------------------------------------------------------
    % 3. Interpolate coef and F from cell centers → nodes
    % ----------------------------------------------------------

    coef2 = interp1(x1, coef, x2, 'spline');
    F2    = interp1(x1, F,    x2, 'spline');

    % ----------------------------------------------------------
    % 4. Interior extraction: match your 2D code
    % ----------------------------------------------------------

    coef_int = coef2(2:K-1);
    F_int    = F2(2:K-1);

    N = K-2;                          % interior unknowns
    h = 1/(K-1);                      % grid spacing

    % ----------------------------------------------------------
    % 5. Build the 1D diffusion matrix
    % ----------------------------------------------------------

    A = sparse(N,N);

    for i = 1:N
        % interface coefficients with averaging (exactly like 2D code)
        a_ip = (coef2(i+2) + coef2(i+1)) / 2;   % coef at (i+1/2)
        a_im = (coef2(i+1) + coef2(i))   / 2;   % coef at (i-1/2)

        % fill matrix
        A(i,i) = (a_ip + a_im) / h^2;

        if i > 1
            A(i,i-1) = -a_im / h^2;
        end
        if i < N
            A(i,i+1) = -a_ip / h^2;
        end
    end

    % ----------------------------------------------------------
    % 6. Solve linear system for interior points
    % ----------------------------------------------------------

    Pint = A \ F_int;

    % ----------------------------------------------------------
    % 7. Impose boundary conditions p(0)=p(1)=0
    % ----------------------------------------------------------

    P2 = [0; Pint; 0];

    % ----------------------------------------------------------
    % 8. Interpolate solution back to cell-centered grid
    % (exact analog of your final interp2(…)')
    % ----------------------------------------------------------

    P = interp1(x2, P2, x1, 'spline');

end

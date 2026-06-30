%%
clear; close all; clc

% define parameters
K1 = 64;
N = 1000000;

% define parameters for random field a and F
alpha = 2;
tau = 3;
F = ones(K1,1);

X1 = 1/(2*K1):1/K1:(2*K1-1)/(2*K1);


% define arrays to store results
x = zeros(N,K1);
sol = zeros(N,K1);

for j=1:N
    tic
    % generate log-normal input
    a = exp(gaussrnd(alpha,tau,K1));
    u = solve_gwf_1D(a,F);
    %u = interp1(X1,u,X2,'spline'); % I commented this line because it is easier to have the pressure measurements at X1 
    % save results
    x(j,:) = log(a);
    sol(j,:) = u;
    disp(j);
    toc
end
  
% reduce dimension of sol
sol = sol(:,4:8:end);


y = sol;

% save final
save('./../../../Data/Experiment3/darcy_data1D_64','x','y','-v7.3')
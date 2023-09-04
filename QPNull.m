function [xstar, fxstar, lambda_star, v_star, KKT_gradX_norm, KKT_gradL_norm] = ...
    QPNull(Q, c, A, b)

% INPUTS: 
% Q = SPD Matrix n,n of the quadratic loss function. 
% c = n-dimensional vector of the quadratic loss function
% A = matrix K,n of the equality constraints
% b = K-dimensional vector of the equality constraints


% OUTPUTS:
% xstar = solution 
% fxstar = value of the loss in xstar
% lambda_star = lagrangian multiplier computed by the function
% v_star = solution of the substitution variable
% KKT_gradX_norm = norm of the corresponding KKT condition w.r.t. X
% KKT_gradL_norm = norm of the corresponding KKT condition w.r.t. Lambda

[K, n] = size(A);


% Z initialization
% We divide A into A1 and A2. The columns of A1 must be linearly indipendent
% we have no problem doing this equation, because the first m columns known from 
% the construction of our problem that they are linearly indipendent
A1 = A(:, 1:K); 
A2 = A(:, K+1:end);
Z = [-A1\A2; speye(n-K)];

% xhat initialization
x_hat = [A1\b; zeros(n-K,1)];

% compute v_star as solution of:
% (Z' Q Z)v = -Z'(Q xhat + c )

v_star = (Z' * Q * Z)\(-Z' * (Q * x_hat + c));


% compute xstar, given v_star and xhat
xstar = Z * v_star + x_hat;

% compute lambda_star given xstar
lambda_star = (A * A')\(-A *(c + Q * xstar));

% compute fxstar
fxstar = 0.5 * xstar' * Q * xstar + c' * xstar;

KKT_gradX_norm = norm(Q * xstar + c + A' * lambda_star);
KKT_gradL_norm = norm(A * xstar - b);


end
function [xstar, fxstar, lambda_star, KKT_gradX_norm, KKT_gradL_norm] = ...
    QPSchur(Q, c, A, b)

% INPUTS: 
% Q = SPD Matrix n,n of the quadratic loss function. 
% c = n-dimensional vector of the quadratic loss function
% A = matrix K,n of the equality constraints
% b = K-dimensional vector of the equality constraints

% OUTPUTS:
% xstar = solution 
% fxstar = value of the loss in xstar
% lambda_star = lagrangian multiplier computed by the function
% KKT_gradX_norm = norm of the corresponding KKT condition w.r.t. x
% KKT_gradL_norm = norm of the corresponding KKT condition w.r.t lambda




% Schur complement computation
Q_hat = A * (Q\A');

% We notice that instead of calculating the inverse of Q, we are gonna use
% the backslash method

% find lambda_star solving the linear system:
% Q_hat * lambda = -b -A * Qinv * c

lambda_star = Q_hat\(-b - A * (Q\c));

% Compute xstar given lambda_star
xstar = Q\(-c - A' * lambda_star);

% compute fxstar
fxstar = 0.5 * xstar' * Q * xstar + c' * xstar;

KKT_gradX_norm = norm(Q * xstar + c + A' * lambda_star);
KKT_gradL_norm = norm(A * xstar - b);


end
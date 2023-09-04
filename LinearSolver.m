function [xstar,fxstar,lambda_star,w, KKT_gradX_norm, KKT_gradL_norm] = LinearSolver(Q,c,A,b,tol, maxit)
% We want to create our system K * w = d  
[K,n] = size(A);
d = [-c; b];
K_matrix = [Q , A'; A , zeros(K,K)];
w = minres(K_matrix, d,tol, maxit);
xstar = w(1:n,:);
lambda_star = w(n+1:end, :);

fxstar = 0.5 * xstar' * Q * xstar + c' * xstar;
KKT_gradX_norm = norm(Q * xstar + c + A' * lambda_star);
KKT_gradL_norm = norm(A * xstar - b);


end
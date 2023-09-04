clc
clear
close all


n = 1e5;
K = 500;
tol = 1e-7;
maxit = 2000;
[Q, c, A, b] = createMatrix(n,K);



% tic
% [xstar, fxstar, lambda_star, KKT_gradX_norm, KKT_gradL_norm] = QPSchur(Q, c, A, b);
% toc

%tic
%[xstar, fxstar, lambda_star, v_star,KKT_gradX_norm, KKT_gradL_norm] = QPNull(Q, c, A, b);
%toc

%tic
%[xstar,fxstar,lambda_star,w, KKT_gradX_norm, KKT_gradL_norm] = LinearSolver(Q,c,A,b,tol, maxit);
%toc

function [Q, c, A, b] = createMatrix(n, K)

c = ones(n, 1);
b = ones(K, 1);

A = zeros(K, n);

for i=1:K
    v1 = zeros(n, 1);
    v1(i:K:end) = 1;
    A(i, :) = v1';
end    

e = ones(n,1);
Q = spdiags([-e 2*e -e],-1:1,n,n);

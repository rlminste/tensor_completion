% test cp_wals

clear

n = 100; r = 5;
info = create_problem('Factor_Generator', 'randn', 'Noise', 1e-10, 'Size', [n n n],'Num_Factors',r);

X = info.Data;
Xtrue = info.Soln;

W = tensor(ones(n,n,n));

[M,out] = cp_wals(X,W,r,'maxiter',50);

relerr = norm(tensor(M)-Xtrue)/norm(Xtrue)

[Mcp,outcp] = cp_als(X,r,'maxiter',50);
relerrcp = norm(tensor(Mcp)-Xtrue)/norm(Xtrue)

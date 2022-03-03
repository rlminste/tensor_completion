function [M,output] = cp_wals(X,W,r,varargin)

d = ndims(X);
WX = W.*X;
normX = norm(WX);

%%
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParameter('dimorder',1:d,@(x) isequal(sort(x),1:d));
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParameter('printitn',1,@isscalar);
params.addParameter('fixsigns',true,@islogical);
params.parse(varargin{:});

%%
errchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;

%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= d
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(2:end)
        if ~isequal(size(Uinit{n}),[size(X,n) r])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    % Observe that we don't need to calculate an initial guess for the
    % first index in dimorder because that will be solved for in the first
    % inner iteration.
    if strcmp(init,'random')
        Uinit = cell(d,1);
        for n = dimorder(2:end)
            Uinit{n} = rand(size(X,n),r);
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
        Uinit = cell(d,1);
        for n = dimorder(2:end)
            Uinit{n} = nvecs(X,n,r);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% 
U = Uinit;
relerr = 0;
lambda = ones(r,1);

% Store the last MTTKRP result to accelerate fitness computation.
U_mttkrp = zeros(size(X, dimorder(end)), r);


%% main iterations
for i = 1:maxiters
    
    errold = relerr;
    
    % iterate over d modes
    for j = dimorder(1:end)
        
        % compute MTTKRP
        MTK = mttkrp(WX,U,j);
        if j == dimorder(end)
            U_mttkrp = MTK;
        end
        
        % iterate over each row
        for k = 1:size(X,j)
            colons = repmat({':'},1,d);
            colons{j} = k;
            % find nonzeros of corresponding parts of W
            inds = find(W(colons{:}));
            
            % form HtH matrix
            HtH = ones(r,r);
            
            for l = 1:size(inds,1)
                for n = 1:j-1
                    Un = U{n};
                    HtH = HtH.* (Un(inds(l,n),:)'*Un(inds(l,n),:));
                end
                for n = j+1:d
                    Un = U{n};
                    HtH = HtH.*(Un(inds(l,n-1),:)'*Un(inds(l,n-1),:));
                end
            end
            
            
            % solve LS problem
            Unew(k,:) = MTK(k,:) / (HtH + diag(lambda));
            
        end % each row
        
        % Normalize each vector to prevent singularities in coefmatrix
        if i == 1
            lambda = sqrt(sum(Unew.^2,1))'; %2-norm
        else
            lambda = max( max(abs(Unew),[],1), 1 )'; %max-norm
        end            

        Unew = bsxfun(@rdivide, Unew, lambda');

        U{j} = Unew;

    end %each mode
    
    M = ktensor(lambda,U);
    
    % This is equivalent to innerprod(X,P).
%         iprod = sum(sum(M.U{dimorder(end)} .* U_mttkrp) .* lambda');
%         if normX == 0
%             relerr = 1- (norm(M)^2 - 2 * iprod);
%         else
%             normresidual = sqrt( normX^2 + norm(M)^2 - 2 * iprod );
%             relerr = (normresidual / normX); 
%         end
        relerr = norm(full(M)-WX)/normX;
        errchange = abs(errold - relerr);
        
        % Check for convergence
        if (i > 1) && (errchange < errchangetol)
            flag = 0;
        else
            flag = 1;
        end
        if (mod(i,printitn)==0) || ((printitn>0) && (flag==0))
            fprintf(' Iter %2d: err = %e err-delta = %7.1e\n', i, relerr, errchange);
        end
        
        % Check for convergence
        if (flag == 0)
            break;
        end      
    
    
end %big iterations
            
%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
M = arrange(M);
% Fix the signs
if params.Results.fixsigns
    M = fixsigns(M);
end

if printitn>0
    if normX == 0
        relerr = 1 - (norm(M)^2 - 2 * innerprod(X,M));
    else
        normresidual = sqrt( normX^2 + norm(M)^2 - 2 * innerprod(X,M) );
        relerr = (normresidual / normX); 
    end
  fprintf(' Final err = %e \n', relerr);
end

output = struct;
output.params = params.Results;
output.iters = i;           
            
            
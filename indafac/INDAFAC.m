function [Factors,Diagnostics] = indafac(X,F,FacIni,varargin);
% Fits a PARAFAC model to a three way array using a Levenberg-Marquadt algorithm
%
% [Factors,Diagnostics] = indefac(X,F[,FacIni[,Option1,Option1Value[,...]]]);
% 
% Input
% X     : data array
% F     : number of components
% FacIni: cell vector with initial estimations for the loading matrices
%
% Options
% cc_fit      : converge if loss function/sum(X(:).^2) < value       (default: 10 * eps)
% cc_grad     : converge if max(abs(gradient)) < value               (default: 1e-9)
% cc_maxiter  : Max number of iterations                             (default: 2500)
% cc_par      : converge if relative change in step's 2-norm < value (default: 1e-8)
% cc_relfit   : converge if relative loss function decrease < value  (default: 1e-6)
% diagnostics : on(default)/off - show/do not show diagnostics
% display     : n(default 5)    - show fit every n iterations
%               none            - no fit is shown
% init_method : random - random values                               (default)
%               orth   - orthogonalised random loadings
%               svd    - nipals
%               dtld   - DTLD/GRAM (only applies to 3-way and no missing values)
% init_maxiter: Max number of iterations for initialisation          (default: 5)
% init_tol    : initialisation stops if relative loss function decrease < value 
%               (applies only when iterative methods are employed)   (default: 1e-5)
% lambdainit  : initial value of the damping parameter expressed     (default: 1e-1)
%               as a fraction of max(diag(JTJ))
% lambdaudpar : lambda update parameters                             (default: [2,1/3])
% weights     : empty if none.                                       (default: [])
%               It must be the same size of the data otherwise           
%
% Output
% Factors     : cell vector with the loading matrices as elements
% Diag        : structure with some diagnostics
%               'fit'           : value of the loss function
%               'it(1)'         : number of iterations
%               'it(2)'         : number of iterations involving a new computation of the Jacobian 
%                                 (i.e. the update was accepted)
%               'convergence(1)': 1 if relative loss function decrease conv. criterion is met 
%               'convergence(2)': 1 if parameters' update conv. criterion is met 
%               'convergence(3)': 1 if absolute loss function precision conv. criterion is met 
%               'convergence(4)': 1 if gradient infinite norm conv. criterion is met 
%               'convergence(5)': not in use 
%               'convergence(6)': 1 if maximum number of iterations is reached 
% 
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% gt@kvl.dk 
%
% Last modified: 03-Sep-2003
%

%Some initial values
ConvCrit   = false(6,1);  %Not converged
it         = [0 0 0];  %Number of iterations [global outer-loop nonlinear update]
Update     = {};
LamHistory = [];

%Check input values
Options = ParOptions(varargin{:});

%individuate the missing and present values and eliminate the all missing values' slabs
[No{1:ndims(X)}]   = Clean(X);
X                  = X(No{:});
dimX               = size(X);

%Find missing values
Miss = isnan(X(:));
if sum(Miss) * 4 <= length(Miss);
   Miss = uint32(find(Miss));
end
SSX     = tss(X);

%Check for initial values and number of factors and initialise
if nargin < 2 | (isempty(F) & ~(exist('FacIni','var') || isempty(FacIni)))
   error('The number of factor must be defined')
end
if exist('FacIni','var') && ~isempty(FacIni)
   
   if length(FacIni) ~= ndims(X)
      error('Inadequate number of loading matrices')
   end
   if ~isempty(F) 
      if ~all(cellfun('size',FacIni,2) == F)
         error('Initial values matrices inconsistent with the desired n. of factors')
      end
   else
      F = size(FacIni{1},2);
   end
   for i = 1:ndims(X)
      Factors{i} = FacIni{i}(No{i},:);
   end
   if any(size(X) ~= cellfun('size',Factors,1))
      error('Initial values matrices not consistent with array dimensions')
   end
   OV = true;
   
else
   
   if ~exist('F','var')
      error('The number of factors and/or initial estimations of the loading matrices must be provided')
   end
   %Exclude initialisations that cannot handle missing values
   if any(Miss) & strcmpi('dtld',Options.initialisation.method)
      Options.initialisation = 'random';
   end
   %Initialise
   [Nil,Factors{1:ndims(X)}] = InitPar(X,F,Options.initialisation,Options.weights);
   OV                        = false;
   
end

%Initial fit
FitNew  = tss(X - nmodel(Factors));

%Show some diagnostics
if strcmpi(Options.diagnostics,'on')
   ShowDiagnostics(Options,F,SSX,dimX,OV,sum(isnan(vec(X)))/prod(dimX))
end

if ~isequal(Options.display,'none')
   fprintf('\n\n            Fit           It      EV %%     Rho            Lambda             Max Gr\n')
end

%Start fitting
while ~any(ConvCrit)
   
   %Update the number of iterations for the outer loop
   it(2) = it(2) + 1;    
   
   %Scale loadings vectors to equal norm   
   [Factors{1:ndims(X)}] = scale_factors(0,Factors{:});
   
   %Old factors
   FactorsOld = Factors;
   FitOld     = FitNew;
   
   %Find element corresponding to the infinite norm
   NormPar = norm(vec(cat(1,Factors{:})')); %Norm of the vector of unknowns
   
   if ~it(1) %Calculate indexes for position of missing and non-zeros for Jacobian
      [Missing,JacIndex] = ParJac_sparse(Factors,X,Options.weights);
   end

   %calculate JTJ and Gradient
   [Gr,JtJ] = ParJac_sparse(Factors,X,Options.weights,JacIndex,Missing);
   
   %If Gradient is "0" stop
   NormGr        = max(abs(Gr));
   ConvCrit(4,1) = Options.convcrit.grad >= NormGr;                                                   %Gradient "equal" to 0
   
   %Initialise the damping parameter if necessary
   if ~it(1)
      Lambda = Options.lambdaini * max(diag(JtJ));
   end
   LamHistory(end + 1) = Lambda;
   
   Do_It     = true;
   while Do_It && ~any(ConvCrit) %Begin the internal loop
      
      it(1) = it(1) + 1;                       %Update n. of iterations             
      Psi   = JtJ + Lambda * eye(size(JtJ,1)); %Compute left hand side of the normal equations               
      
      %Solve the normal equations
      [Psi,CFlag]  = chol(Psi);      %Cholesky factor
      if ~CFlag                      %Choleski is positive definite
         
         warning off
         lastwarn('')
         delta = Psi\(Psi'\Gr);  %Update calculated by back substitution
         warning on
         
         if isempty(lastwarn) %The matrix is nicely scaled
            
            NormDelta    = norm(delta);
            Count        = 0;
            for m = 1:length(dimX)
               Update{m} = reshape(delta(Count + 1:Count + dimX(m) * F),F,dimX(m))';
               Count     = Count + dimX(m) * F;
               Fac_Up{m} = Factors{m} + Update{m};
            end
            
            % Update fit and save old
            FitNew      = tss(X - nmodel(Fac_Up));
            LinDecrease = delta(:)' * (Lambda * delta + Gr); %the plus is just for the fact that D = -J
            Do_It       = false;
            
            % Compute Gain Ratio
            warning off
            Rho  = (FitOld - FitNew) / LinDecrease; 
            warning on
            
            %Reject step if the gain is insufficient
            if Rho <= 0
               Do_It = true;
               Lambda = Lambda * Options.lambdaudpar(1);
            elseif Rho < 0.25
               Lambda = Lambda * Options.lambdaudpar(1);
            elseif Rho > 0.75
               Lambda = Lambda * Options.lambdaudpar(2);
            end
            
            %Check convergence
            if ~FitNew %Relative fit decrease
               ConvCrit(1,1) = true ; %
            else
               ConvCrit(1,1) = Options.convcrit.relfit >= abs(FitOld - FitNew)/FitOld; %Relative fit decrease
            end
            %Check convergence
            ConvCrit(2,1) = Options.convcrit.par >= NormDelta/NormPar;                                        %Relative change in the parameters
            
         else
            
            %Missing values typically cause this
            Lambda = Lambda * Options.lambdaudpar(1); 
            
         end

      else
         
         %The Hessian approximation is "nearly" singular
         %This have the step rejected and the trust region shrunk
         Lambda = Lambda * Options.lambdaudpar(1); 
         
      end
      ConvCrit(6,1) = it(1) == Options.convcrit.maxiter;  %Max. number of iterations
      
   end %End of the inner loop
   Factors = Fac_Up;
   
   ConvCrit(3,1) = Options.convcrit.fit  >= FitNew / SSX; %Absolute fit value
   if ~isequal(Options.display,'none') && ~rem(it(1),Options.display) && ~any(ConvCrit)
      DisplayIt(it(1),FitNew,SSX,Rho,LamHistory(end),NormGr,Options)
   end
   
end %end of the outer loop

%Scale the factors according to the common convention
[Factors{end:-1:1}] = scale_factors(1,Factors{end:-1:1});

%Sort the factors according to their norm (in decresing order)
[nil,Seq]             = sort(-sum(Factors{1}.^2));
[Factors{1:ndims(X)}] = FacPerm(Seq,Factors{:});

if nargout > 1
   
   Diagnostics = struct('fit',[],'it',[],'convergence',[]);
   Diagnostics.fit         = FitNew;
   Diagnostics.it          = it;
   Diagnostics.convergence = ConvCrit;
   
end
if strcmpi(Options.diagnostics,'on')  
   
   CCMsg = {'Relative loss function decrease','Parameters'' relative change','Loss function value close to required precision',...
         'Gradient equal to zero','Congruence relative change','Max number of iterations reached'};
   t = find(ConvCrit);
   if ~ConvCrit(6)
      
      fprintf(['\n\n The PARAFAC-LM algorithm has converged after ',num2str(it(1)),' iterations'])
      if length(t) == 1
         fprintf(['\n Met convergence criterion: ',CCMsg{t}])
      else
         fprintf(['\n Met convergence criteria: ',CCMsg{t}])
         for n = 2:length(t)
            fprintf(['\n                          : ',CCMsg{t(n)}])
         end
      end
      
   else
      fprintf('\n Max number of iterations reached')
   end
   fprintf('\n\n')
   
end

%--------------------------------------------------------------------------------------------------------------------------------

function DisplayIt(It,Fit,SSX,Rho,LambdaOld,Gr,Options)
FitStr = sprintf('%12.10f',Fit);
FitStr = [char(32*ones(1,22-length(FitStr))),FitStr];
ItStr  = num2str(It);
ItStr  = [char(32*ones(1,length(num2str(Options.convcrit.maxiter))-length(ItStr))),ItStr];
VarStr = sprintf('%2.4f',100*(1-Fit/SSX));
VarStr = [char(32*ones(1,7-length(VarStr))),VarStr];
RatStr = sprintf('%2.4f',Rho);
RatStr = [char(32*ones(1,7-length(RatStr))),RatStr];
LamStr = sprintf('%12.4f',LambdaOld);
LamStr = [char(32*ones(1,17-length(LamStr))),LamStr];
GrStr  = sprintf('%8.4f',Gr);
GrStr  = [char(32*ones(1,19-length(GrStr))),GrStr];
fprintf([' ',FitStr,'  ',ItStr,'  ',VarStr,'  ',RatStr,'  ',LamStr,'   ',GrStr]);
fprintf('\n');

%--------------------------------------------------------------------------------------------------------------------------------

function ShowDiagnostics(Options,F,SSX,dimX,OV,MV)
fprintf( '\n Compute PARAFAC model using the Levenberg-Marquadt algorithm')
fprintf(['\n Array dimensions    : [ %i',sprintf(' x %i',dimX(2:end)),' ]'],dimX(1));
if MV
   fprintf( '\n Missing values      : %3.2f%%',MV * 100)
else
   fprintf( '\n No missing values')
end
fprintf( '\n Number of factors   : %i',F)
fprintf( '\n\n Convergence criteria')
fprintf( '\n Max number of iterations        : %i',Options.convcrit.maxiter)
fprintf( '\n Relative loss function decrease : %1.1e',Options.convcrit.relfit)
fprintf( '\n Loss function value             : %1.1e',Options.convcrit.fit * SSX)
fprintf( '\n Relative parameters'' update norm: %1.1e',Options.convcrit.par)
fprintf( '\n Gradient infinite norm          : %1.1e',Options.convcrit.grad)
fprintf( '\n\n Initialisation')
if OV
   fprintf('\n Using old values for initialisation')
else
   
   switch(Options.initialisation.method)
      case 'random'
         fprintf('\n Matrices of random numbers as initialisation')
      case 'orth'
         fprintf('\n Column-wise orthogonal matrices of random numbers as initialisation')
      case 'svd'
         fprintf('\n Using NIPALS algorithm for initialisation')
      case 'dtld'
         fprintf('\n Using DTLD for initialisation')
   end
   
end
return

%--------------------------------------------------------------------------------------------------------------------------------

function varargout = FacPerm(Perm,varargin)
% Permute columns of a, b, c, ... according to Perm
%
% [ap,bp,cp,...] = FacPerm (Perm,a,b,c,...);
% 
% Inputs: 
% Perm:      permutation matrix.
% a,b,c,...: matrices to be permuted.
% 
% Outputs:
% ap,bp,cp,...: column-wise permuted matrices
% 
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% 
% Last modified: September 3rd, 2004
%
if size(Perm,1) == size(Perm,2)
   for i=1:nargin-1
      varargout{i} = varargin{i} * Perm;
   end
elseif length(Perm) == prod(size(Perm));
   for i=1:nargin-1
      varargout{i} = varargin{i}(:,Perm);
   end
end

%--------------------------------------------------------------------------------------------------------------------------------

function varargout = Clean(X);
% Provide the indexes of subarrays which do not contain only missing values
%
% [IndX1,IndX2,...IndXn] = CleanData(X);
% 
% Inputs: 
% X: n-way array (where n >= 2)
% 
% Outputs: 
% IndX1,IndX2,...IndXn: are the indexes in the n-dimensions of X.
% 
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% gt@kvl.dk 
% 
% Last modified: September 3rd, 2004 
%
NDX     = ndims(X);
Xnan    = isnan(X);
SizeXor = size(X);
for i = NDX:-1:1
   S            = SizeXor([i,1:i-1,i+1:NDX]);
   Xv           = reshape(permute(Xnan,[i,1:i-1,i+1:NDX]),S(1),prod(S(2:end)));
   ExX          = ~all(Xv,2);
   varargout{i} = find(ExX);
end

%--------------------------------------------------------------------------------------------------------------------------------

function [varargout] = scale_factors(varargin)
% Scales loading matrices.
% 
% function [As,Bs,Cs,...] = Scale_factors (flag,A,B,C,...);
% 
% Inputs: 
% flag     : 0 -> the loading vectors in the different loading matrices will have the same norm
%            1 -> the loading vectors in the first M-1 loading matrices will have norm 1.
% A,B,C,...: loading matrices to be scaled
% 
% Outputs:
% As,Bs,Cs,...: scaled loading matrices.
% 
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% 
% Last modified: September 3rd, 2004
% 
% Contact: Giorgio Tomasi, gt@kvl.dk 
%

if islogical(varargin{1}) | prod(size(varargin{1})) == 1;
   flag     = varargin{1};
   varargin = varargin(2:end);
else
   flag = false;
end
if nargout ~= length(varargin)
   error('The number of outputs should match the number of inputs.')
end
for i = 1:nargout - 1
   sc(i,:)      = sqrt(sum(varargin{i}.^2)) .* sign(sum(varargin{i}.^3));
   varargout{i} = varargin{i} * diag(sc(i,:))^-1;
end
sig = diag(prod(sc,1));
if flag
   varargout{nargout} = varargin{end} * sig;
else
   
   sc(nargout,:) = sqrt(sum(varargin{nargout}.^2));
   vn            = diag(prod(abs(sc),1).^(1/length(varargin)));
   for i = 1:nargout - 1
      varargout{i} = varargout{i} * vn;
   end
   varargout{nargout} = varargin{nargout} * diag(sc(nargout,:).^-1) * vn * sign(sig);
   
end

%--------------------------------------------------------------------------------------------------------------------------------

function Options = ParOptions(varargin);
% function Options = ParOptions (varargin);
% 
% Description:
% Defines the Options for GenPARAFAC and GNPARAFAC
% 
% Inputs: 
% varargin: it can be one structure (in which case the function controls only if the values in the
% structure are available) or a list of option names and obtions values
% Options = ParOptions (optionname1,optionvalue1,optionname2,....);
% In the latter case the non specified options will have default value.
% 
% 
% Outputs:
% Options: Options structure
% 
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% 
% Last modified: September 3rd, 2004
% 
% Contact: Giorgio Tomasi, gt@kvl.dk 
%

if ~nargin
   
   Options = struct('convcrit',struct('fit',10 * eps,...
      'grad',1e-9,...
      'maxiter',2500,...
      'par',1e-8,...
      'relfit',1e-6),...
      'diagnostics','off',...
      'display',1,...
      'initialisation',struct('maxiter',5,...
      'method','random',...
      'tol',1e-5),...
      'lambdaudpar',[2,1/3],...
      'lambdaini',1e-1,...
      'weights',[]);
   
elseif nargin == 1
   
   Op = fieldnames(varargin{1});
   for i = 1:length(Op)
      
      V = varargin{1}.(Op{i});
      switch lower(Op{i})
         
         case 'convcrit'
            CCOp = fieldnames(V);
            for j = 1:length(CCOp)
               
               U = V.(CCOp{j});
               switch lower(CCOp{j})                     
                  case 'fit'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''CC_Fit''. Value must be a number')
                     end
                     
                  case 'grad'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''CC_Grad''. Value must be a number')
                     end
                     
                  case 'maxiter'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''CC_MaxIter''. Value must be an integer')
                     elseif rem(U,1)
                        error('Invalid property assignment for ''CC_MaxIter''. Value must be an integer')
                     end         
                     
                  case 'par'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''CC_Par''. Value must be a number')
                     end
                     
                  case 'relfit'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''CC_RelFit''. Value must be a number')
                     end
                     
                  otherwise
                     error('Invalid property name')
                     
               end
               
            end
            
         case 'diagnostics'
            if ~any(strcmp({'on','off'},lower(V)))
               error('Invalid assignment for ''Diagnostics''. It must be ''on'' or ''off''')
            end
            
         case 'display'
            if ~strcmp('none',lower(V)) & isa(V,'double')
               if rem(V,1)
                  error('Invalid property assignment for ''Display''. Value must be an integer or ''None''')
               end
            end
            
         case 'initialisation'
            InOp = fieldnames(V);
            for j = 1:length(InOp)
               
               U = V.(InOp{j});
               switch lower(InOp{j})
                  case 'method'
                     if ~any(strcmp({'random','orth','nipals','dtld','swatld','best'},lower(U)))
                        error('Invalid property assignment for ''Init_Method''. Value must be either ''random'',''orth'',''nipals'',''dtld'' or ''swatld''')
                     end
                     
                  case 'maxiter'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''Init_MaxIter''. Value must be an integer')
                     elseif rem(U,1)
                        error('Invalid property assignment for ''Init_MaxIter''. Value must be an integer')
                     end         
                     
                  case 'tol'
                     if ~isa(U,'double')
                        error('Invalid property assignment for ''Init_Tol''. Value must be a number')
                     end
                     
                  otherwise
                     error('Invalid property name')
                     
               end
               
            end
            
         case 'lambdaudpar'
            if ~isa(V,'double') | length(V) ~= 2
               error('Invalid property assignment for ''LambdaUDPar''. Value must be a vector of 2 numbers')
            elseif V(2) >= 1
               error('Invalid property assignment for ''LambdaUDPar'', Value(2) must be less than 1')
            end         
            
         case 'lambdaini'
            if ~isa(V,'double') | length(V) ~= 1
               error('Invalid property assignment for ''LambdaIni''. Value must be either a double')
            end
            
         case 'weights'
            if ~isa(V,'double')
               error('Invalid property assignemente for ''Weights''. Value must be an array of doubles')
            end
            
         otherwise
            error('Invalid property name')
            
      end
   end
   Options = varargin{1};
   
else
   
   Options = struct('convcrit',struct('fit',10 * eps,...
      'grad',1e-9,...
      'maxiter',2500,...
      'par',1e-8,...
      'relfit',1e-6),...
      'diagnostics','off',...
      'display',1,...
      'initialisation',struct('maxiter',5,...
      'method','random',...
      'tol',1e-5),...
      'lambdaudpar',[2,1/3],...
      'lambdaini',0.1,...
      'weights',[]);
   
   if rem(nargin,2)
      
      if isa(varargin{1},'struct')
         if ~all(strcmp(sort(fieldnames(varargin{1})),sort(fieldnames(Options))))
            error('Incompatible structures')
         else 
            Options_old = varargin{1};
            Options     = varargin{1};
            varargin(1) = [];
            EM          = 'The input Options structure is returned';
         end
      else
         error('Properties and values must come in pairs')
      end
      
   else
      Options_old = Options;
      EM          = 'The default Options structure is returned';
   end
   
   for i = 1:2:length(varargin)
      
      if isequal(lower(varargin{i}(1:min(5,length(varargin{i})))),'init_') 
         Options.initialisation.(lower(varargin{i}(6:end))) = lower(varargin{i+1});
      elseif isequal(lower(varargin{i}(1:min(3,length(varargin{i})))),'cc_')
         Options.convcrit.(lower(varargin{i}(4:end))) = lower(varargin{i+1});
      else
         Options.(lower(varargin{i})) = lower(varargin{i+1});
      end
      
   end
   
   try
      Options = ParOptions(Options);
   catch
      warning('')
      fprintf(['The required update generated an error:\n',lasterr,'\n'])
      fprintf(EM)
      Options = ParOptions(Options_old);
   end
   
end

%--------------------------------------------------------------------------------------------------------------------------------

function [Gr,JtJ] = ParJac_sparse(Factors,X,Weights,XSort,Missing)
% Compute JTJ and Gradient of a PARAFAC model using sparse a sparse matrix for the Jacobian
%
% [Gr,JtJ] = ParJac_sparse(Factors,X,Weights,XSort,Missing)
% 
% Inputs
% Factors: cell vector with the loading matrices as elements
% X      : data array
% Weights: weights (optional).
%          Empty         : no weights
%          Same size as X: Weighted Least Squares
%
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% Contact: gt@kvl.dk 
% 
% Last modified: September 3rd, 2004 
%

if ~exist('Opt','var')
   Opt = false;
end
dimX = size(X);
Nel  = prod(dimX);
F    = size(Factors{1},2);
JtJ  = [];
Gr   = [];
if ~exist('XSort','var') | isempty(XSort) 
   
   %Return indexes for non-zero elements for the Jacobian, if not given
   Xs  = reshape(1:Nel,dimX);
   Xm = vec(isnan(X));
   if exist('Weights','var') && ~isempty(Weights) && ~all(size(Weights) == Nel)
      Xm = Xm | ~vec(Weights);
   end
   if sum(Xm) * 4 < Nel 
      %Check if using the indexes requires less space than using logical values
      Xm = uint32(find(Xm));
   end
   for m = 1:ndims(X);
      
      x   = permute(Xs,[m,1:m - 1,m + 1:length(dimX)]);
      x   = repmat(x(:),1,F);
      ord = [ndims(X):-1:m + 1,m - 1:-1:1];
      s   = prod(dimX(ord));
      fo  = repmat(0:F - 1,s,1) * Nel;
      for i = 1:dimX(m)
         
         a            = repmat([i:dimX(m):Nel]',1,F);
         b            = ((i - 1) + sum(dimX(1:m - 1))) * F * Nel;
         rs           = (i - 1) * s * F + 1;
         re           = i * s * F;
         Ind(rs:re,m) = x(vec(a)) + vec(fo) + b;
         
      end
      
   end
   [JtJ(:,1),JtJ(:,2)] = ind2sub([Nel,sum(dimX) * F],Ind(:));
   Ind                 = true(Nel * F * ndims(X),1);
   if any(Xm)
      %Handle missing values or 0 weights for any element
      Ind      = false(Nel * F * ndims(X),1);
      Gr       = Xm;
      [Nil,a]  = sort(JtJ,1);
      a        = reshape(a(:,1),ndims(X) * F,Nel);
      a(:,Xm)  = [];
      Ind(a)   = true;
   end
   if all(Ind)
      %It simply to save memory. If all elements are present Ind would
      %occupy as much space as X
      Ind = [];
   end
   Ind = uint32(find(Ind));
   %Storage is halved without remarkable effects on the speed
   if ~isempty(Ind)
      JtJ      = uint32(JtJ(Ind,:));
      JtJ(:,3) = Ind;
   else
      JtJ = uint32(JtJ);
   end
   
else
   
   Fix = logical([]);
   for m = 1:ndims(X);
      
      t                                = ones(1,F);
      ord = [ndims(X):-1:m + 1,m - 1:-1:1];
      for n = 1:length(ord)
         t = kr(t,Factors{ord(n)});
      end
      JtJ((m - 1) * Nel * F + 1:m * Nel * F) = vec(kron(ones(1,dimX(m)),t));
      
   end
   if size(XSort,2) == 2
      JtJ = sparse(double(XSort(:,1)),double(XSort(:,2)),JtJ,Nel,sum(dimX) * F,Nel * F * ndims(X));
   else
      JtJ = sparse(double(XSort(:,1)),double(XSort(:,2)),JtJ(XSort(:,3)),prod(dimX),sum(dimX) * F,size(XSort,1));
   end
   R   = vec(X - nmodel(Factors));
   if exist('Missing','var') & any(Missing)
      R(Missing) = 0;
   end
   if exist('Weights','var') & ~isempty(Weights)
      
      if ~issparse(Weights)
         
         %If one desires correlated noise this sparse diagonalisation can
         %be avoided by passing the weights as sparse. The noise
         %correlation matrix would be anyhow unviably large for normal problems.
         Weights = vec(Weights);
         k       = 1:length(Weights);
         Weights = sparse(k,k,Weights,length(Weights),length(Weights));
         
      end
      Gr  = JtJ' * (Weights * R);
      JtJ = full(JtJ' * (Weights * JtJ));   
      
   else
      
      Gr  = JtJ' * R;
      JtJ = full(JtJ' * JtJ);
      
   end
   
end

%--------------------------------------------------------------------------------------------------------------------------------

function vecx = vec(x);
% Vec operator
vecx = x(:);

%--------------------------------------------------------------------------------------------------------------------------------

function [conv,varargout] = InitPar(X,F,Options,Weights)
% function [conv,A,B,C,...] = InitPar (X,F,Options,Const);
% 
% Description:
% Find initial estimates for a PARAFAC model with F components fit on X
% 
% Inputs: 
% X: array of double
% F: number of factors to extract
% Options contains three fields
% 'method': six type of initialisation are available
% 
%    'orth'  : ten starts with orthogonal random matrices. The best is picked after 'iter' iterations 
%    'random': ten starts with random matrices. The best is picked after 'iter' iterations 
%    'nipals': the matrices are obtained via SVD  
%    'dltd'  : DTLD algorithm, available only up to 3 way and no missing values 
%    'best'  : uses all the possible initalisations and picks the best one 
% 
% 
% 'iter' : max number of iterations for the iterative methods
% 'tol'  : tolerance (i.e. relative fit decrease) for convergence in the iterative methods
% Const  : constraints (if required the initial values are found according to constraints); the format is
%        identical to the one used for PARAFAC.m
% 
% 
% Outputs:
% conv: returns 0 if the initialisation does not converge within the max number of iter.
% A,B,C,...: initial estimations for the loadings
% 
% Calls in the n-way toolbox: dtld, kr, ini, missmult, nmodel, normit, nshape, parafac
% 
% 
% Author: 
% Giorgio Tomasi 
% Royal Agricultural and Veterinary University 
% MLI, LMT, Chemometrics group 
% Rolighedsvej 30 
% DK-1958 Frederiksberg C 
% Danmark 
% gt@kvl.dk
%
% Last modified: September 3rd, 2004
% 

conv = 1;
if ~exist('Weights','var')
   Weights  = [];
end
if ~exist('Const','var')
   Const  = [];
end
Fit0 = inf;
dimX = size(X);
if strcmp('dtld',Options.method)
   if any(isnan(X(:)))
      warning(['Due to missing values ', Options.method, ' cannot be employed, random starts will be used instead'])
      Options.method = 'random';
   end   
   if ndims(X) ~= 3
      warning([Options.method,' cannot be employed as initialisation for ' int2str(ndims(X)) '-way arrays, random starts will be used instead'])
      Options.method = 'random';
   end
   if ndims(X) ~= 3 & any(strcmp({'dtld','nipals'},Options.method))
      warning(['Weights will not be used for the initialisation step using: ',Options.method]) 
   end 
end
switch Options.method
   case {'random','orth'}
      
      for i=1:10
         
         Z     = ones(1,F);
         Ztemp = ones(F);
         for j = ndims(X):-1:2
            
            Fac{j} = normit(rand(dimX(j),F)); %set non orthogonal basis
            if strcmp(Options.method,'orth') == 2
               Fac{j} = orth(Fac{j});  %set orthogonal basis for factors
            end
            Z     = kr(Z,Fac{j});
            Ztemp = Ztemp .* (Fac{j}'*Fac{j});
            
         end
         Fac{1}        = missmult(nshape(X,1),Z * pinv(Ztemp));
         [Fac,nil,fit] = parafac(X,F,[Options.tol,0,0,0,NaN,Options.maxiter],Const,Fac,[],Weights); %run It iterations with ALS
         if fit < Fit0
            [varargout{1:ndims(X)}] = deal(Fac{:});
            Fit0 = fit;
         end
         
      end
      
   case 'nipals'
      Fact                    = ini(X,F,2);
      [varargout{1:ndims(X)}] = deal(Fact{:});
      
   case 'dtld'
      [varargout{1:3}] = dtld(X,F,0);
      
   case 'best'
      Options.method              = 'orth';
      [Nil,Factors(1,1:ndims(X))] = InitPar(X,F,Options,Weights);
      Fit(1)                      = tss(X - nmodel(Factors(1,1:ndims(X))));
      Options.method              = 'random';
      [Nil,Factors(2,1:ndims(X))] = InitPar(X,F,Options,Weights);
      Fit(2)                      = tss(X - nmodel(Factors(2,1:ndims(X))));
      Options.method              = 'nipals';
      [Nil,Factors(3,1:ndims(X))] = InitPar(X,F,Options,Weights);
      Fit(3)                      = tss(X - nmodel(Factors(3,1:ndims(X))));
      if ~(any(isnan(X(:))) | ndims(X) == 3)
         
         Options.method              = 'dtld';
         [Nil,Factors(5,1:ndims(X))] = InitPar(X,F,Options,Weights);
         Fit(4)                      = tss(X - nmodel(Factors(5,1:ndims(X))));
         
      end
      [Nil,Best] = min(Fit);
      varargout = Factors(Best,:);
      
   otherwise 
      error('Initialisation not available!')
      
end

%-----------------------------------------------------------------------------------------------------------------------

function t = tss(X,miss)
if nargin == 1 || (nargin == 2 && miss)
   X(isnan(X)) = 0;
end
t = norm(X(:))^2;

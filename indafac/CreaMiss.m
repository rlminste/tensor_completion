function [X,Seed,varargout] = CreaMiss(Fac,DimX,Noise,Congruence,Missing,Mode,SD)
% Multi-way data generator with missing values
% The data has an underlying PARAFAC structure
%
% [X,SeedOut,Factors{1:length(dimX)}] = CreaMiss(Rank,dimX,Noise,Congruence,Missing,Mode,SeedIn);
% 
% Input:
% Fac       : number of underlying factors
% dimX      : array dimensions
% Noise(1)  : fraction ([0 1)) of homoscedastic noise over the total sum of squares of X
% Noise(2)  : fraction ([0 1)) of heteroscedastic noise over the total sum of squares of X
% Congruence: cosine of the angle between the loading vectors of the single components 
%             The congruence between components equal to Congruence^(order of the array).
% Missing   : fraction ([0 1)) of missing values;
% Mode      : 'RMV' => Randomly Missing Values
%             'RMS' => Randomly Missing Spectra
%             'SMS' => Systematically Missing Spectra (fluorescence)
%                      Data are removed diagonally from the "top left corner" and from the 
%                      "bottom right corner"
%                      The pattern is constant for all horizontal slabs (and up to the 
%                      (N - 2)th dimension in general)
% SeedIn    :  (optional) Seed for the random number generator
%              For a given Seed and for identical conditions (rank, dimensions...), the same
%              X is generated. 
%
% To give a slight resemblance of the real data one has to deal with with PARAFAC (i.e. 
% spectral data or smooth trajectories in two of the modes. The components in the 2nd to the n-th
% order are smooth while in the first they are derived from uniformely distributed random numbers.
%
% Output
% X      : syntethic data array
% Seed   : Seed used to generate the data, it allows recreating exactly the same array
% Factors: Underlying factors used to generate the model
%
% Author: Giorgio Tomasi
%         Royal Agricolture University, MLI, LMT
%         1958 Frederiksberg C
%         Danmark
%         gt@kvl.dk
%
% Last update: September 3rd, 2004
%

if ~exist('SD','var') | isempty(SD) | isnan(SD) 
   SD = sum(100*clock);
end
CutDims = length(DimX)-1:length(DimX);
Seed    = SD;
rand('state',SD(1))
[Xold,varargout{1:length(DimX)}] = CreaData(Fac,DimX,Noise,Congruence,SD);
Done                             = 0;
Count                            = 0;
while ~Done 
   X = Xold;
   switch Mode
      case 'RMV'
         % Randomly Missing Values
         w    = randperm(prod(DimX));
         w    = w(1:fix(Missing * prod(DimX)));
         X(w) = NaN;
         
      case 'RMS'
         % Randomly Missing Spectra
         w      = randperm(prod(DimX(1:end-1)));
         w      = w(1:fix(Missing * prod(DimX(1:end-1))));
         Ord    = [ndims(X) 1:ndims(X) - 1];
         X      = nshape(X,ndims(X));
         X(:,w) = NaN;
         X      = ipermute(reshape(X,DimX(Ord)),Ord);
         
      case 'SMS'
         %Systematically Missing Spectra
         if all(DimX(CutDims) == DimX(CutDims(1)))
            
            t         = floor(sqrt(prod(DimX(CutDims)) * Missing));
            Mat       = false(DimX(CutDims));
            for n = 1:t
               Mat(n,1:t - n + 1) = true;
            end
            
         else

            [t,i]     = sort(DimX(CutDims));
            S         = t * sqrt(Missing);
            Sf        = floor(S);
            Sc        = ceil(S);
            [nil,ind] = min(abs([Sf(1) * Sc(2) Sf(2) * Sc(1)] - fix(prod(t)*Missing)));
            Mat       = false(t);
            if ind(1) == 1
               a = Sc(2)/Sf(1);
               m = Sf(1);
            else
               a = Sf(2)/Sc(1);
               m = Sc(1);
            end
            for n = 1:t(1)
               Mat(n,fix([1:a * (n - t(1) + m) - 1])) = true;
            end
            Mat  = ipermute(Mat,i);
            
         end
         Mat  = flipud(fliplr(Mat)) | Mat;
         EDim = setdiff([1:length(DimX)],CutDims(:)'); 
         Ord  = [CutDims(:)',EDim]; 
         Temp = repmat(Mat,[1 1 DimX(EDim)]);
         Temp = ipermute(Temp,Ord);
         X(Temp) = NaN;
         
   end
   [SX{1:ndims(X)}] = Clean(X);
   if size(X) == cellfun('length',SX(:)');
      Done = 1;
   else
      Done = 0;
      Count = 1;
   end

end

%----------------------------------------------------------------------------------------------------------

function [X,varargout] = CreaData(Comp,Dims,NoiseLev,Corr,Seed);
% Multi-way data generator
% The data has an underlying PARAFAC structure
%
% [X,SeedOut,Factors{1:length(dimX)}] = CreaMiss(Rank,dimX,Noise,Congruence,Missing,Mode,SeedIn);
% 
% Input:
% Fac       : number of underlying factors
% dimX      : array dimensions
% Noise(1)  : fraction ([0 1)) of homoscedastic noise over the total sum of squares of X
% Noise(2)  : fraction ([0 1)) of heteroscedastic noise over the total sum of squares of X
% Congruence: cosine of the angle between the loading vectors of the single components 
%             The congruence between components equal to Congruence^(order of the array).
% Missing   : fraction ([0 1)) of missing values;
% Mode      : 'RMV' => Randomly Missing Values
%             'RMS' => Randomly Missing Spectra
%             'SMS' => Systematically Missing Spectra (fluorescence)
%                      Data are removed diagonally from the "top left corner" and from the 
%                      "bottom right corner"
%                      The pattern is constant for all horizontal slabs (and up to the 
%                      (N - 2)th dimension in general)
% SeedIn    :  (optional) Seed for the random number generator
%              For a given Seed and for identical conditions (rank, dimensions...), the same
%              X is generated. 
%
% To give a slight resemblance of the real data one has to deal with with PARAFAC (i.e. 
% spectral data or smooth trajectories in two of the modes. The components in the 2nd to the n-th
% order are smooth while in the first they are derived from uniformely distributed random numbers.
%
% Output
% X      : syntethic data array
% Seed   : Seed used to generate the data, it allows recreating exactly the same array
% Factors: Underlying factors used to generate the model
%
% Author: Giorgio Tomasi
%         Royal Agricolture University, MLI, LMT
%         1958 Frederiksberg C
%         Danmark
%         gt@kvl.dk
%
% Last update: September 3rd, 2004
%

if ~exist('Seed','var')
   Seed = 100 * sum(clock);
end
randn('state',Seed);
rand('state',Seed);
if length(NoiseLev) < 2
    NoiseLev(2) = 0;
end
NoiseLev  = sqrt(NoiseLev./(1-NoiseLev));
Fac   = [];
Sc    = eye(Comp);

%Create gaussian curves the first Comp/2 factors consist of 2 overimposed gaussian
for i = 2:length(Dims)
   if Dims(i) < Comp*1.5
      T = randperm(Comp*1.5);
   else
      T = randperm(Dims(i));
   end
   for j=1:Comp
      [y,varargout{i}(j,:)] = gacu([1 Dims(i)],Dims(i),T(j),Dims(i)*0.15*(rand(1)+0.1));
   end
   Ex = varargout{i};
   for j = 1:fix(Comp*0.5);
      [b,y]               = gacu([1 Dims(i)],Dims(i),T(Comp+j),Dims(i)*0.6*(rand(1)+0.1));
      [varargout{i}(j,:)] = Ex(j,:) + y;
   end
   Fict = sqrt(diag(sum((varargout{i}').^2)));
   varargout{i} = varargout{i}'*Fict^-1;
end

%Generate first mode loadings, the numbers are drawn from a [0,1] distribution
varargout{1}                               = rand(Dims(1),Comp);
t                                          = randperm(Dims(1)*Comp);
varargout{1}(t(1:ceil(0.05*Dims(1)*Comp))) = 0;

%Apply a suitable strategy for giving the proper congruence to the factors
Corr                                       = Corr * ones(Comp) + (1 - Corr)*eye(Comp);
for i=1:length(Dims)
   varargout{i} = orth(varargout{i}) * chol(Corr);
   if i == 1
      varargout{i} = varargout{i} * 200;
   end
end
X = nmodel(varargout);

%Generate noise
%It is not optimal because the total noise expressed as a percentage will be slightly smaller
%than the requested value. In general the approximation is close enough

%Homoscedastic noise
[I,J]  = size(nshape(X,1));
Noise  = randn(I,J);
Noise  = reshape(Noise,size(X)) / sqrt(sum(Noise(:).^2));
Noise  = Noise * NoiseLev(1) * sqrt(sum(X(:).^2));

%Heteroscedastic noise
Noise2  = randn(I,J);
Noise2 = (reshape(Noise2,size(X)) .* X);
Noise2 = Noise2 / sqrt(sum(Noise2(:).^2));
Noise2 = Noise2 * NoiseLev(2) * sqrt(sum(X(:).^2));

%Sum underlying model to noise
X      = X + Noise + Noise2;

%Save seed if internally generated
if nargin < 5
   varargout = [{Seed},varargout(:)'];
end

%------------------------------------------------------------------------------------------------

function [x,y]=gacu(Int,Len,Mu,Sigma)
%Generates a gaussian curve of mean Mu and standard deviation Sigma
%The x values (Len values between Int(1) and Int(2)) are given in x.
%y contains the values of the curve.
x = linspace(Int(1),Int(2),Len);
y = sqrt(2 * pi * Sigma)^-1 * exp(-(x - Mu).^2 / (2 * Sigma^2));

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

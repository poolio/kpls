function [beta,testY, alpha,varargout] = dualpls(K,Ktest,Y,T,varargin)

%function [beta,testY,alpha,varargout] = dualpls(K,Y,Ytest,T,varargin)
%
% Performs dual (kernel) PLS discrimination
%
%INPUTS
% K = the training kernel matrix
% Ktest = the kernel matrix (dimension ell x elltest)
% Y = the training label matrix (ell x m)
% T = the number of PLS components to take
% varargin = optional argument specifying the true test label matrix
%            of size elltest x m, if 
%
%OUTPUTs
% alpha = the dual vectors corresponding to the PLS classfier
% testY = the estimated label matrix for the test samples
% varargout = the test error, optional when varargin is specified (the
%             true test labels)
%
%
%For more info, see www.kernel-methods.net
%
%Note: this code has not been tested extensively.

% K is an ell x ell kernel matrix
% Y is ell x m containing the corresponding output vectors
% T gives the number of iterations to be performed
ell=size(K,1);
trainY=0;
KK = K; YY = Y;
for i=1:T
    YYK = YY*YY'*KK;
    beta(:,i) = YY(:,1)/norm(YY(:,1));
    if size(YY,2) > 1, % only loop if dimension greater than 1
        bold = beta(:,i) + 1;
        while norm(beta(:,i) - bold) > 0.001,
            bold = beta(:,i);
            tbeta = YYK*beta(:,i);
            beta(:,i) = tbeta/norm(tbeta);
        end
    end
    tau(:,i) = KK*beta(:,i);
    val = tau(:,i)'*tau(:,i);
    c(:,i) = YY'*tau(:,i)/val;
    trainY = trainY + tau(:,i)*c(:,i)';
    trainerror = norm(Y - trainY,'fro')/sqrt(ell);
    w = KK*tau(:,i)/val;
    KK = KK - tau(:,i)*w' - w*tau(:,i)' + tau(:,i)*tau(:,i)'*(tau(:,i)'*w)/val;
    YY = YY - tau(:,i)*c(:,i)';
end

% Regression coefficients for new data
alpha = beta * ((tau'*K*beta)\tau')*Y;

%  Ktest gives new data inner products as rows, Ytest true outputs
if ~isempty(Ktest)
elltest = size(Ktest',1);
testY = Ktest' * alpha;
if ~isempty(varargin)
    Ytest = varargin{1};
    testerror = norm(Ytest - testY,'fro')/sqrt(elltest)
    varargout = testerror;
end
end

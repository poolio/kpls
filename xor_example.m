% Example comparing Kernel PCA and Kernel PLS on a toy dataset
% resembling an XOR operation.

function xor_example()
  n = 1000;
  d = 5;
  s2 = 1;
  T = 4;
  offset = 2;
  s= 1;

  [Xtrain, ytrain, Xtest, ytest] = xor_dataset(n,d,offset, s);

  % Build kernel
  k = @(x1,x2) exp(-sum((x2-repmat(x1,[1 size(x2,2)])).^2,1)/(2*s2));
  Ktrain = build_kernel(Xtrain, Xtrain, k);

  D = 2;
  for T=D%1:D
     T
    [beta] = dualpls(Ktrain,[], ytrain,T);
    [alpha] = dualpca(Ktrain, [], T);
    err_pca(T) = evaluate_kernel(Ktrain*alpha, ytrain);
    err_pls(T) = evaluate_kernel(Ktrain*beta, ytrain);
    alphas{T} = alpha;
    betas{T} = beta;
  end
  y = Ktrain*betas{2};
  y2 = Ktrain*alphas{2};
  figure,scatter(Xtrain(1,:),Xtrain(2,:),[],ytrain)
  figure,scatter(y(:,1),y(:,2),[],ytrain)
  figure,scatter(y2(:,1), y2(:,2),[],ytrain)
  keyboard
end

function [Xtrain, ytrain, Xtest, ytest] = xor_dataset(n,d,offset, s)
  X = [randn(n,1)-offset, randn(n,1)-offset; randn(n,1)+offset, randn(n,1)+offset; randn(n,1)-offset, randn(n,1)+offset; randn(n,1)+offset, randn(n,1)-offset];
  X(:,3:d) = s * randn(4*n,d-2);
  y = [zeros(2*n,1); ones(2*n,1)];
  Xtrain = X(1:2:end,:)';
  ytrain = y(1:2:end);
  Xtest = X(2:2:end-2,:)';
  ytest = y(1:2:end-2);
end

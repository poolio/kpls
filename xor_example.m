% Example comparing Kernel PCA and Kernel PLS on a toy dataset
% resembling an XOR operation.

function xor_example()
  n = 1000;
  d = 5;
  s2 = 1;
  T = 4;
  offset = 1;
  X = [randn(n,1)-offset, randn(n,1)-offset; randn(n,1)+offset, randn(n,1)+offset; randn(n,1)-offset, randn(n,1)+offset; randn(n,1)+offset, randn(n,1)-offset];


  s = 2;
  svals = [0 1 sqrt(5) sqrt(10)];
  for sidx=1:length(svals)
    s = svals(sidx);
  X(:,3:d) = s * randn(4*n,d-2);
  y = [zeros(2*n,1); ones(2*n,1)];

  Xtrain = X(1:2:end,:)';
  ytrain = y(1:2:end);
  Xtest = X(2:2:end-2,:)';
  ytest = y(1:2:end-2);

  % Build kernel
  k1 = kfnc_s(s2);
  %k1 = @(x1, x2) (x2'*x1);
  %load K1
  Ktrain = buildk(Xtrain, Xtrain, k1);
  Ktest = buildk(Xtest, Xtrain, k1)';

  Ktest2 = Ktest;
  e = size(Ktest2,1);
  for i=1:size(Ktest2,2)
    Ktest2(e+1,i) = k1(Xtest(:,i), Xtest(:,i));
  end
  D = 2;
  for T=D%1:D
    T
  [alpha1,testYt, beta] = dualpls(Ktrain, Ktest, ytrain,T,ytest);
  [alpha, L, Knew, Ktestnew, Ktestvstest] = dualpca(Ktrain, Ktest2, T);
  err_pca(T) = get_acc(Ktrain*alpha, ytrain);
  err_pls(T) = get_acc(Ktrain*beta, ytrain);
  alphas{T} = alpha;
  betas{T} = beta;
  end
  pca_errs{sidx} = err_pca;
  pls_errs{sidx} = err_pls;
  pca_alphas{sidx} = alphas;
  pls_betas{sidx} = betas;
  end
  y = Ktrain*betas{2};
  y2 = Ktrain*alphas{2};
  figure,scatter(Xtrain(1,:),Xtrain(2,:),[],ytrain)
  figure,scatter(y(:,1),y(:,2),[],ytrain)
  figure,scatter(y2(:,1), y2(:,2),[],ytrain)
  keyboard
end

function err=get_acc(X, y)
  X = [ones(size(X,1),1) X];
  b = regress(y,X);
  ypred = X*b;
  err=1/size(X,1) * sum((ypred-y).^2);
end

function k = kfnc_s(s2)
  k = @(x1,x2) exp(-sum((x2-repmat(x1,[1 size(x2,2)])).^2,1)/(2*s2));
end

function K=buildk(data,train_data,k)
  n = size(data,2);
  K = zeros(n,size(train_data,2));
  for i=1:size(data, 2)
    if mod(i,100) == 0
      i
    end
    K(i,:) = k(data(:,i), train_data);
  end
end


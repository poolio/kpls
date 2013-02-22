function err=evaluate_kernel(X, y)
  X = [ones(size(X,1),1) X];
  b = regress(y,X);
  ypred = X*b;
  err=1/size(X,1) * sum((ypred-y).^2);
end


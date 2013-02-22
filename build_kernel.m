function K=build_kernel(data,train_data,k)
  n = size(data,2);
  K = zeros(n,size(train_data,2));
  for i=1:size(data, 2)
    if mod(i,100) == 0
      i
    end
    K(i,:) = k(data(:,i), train_data);
  end
end


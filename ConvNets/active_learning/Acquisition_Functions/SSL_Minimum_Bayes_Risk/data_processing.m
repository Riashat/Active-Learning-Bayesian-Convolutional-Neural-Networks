load('dataset.mat')
X_train = mean(X_train, 2);
X_test = mean(X_test, 2);

[y1, ~] = find(y_train==1);
[y2, ~] = find(y_train==2);

y2 = y2(1:length(y1), :);

X1 = X_train(y1, :, :, :);
X2 = X_train(y2, :, :, :);

% take for hypothetical example - image data set is (1000, 64);
% each image being 64 dimensional 
sigma = 0.5;
data = rand(2000, 64);      % assuming 16x16 downsampled images
W = zeros(size(data,1), size(data,1));
for i = 1:size(data, 1)
    for j = 1:size(data,1)
        W(i,j) = exp((-norm(data(i,:)' - data(j,:)'.^2)./2*sigma.^2));
    end
end

d = zeros(size(W,1), 1);
for k = 1:size(W,1)
    d(k,1) = sum(W(k,:),2);
end

D = diag(d);

Delta = D - W;
Delta_ll = Delta(1:500, 1:500);
Delta_lu = Delta(1:500, 501:2000);
Delta_ul = Delta(501:2000, 1:500);
Delta_uu = Delta(501:2000, 501:2000);

% Compute f

% for is a vector for each i



save 'Processed_Results.mat'

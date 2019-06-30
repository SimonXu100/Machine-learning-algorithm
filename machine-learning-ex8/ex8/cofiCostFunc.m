function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% unregularized cost function
M = (X*Theta' - Y).^2;
J  = 0.5 * (sum(sum(R.*M)));
 
 
% unregularized gradient
for i=1:num_movies
  idx = find(R(i,:)==1);
  Thetatemp = Theta(idx, :);
  Ytemp = Y(i, idx);
  X_grad(i, :) = ( X(i,:) * Thetatemp' - Ytemp)* Thetatemp;
  
endfor

for j=1:num_users

  idx = find(R(:,j)==1);
  Xtemp = X(idx,:);
  Ytemp = Y(idx,j);
  Theta_grad(j,:) = ( Theta(j,:) * Xtemp' - Ytemp' ) * Xtemp;
  
endfor

% advanced vectorization


% regularized cost function
J = J + lambda * 0.5 *(sum(sum(X.^2))+ sum(sum(Theta.^2)));


% regularizated gradient
for i=1:num_movies

  X_grad(i, :) =   X_grad(i, :) + lambda .* X(i,:);
  
endfor

for j=1:num_users

  Theta_grad(j,:) = Theta_grad(j,:) + lambda * Theta(j,:)
  
endfor







% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

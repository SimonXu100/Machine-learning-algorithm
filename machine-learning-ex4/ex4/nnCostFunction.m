function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% hypothesis
h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% cost function without regularization term
% adpoting for-loop over examples here
% advanced vertorization: diagnostic searching

% adpoting for-loop over examples here
% recode y: 转化为最大的
sum1 = 0;
for i = 1:m
  yVec = 1:num_labels;
  yVec = (yVec == y(i,1));
  h2Vec = h2(i,:);
  sum1 = sum1 + (-(yVec * log(h2Vec')) - (1 - yVec) * log(1-h2Vec'));
end
J = sum1 / m;


% regularization
Theta1_r = Theta1(:,2:(input_layer_size+1));
Theta2_r = Theta2(:,2:(hidden_layer_size+1));
% name confliction
%sum(1:10);
%Theta1_double = Theta1_r.^2;
%Theta2_double = Theta2_r.^2;
%temp1 = sum((Theta1_r.^2));
%temp2 = sum((Theta2_r.^2));
%Theta1_sum = sum(temp1);
%Theta2_sum = sum(temp2);
%J = J + lambda *( Theta1_sum + Theta2_sum) / (2*m);
J = J + lambda*( sum(sum(Theta1_r.^2)) + sum(sum(Theta2_r.^2))) / (2*m);

% graident: backpropagation
%remove delta0
delta_3 = zeros(1,num_labels);
delta_2 = zeros(1,hidden_layer_size);
%delta_1 = zeros(1,input_layer_size);
Dvec1 = zeros(size(Theta1));
Dvec2 = zeros(size(Theta2));
for t = 1:m
 z2= [1, X(t,:)] * Theta1';
 a2= [1,sigmoid(z2)];
 z3= a2*Theta2'
 a3 = sigmoid(z3);
 
 yVec = 1:num_labels;
 yVec = (yVec == y(t,1));
 delta_3 = a3 - yVec;
 % 1 * 26
 delta_2 = delta_3 * Theta2.*sigmoidGradient([1,z2]);
 %1 * 25
 delta_2 = delta_2(1,2:end);

 Dvec2 = Dvec2 + delta_3' * a2;
 Dvec1 = Dvec1 + delta_2' * [1, X(t,:)];
endfor
Theta1_grad = Dvec1 / m;
Theta2_grad = Dvec2 / m;

% regularized back propagation
Theta1_grad(:,1) = Dvec1(:,1) / m;
Theta1_grad(:,2:end) = ( Dvec1(:,2:end) + lambda * Theta1(:,2:end))/ m;

Theta2_grad(:,1) = Dvec2(:,1) / m;
Theta2_grad(:,2:end) = ( Dvec2(:,2:end) + lambda * Theta2(:,2:end))/ m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

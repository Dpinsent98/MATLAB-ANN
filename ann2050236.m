function [Yn On Yt Ot] = ann2050236(ID, N_ep, lr, bp, u, v, w, cf, gfx, Nh)
% MA2647: Artificial Neural Network, assignment Spring 2023 (MNIST data).
%
% Usage:
%    [Yn On Yt Ot] = ann2050236(ID, N_ep, lr, bp, u, v, w, cf, gfx, Nh)
%
% Where, for the MNIST data sets,
% - ID is stdent ID (e.g. 2216789)
% - N_ep the number of epochs, lr is the learning rate
% - bp = 1, heuristic unscaled, 2 heuristic scaled, 3 calculus based bp
% - u, v, w the number of nodes on the three hidden layers, u is used for
% one hidden layer
% - cf = 1 use total squared error; 2 use cross-entropy
% - gfx = 1 display intermediate graphics; 2 dont display graphics
% - Nh = 1 One hidden layer; 2 Three Hidden Layers
% - Yn are the exact training labels/targets, Ot the training predictions
% - Yt are the exact testing  labels/targets, On the testing  predictions
% The MNIST CSV files are not to be altered in any way, and should be
% in the same folder as this matlab code.
%
% [Yn On Yt Ot] = ann2050236(2050236,50,0.025,1,10,1,1,1,1,1)
%
% This code is based on the book Make Your Own Neural Network by Tariq
% Rashid (CreateSpace Independent Publishing Platform (31 Mar. 2016),
% ISBN-10: 1530826608; ISBN-13: 978-1530826605)

% set some useful defaults
set(0,'DefaultLineLineWidth', 2);
set(0,'DefaultLineMarkerSize', 10);

% REMOVE THIS EVENTUALLY

% As a professional touch we should test the validity of our input
if or(N_ep <= 0, lr <= 0)
  error('N_ep and/or lr are not valid')
end
if ~ismember(bp,[1,2,3])
  error('back prop choice is not valid')
end
if ~all([u,v,w] > 0)
  error('u,v,w values have an error')
end
if ~ismember(cf,[1,2])
  error('performance index choice is not valid')
end

if Nh == 1
% obtain the training data
A = readmatrix('mnist_train_1000.csv');
% convert and NORMALIZE it into training inputs and target outputs
X_train = A(:,2:end)'/255;  % beware - transpose, data is in columns!
N_train = size(X_train,2); % size(X_train,1/2) gives number of rows/columns
Y_train = zeros(2,N_train);
% set up the one-hot encoding: Y(1,:)=1 if ID contains, else y(2,:)=1
for i=1:N_train
  if contains(int2str(ID),int2str(A(i,1)))
    Y_train(1,i) = 1;
%    fprintf('YES: ID = %d, value = %d\n', ID, A(i,1)')
  else
    Y_train(2,i) = 1;
%    fprintf('NO:  ID = %d, value = %d\n', ID, A(i,1))
  end  
end

% default variables
Ni   = 784;             % number of input nodes
N1   = u;               % number of nodes on first hidden layer
N2   = v;               % number of nodes on second hidden layer
N3   = w;               % number of nodes on third hidden layer
No   = 2;               % number of output nodes
% set up weights and biases
W2 = 0.5-rand(784,N1); b2 = zeros(N1,1);
W3 = 0.5-rand(N1, 2); b3 = zeros(2,1);
% set up a sigmoid activation function for layers 2, 3, 4, 5
sig2  = @(x) 1./(1+exp(-x));
dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
% output layer activation depends on cf
if cf == 1;
  sig3  = @(x) 1./(1+exp(-x));
  dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
elseif cf == 2;
  sig3 = @(x) exp(x)/sum(exp(x));
else
  error('cf has improper value')
end

% we'll calculate the performance index at the end of each epoch
pivec = zeros(1,N_ep);  % row vector

% we now train by looping N_ep times through the training set
for epoch = 1:N_ep
  mixup = randperm(N_train);
  for j = 1:N_train
    i = mixup(j);
    % get X_train(:,i) as an input to the network
    a1 = X_train(:,i);
    % forward prop to the next layer, activate it, repeat
    n2 = W2'*a1 + b2; a2 = sig2(n2);
    n3 = W3'*a2 + b3; a3 = sig3(n3);
    % this is then the output
    y = a3;

    % calculate A, the diagonal matrices of activation derivatives
    A2 = diag(dsig2(n2));
    % we calculate the error in this output, and get the S3 vector
    e3 = Y_train(:,i) - y;
    if cf == 1;
      A3 = diag(dsig3(n3)); S3 = -2*A3*e3;
    elseif cf == 2;
      S3 = -e3;
    end  
    % back prop the error
    if     bp == 1; e2 = W3*e3; S2 = -2*A2*e2;
    elseif bp == 2; e2 = W3*diag(1./sum(W3))*e3; S2 = -2*A2*e2;
    elseif bp == 3; S2 = A2*W3*S3;
    end 
    
    % and use a learning rate to update weights and biases

    W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
    W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
  end
  % calculate the sum of squared errors and store for plotting
  for i=1:N_train
    y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
    if cf == 1;
      err = Y_train(:,i) - y;
      % each error is itself a vector - hence the norm ||err||
      pivec(epoch) = pivec(epoch) + norm(err,2)^2;
    elseif cf == 2
      xent = -sum(Y_train(:,i).*log(y));
      pivec(epoch) = pivec(epoch) + xent;
    end   
  end
end


% add a subplot; plot the performance index vs (row vector) epochs
if gfx ~= 0
  subplot(2,2,1)
  plot([1:N_ep],pivec,'b');
  xlabel('epochs'); ylabel('performance index');
end

% loop through the training set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(2,N_train);
for i = 1:N_train
  y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
  y_pred(:,i) = y; 
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_train(:,i));
  barcol = 'r';
  if indx1 == indx2; wins = wins+1; barcol = 'b'; end 
  if gfx ~= 0
    % plot the output 2-vector at top right
    subplot(2,2,2); bar(0:1,y_pred(:,i),barcol); ylim([0,1]);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    B = reshape(1-X_train(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(B','InitialMagnification','fit');
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_train]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)    
    drawnow
  end
end
fprintf('training set wins = %d/%d, %f%%\n',wins,N_train,100*wins/N_train)

% assign outputs
Yn=Y_train; On=y_pred;

% obtain the test data
B = readmatrix('mnist_test_100.csv');
% convert and NORMALIZE it into testing inputs and target outputs
X_test = B(:,2:end)'/255;
% the number of data points
N_test = size(X_test,2);
Y_test = zeros(2,N_test);
% set up the one-hot encoding: Y(1,:)=1 if ID contains, else y(2,:)=1
for i=1:N_test
  if contains(int2str(ID),int2str(B(i,1)))
    Y_test(1,i) = 1;
  else
    Y_test(2,i) = 1;
  end  
end

% loop through the test set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(2,N_test);
for i = 1:N_test

  y = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
  y_pred(:,i) = y;
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_test(:,i));
  barcol = 'r';
  if indx1 == indx2; wins = wins+1; barcol = 'b'; end 
  if gfx ~= 0
    % plot the output 10-vector at top right
    subplot(2,2,2); bar(0:1,y_pred(:,i),barcol); ylim([0,1]);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    B = reshape(1-X_test(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(B','InitialMagnification','fit');
    % animate the wins and losses bottom right
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_test]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)
    drawnow;
    pause(0.01)
  end
end
fprintf('testing  set wins = %d/%d, %f%%\n',wins,N_test,100*wins/N_test)

% assign outputs
Yt=Y_test; Ot=y_pred;

figure(1)
plotconfusion(Yn, On,'Training Data Confusion Matrix')
figure(2)
plotconfusion(Yt, Ot,'Test Data Confusion Matrix')  

elseif Nh == 2    
% obtain the training data
A = readmatrix('mnist_train_1000.csv');
% convert and NORMALIZE it into training inputs and target outputs
X_train = A(:,2:end)'/255;  % beware - transpose, data is in columns!
N_train = size(X_train,2); % size(X_train,1/2) gives number of rows/columns
Y_train = zeros(2,N_train);
% set up the one-hot encoding: Y(1,:)=1 if ID contains, else y(2,:)=1
for i=1:N_train
  if contains(int2str(ID),int2str(A(i,1)))
    Y_train(1,i) = 1;
%    fprintf('YES: ID = %d, value = %d\n', ID, A(i,1)')
  else
    Y_train(2,i) = 1;
%    fprintf('NO:  ID = %d, value = %d\n', ID, A(i,1))
  end  
end

% default variables
Ni   = 784;             % number of input nodes
N1   = u;               % number of nodes on first hidden layer
N2   = v;               % number of nodes on second hidden layer
N3   = w;               % number of nodes on third hidden layer
No   = 2;               % number of output nodes
% set up weights and biases
W2 = 0.5-rand(784,N1); b2 = zeros(N1,1);
W3 = 0.5-rand(N1, N2); b3 = zeros(N2,1);
W4 = 0.5-rand(N2, N3); b4 = zeros(N3,1);
W5 = 0.5-rand(N3, 2); b5 = zeros(2,1);
% set up a sigmoid activation function for layers 2, 3, 4, 5
sig2  = @(x) 1./(1+exp(-x));
dsig2 = @(x) exp(-x)./((1+exp(-x)).^2);
sig3  = @(x) 1./(1+exp(-x));
dsig3 = @(x) exp(-x)./((1+exp(-x)).^2);
sig4  = @(x) 1./(1+exp(-x));
dsig4 = @(x) exp(-x)./((1+exp(-x)).^2);
% output layer activation depends on cf
if cf == 1;
  sig5  = @(x) 1./(1+exp(-x));
  dsig5 = @(x) exp(-x)./((1+exp(-x)).^2);
elseif cf == 2;
  sig5  = @(x) exp(x)/sum(exp(x));
else
  error('cf has improper value')
end

% we'll calculate the performance index at the end of each epoch
pivec = zeros(1,N_ep);  % row vector

% we now train by looping N_ep times through the training set
for epoch = 1:N_ep
  mixup = randperm(N_train);
  for j = 1:N_train
    i = mixup(j);
    % get X_train(:,i) as an input to the network
    a1 = X_train(:,i);
    % forward prop to the next layer, activate it, repeat
    n2 = W2'*a1 + b2; a2 = sig2(n2);
    n3 = W3'*a2 + b3; a3 = sig3(n3);
    n4 = W4'*a3 + b4; a4 = sig4(n4);
    n5 = W5'*a4 + b5; a5 = sig5(n5);
    % this is then the output
    y = a5;
    
    % calculate the Ai, the diagonal matrices of activation derivatives
    A4 = diag(dsig4(n4)); A3 = diag(dsig3(n3)); A2 = diag(dsig2(n2));
    % we calculate the error in this output, and get the S3 vector
    e5 = Y_train(:,i) - y;
    if cf == 1;
      A5 = diag(dsig5(n5)); S5 = -2*A5*e5;
    elseif cf == 2;
      S5 = -e5;
    end  
    % back prop the error
    if bp == 1
      e4 = W5*e5; e3 = W4*e4; e2 = W3*e3;
      % from these we compute the S vectors
      S4 = -2*A4*e4; S3 = -2*A3*e3; S2 = -2*A2*e2;
    elseif bp == 2
      e4 = W5*diag(1./sum(W5))*e5;
      e3 = W4*diag(1./sum(W4))*e4;
      e2 = W3*diag(1./sum(W3))*e3;
      % from these we compute the S vectors
      S4 = -2*A4*e4; S3 = -2*A3*e3; S2 = -2*A2*e2;
    elseif bp == 3
      S4 = A4*W5*S5; S3 = A3*W4*S4; S2 = A2*W3*S3;
    end 
    
    % and use a learning rate to update weights and biases
    W5 = W5 - lr * a4*S5'; b5 = b5 - lr * S5;
    W4 = W4 - lr * a3*S4'; b4 = b4 - lr * S4;
    W3 = W3 - lr * a2*S3'; b3 = b3 - lr * S3;
    W2 = W2 - lr * a1*S2'; b2 = b2 - lr * S2;
  end
  % calculate the sum of squared errors and store for plotting
  for i=1:N_train
    y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
    y = sig5(W5'*sig4(W4'*y + b4) + b5);
    if cf == 1;
      err = Y_train(:,i) - y;
      % each error is itself a vector - hence the norm ||err||
      pivec(epoch) = pivec(epoch) + norm(err,2)^2;
    elseif cf == 2
      xent = -sum(Y_train(:,i).*log(y));
      pivec(epoch) = pivec(epoch) + xent;
    end   
  end
end


% add a subplot; plot the performance index vs (row vector) epochs
if gfx ~= 0
  subplot(2,2,1)
  plot([1:N_ep],pivec,'b');
  xlabel('epochs'); ylabel('performance index');
end

% loop through the training set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(2,N_train);
for i = 1:N_train
  y = sig3(W3'*sig2(W2'*X_train(:,i) + b2) + b3);
  y_pred(:,i) = sig5(W5'*sig4(W4'*y + b4) + b5);
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_train(:,i));
  barcol = 'r';
  if indx1 == indx2; wins = wins+1; barcol = 'b'; end 
  if gfx ~= 0
    % plot the output 2-vector at top right
    subplot(2,2,2); bar(0:1,y_pred(:,i),barcol); ylim([0,1]);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    B = reshape(1-X_train(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(B','InitialMagnification','fit');
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_train]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)    
    drawnow
  end
end
fprintf('training set wins = %d/%d, %f%%\n',wins,N_train,100*wins/N_train)

% assign outputs
Yn=Y_train; On=y_pred;

% obtain the test data
B = readmatrix('mnist_test_100.csv');
% convert and NORMALIZE it into testing inputs and target outputs
X_test = B(:,2:end)'/255;
% the number of data points
N_test = size(X_test,2);
Y_test = zeros(2,N_test);
% set up the one-hot encoding: Y(1,:)=1 if ID contains, else y(2,:)=1
for i=1:N_test
  if contains(int2str(ID),int2str(B(i,1)))
    Y_test(1,i) = 1;
  else
    Y_test(2,i) = 1;
  end  
end

% loop through the test set and evaluate accuracy of prediction
wins = 0;
y_pred = zeros(2,N_test);
for i = 1:N_test

  y = sig3(W3'*sig2(W2'*X_test(:,i) + b2) + b3);
  y_pred(:,i) = sig5(W5'*sig4(W4'*y + b4) + b5);
  [~, indx1] = max(y_pred(:,i));
  [~, indx2] = max(Y_test(:,i));
  barcol = 'r';
  if indx1 == indx2; wins = wins+1; barcol = 'b'; end 
  if gfx ~= 0
    % plot the output 10-vector at top right
    subplot(2,2,2); bar(0:1,y_pred(:,i),barcol); ylim([0,1]);
    title('predicted output (approximate one-hot)')
    % plot the MNIST image at bottom left
    B = reshape(1-X_test(:,i),[28,28]);
    subplot(2,2,3); imh = imshow(B','InitialMagnification','fit');
    % animate the wins and losses bottom right
    subplot(2,2,4);
    b = bar(categorical({'Wins','Losses'}), [wins i-wins]);
    ylim([0,N_test]);
    b.FaceColor = 'flat'; b.CData(1,:) = [1 0 0]; b.CData(2,:) = [0 0 1];
    a = get(gca,'XTickLabel'); set(gca,'XTickLabel',a,'fontsize',18)
    drawnow;
    pause(0.01)
  end
end
fprintf('testing  set wins = %d/%d, %f%%\n',wins,N_test,100*wins/N_test)

% assign outputs
Yt=Y_test; Ot=y_pred;

figure(1)
plotconfusion(Yn, On,'Training Data Confusion Matrix')
figure(2)
plotconfusion(Yt, Ot,'Test Data Confusion Matrix')  

end
end

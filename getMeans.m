function [ M ] = getMeans( Des, K )
%Classification Image recognition
%   Detailed explanation goes here

% ims  :  number of images per class to use for building the dictionary
% K    :  size of the visual dictionary (number of visual words/clusters)
% T    :  number of images per class to use for training the SVMs
% k    :  type of kernel for the SVMs
%         0 for linear
%         1 for multinomial
%         2 for radial basis
%         3 for sigmoid

%E      =  50; % number of test images per class
flag   =  1; % controls where to use built-in kmeans of matlab or vl_kmeans
%               for building the visual dictionary
%              0 for kmeans
%              1 for vl_kmeans

tic


% K-MEANS CLUSTERING
fprintf('Building visual dectionary of %d visual words...\n\n', K);

if flag == 0
    % M is the visual dictionary, each row is a visual word
    % M : [ K x 128] matrix with the components of the cluster means
    [~, M] = kmeans(Des', K, 'Display', 'iter');
else
    [M, ~] = vl_kmeans(Des, K, 'distance', 'l2', 'algorithm', 'ann');
    M = M'; % M : [ K x 128] matrix with the components of the cluster means
end


toc
end


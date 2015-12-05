function [ TAP, APs, lists ] = rest(ims, method, M, T, k, s )
%Classification Image recognition
%   Detailed explanation goes here

% ims  :  number of images per class to use for building the dictionary
% K    :  size of the visual dictionary (number of visual words/clusters)
% T    :  number of images per class used for training the SVMs
% k    :  type of kernel for the SVMs
%         0 for linear
%         1 for multinomial
%         2 for radial basis
%         3 for sigmoid
%s     : color space
%         1 for RGB
%         2 for opponent colors

E      =  50; % number of test images per class
%flag   =  1; % controls whether to use built-in kmeans of matlab or vl_kmeans
%              for building the visual dictionary
%           0 for kmeans
%           1 for vl_kmeans

% a little trick to construct " '-b 1 -t k' ". Althouth it can be passed
% as an argument into svmtrain (no runtime error) the training process
% completely ignores it like it was never used...
%kernel = horzcat(('''-b 1 -t '), int2str(k), '''');
tic


% Building the histograms of the visual words to be used for training
fprintf('Extracting descriptors and building histograms taking\n');
fprintf('%d random images per class...\n', T);

fprintf('Processing airplanes_train...\n');
h1 = getHists(ims, T, 'data/airplanes_train', M, method, s);

fprintf('Processing cars_train...\n');
h2 = getHists(ims, T, '../cars_train', M, method, s);

fprintf('Processing faces_train...\n');
h3 = getHists(ims, T, '../faces_train', M, method, s);

fprintf('Processing motorbikes_train...\n\n');
h4 = getHists(ims, T, '../motorbikes_train', M, method, s);

% final matrix of all the histograms H : [4*T x K]
H = [h1; h2; h3; h4];

% constructing labels matrix with four columns each column corresponds
% to labels -1 or 1 according to the respective class
labels = (-1) * ones(size(H,1), 4);
ind = sub2ind(size(labels), 1:size(labels,1), [ones(1,T), ones(1,T)*2, ones(1,T)*3, ones(1,T)*4]);
labels(ind) = 1;

% TRAINING the 4 binary classifiers with the user defined kernel
[c1, c2, c3, c4] = myTrain(labels, H, k);

% Building the histograms of the visual words to be used for testing
fprintf('\nGetting test histograms...\n');

fprintf('Processing airplanes_test...\n');
h11 = getHists2('../airplanes_test', M, E, method, s);

fprintf('Processing cars_test...\n');
h22 = getHists2('../cars_test', M, E, method, s);

fprintf('Processing faces_test...\n');
h33 = getHists2('../faces_test', M, E, method, s);

fprintf('Processing motorbikes_test...\n\n');
h44 = getHists2('../motorbikes_test', M, E, method, s);

% final matrix of all the histograms test_H : [4*E x K] ([200 x K] default)
test_H = [h11; h22; h33; h44];

% constructing a list with images
airplanes = num2cell( strcat('airplane', num2str((1:E)')), 2 );
cars =  num2cell( strcat('car', num2str((1:E)')), 2 );
faces = num2cell( strcat('face', num2str((1:E)')), 2);
motorbikes = num2cell( strcat('motorbike', num2str((1:E)')), 2);

List = [airplanes; cars; faces; motorbikes];

% constructing labels matrix with four columns each column corresponds
% to labels -1 or 1 according to the respective class
labels = (-1) * ones(size(test_H,1), 4);
ind = sub2ind(size(labels), 1:size(labels,1), [ones(1,E), ones(1,E)*2, ones(1,E)*3, ones(1,E)*4]);
labels(ind) = 1;

% Doing probabilistic predictions and building the four ranked lists
[~, ~, prob] = svmpredict(labels(:,1), test_H, c1, '-b 1');
[~, ind] = sort(prob(:,1), 'descend');
list1 = List(ind);

[~, ~, prob] = svmpredict(labels(:,2), test_H, c2, '-b 1');
[~, ind] = sort(prob(:,1), 'descend');
list2 = List(ind);

[~, ~, prob] = svmpredict(labels(:,3), test_H, c3, '-b 1');
[~, ind] = sort(prob(:,1), 'descend');
list3 = List(ind);

[~, ~, prob] = svmpredict(labels(:,4), test_H, c4, '-b 1');
[~, ind] = sort(prob(:,1), 'descend');
list4 = List(ind);

lists = [list1, list2, list3, list4];

% average precisions for every class
fprintf('Computing APs...\n');
av_prec_1 = getAP(list1, 'air', E);
av_prec_2 = getAP(list2, 'car', E);
av_prec_3 = getAP(list3, 'fac', E);
av_prec_4 = getAP(list4, 'mot', E);
APs = [av_prec_1, av_prec_2, av_prec_3, av_prec_4];

% Total Average Precision
TAP = (av_prec_1 + av_prec_2 + av_prec_3 + av_prec_4) / 4;

cd('../..');
toc
end


function im = myRead(image, s)
% reads an image converts to gray IF needed and converts to double
im = imread(image);
if s == 1
    if ndims(im) == 3
        im = rgb2gray(im);
    end
elseif s == 2
    if ndims(im) == 3
        R  = im(:,:,1);
        G  = im(:,:,2);
        B  = im(:,:,3);
        %convert to opponent space
        O1 = (R-G)./sqrt(2);
        O2 = (R+G-2*B)./sqrt(6);
        O3 = (R+G+B)./sqrt(3);
        im(:,:,1) = O1;
        im(:,:,2) = O2;
        im(:,:,3) = O3;
        im = rgb2gray(im);
    end
end
im = im2double(im);
end

function descriptors = mySift(image, method)
% returns descriptors for a single image using the method specified
switch method
    case 1
        [~, descriptors] = sift2(single(image));
    case 2
        [~, descriptors] = vl_dsift(single(image));
        descriptors = double(descriptors);
end

end

function images = browse(path)
% browses into a folder (path) and returns a list with strings of images's
% names

cd(path);
images = ls;
if strcmp(images(3, :), 'img001.jpg'); % avoiding some mishappenings
    images = images(3:size(images,1), :);
else
    images = images(4:size(images,1), :);
end

end


function hists = getHists(used, train_size, path, Means, method, s)
% returns normalized histograms of visual words
images = browse(path);
images = images(used+1 : size(images,1), :);
% Sample from the unused images of size train_size
sample = datasample(images, train_size, 1, 'Replace', false);
hists = zeros(train_size, size(Means,1));

for i = 1:train_size
    im1 = myRead(sample(i, :), s); 
    des = mySift(im1, method);
    
    % smart implementation of euclidean distances avoiding loops
    dd = sum(des'.^2, 2);
    dm = double(des') * double(Means');
    mm = sum(Means.^2, 2)';
    dists = sqrt(bsxfun(@plus, mm, bsxfun(@minus, dd, 2*dm)));
    
    % dists is a [L x K] matrix where L : number of descriptors and K :
    % number of clusters
    % in other words it contains the distances of all the descriptors to
    % all the means
    
    % I is a column vector of size L containing for every descriptor
    % its closest visual word index (cluster mean)
    [~, I] = min(dists,[],2);
    % h1 is column vector of size K containing the frequencies of the visual words
    h1 = histc(I, 1:size(Means,1));
    % normalizing h1
    h1 = h1/sum(h1);
    hists(i,:) = h1'; %matrix with histograms of all images
end

end

function hists = getHists2(path, Means, quantity, method, s)
% similar function to getHists
images = browse(path);
images = images(1:quantity, :);

hists = zeros(quantity, size(Means,1));

for i = 1:quantity % quantity represents the number of images for testing
    im1 = myRead(images(i, :), s); 
    des = mySift(im1, method);
    
    dd = sum(des'.^2, 2);
    dm = des' * Means';
    mm = sum(Means.^2, 2)';
    dists = sqrt(bsxfun(@plus, mm, bsxfun(@minus, dd, 2*dm)));
    
    % dists is a [L x K] matrix where L : number of descriptors and K :
    % number of clusters
    % in other words it contains the distances of all the descriptors to
    % all the means
    
    % I is a column vector of size L containing for every descriptor
    % its closest visual word index (cluster mean)
    [~, I] = min(dists,[],2);
    % h1 is column vector of size K containing the frequencies of the visual words
    h1 = histc(I, 1:size(Means,1));
    % normalizing h1
    h1 = h1/sum(h1);
    hists(i,:) = h1';
end

end

function [c1, c2, c3, c4] = myTrain(labels, H, kernel)
% trains a classifier with the corresponding kernel
switch kernel
    case 0
        c1 = svmtrain(labels(:,1), H, '-b 1 -t 0');
        c2 = svmtrain(labels(:,2), H, '-b 1 -t 0');
        c3 = svmtrain(labels(:,3), H, '-b 1 -t 0');
        c4 = svmtrain(labels(:,4), H, '-b 1 -t 0');
    case 1
        c1 = svmtrain(labels(:,1), H, '-b 1 -t 1');
        c2 = svmtrain(labels(:,2), H, '-b 1 -t 1');
        c3 = svmtrain(labels(:,3), H, '-b 1 -t 1');
        c4 = svmtrain(labels(:,4), H, '-b 1 -t 1');
    case 2
        c1 = svmtrain(labels(:,1), H, '-b 1 -t 2');
        c2 = svmtrain(labels(:,2), H, '-b 1 -t 2');
        c3 = svmtrain(labels(:,3), H, '-b 1 -t 2');
        c4 = svmtrain(labels(:,4), H, '-b 1 -t 2');
    case 3
        c1 = svmtrain(labels(:,1), H, '-b 1 -t 3');
        c2 = svmtrain(labels(:,2), H, '-b 1 -t 3');
        c3 = svmtrain(labels(:,3), H, '-b 1 -t 3');
        c4 = svmtrain(labels(:,4), H, '-b 1 -t 3');
end

end

function prec = getAP(list, class, elem)
% returns the average precision of a class
j = 1; 
prec = 0;
for i = 1:size(list, 1)
    if list{i}(1:3) == class;
        prec = prec + j/i;
        j = j + 1;
    end
end

prec = prec /  elem;    

end

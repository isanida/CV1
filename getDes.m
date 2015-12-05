function [ Des ] = getDes( ims, method, s )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

tic
% Descriptors
Des = [];

% FINDING DESCRIPTORS
fprintf('Finding descriptors using %d images per class...\n', ims);

fprintf('Processing airplanes_train...\n');
Des = [Des, getDescriptors('data/airplanes_train', ims, method, s)];

fprintf('Processing cars_train...\n');
Des = [Des, getDescriptors('../cars_train', ims, method, s)];

fprintf('Processing faces_train...\n');
Des = [Des, getDescriptors('../faces_train', ims, method, s)];

fprintf('Processing motorbikes_train...\n\n');
Des = [Des, getDescriptors('../motorbikes_train', ims, method, s)];

toc
end

function Des = getDescriptors(path, num_images, method, s)
% returns descriptors for the number of images specified in the folder path
% using the method for extracting them
Des = [];
images = browse(path);
for i = 1:num_images
    im1 = myRead(images(i, :), s);
    des = mySift(im1, method);
    Des = [Des, des]; % stacks the descriptors of all images into Des
end
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
        [~, descriptors] = sift(image);
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
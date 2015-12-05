function h1 = getHist( im, Means )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    [~, des] = sift(im);
    
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

end


function [NV] = indivNorm(V,p)
%indivNorm Return the p-norms of a set of vectors
%   V 3xn

[~,n] = size(V);

NV = zeros(1,n);

for i = 1:n
	NV(i) = norm(V(:,i),p);
end

end
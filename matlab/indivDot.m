function [dotUV] = indivDot(U,V)
%indivDot Return the dot products of a set of vectors
%   U 3xn, V 3x1

[~,n] = size(U);

dotUV = zeros(1,n);

for i = 1:n
	dotUV(i) = dot(U(:,i),V);
end

end
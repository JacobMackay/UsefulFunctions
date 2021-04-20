function [crossUV] = indivCross(U,V,swapmulti)
%indivCross Return the cross products of a set of vectors
%   U 3xn, V 3x1

if nargin==2
	swapmulti = 0;
end

if swapmulti==0
	
	[~,n] = size(U);
	
	crossUV = zeros(3,n);
	
	for i = 1:n
		crossUV(:,i) = cross(U(:,i),V);
	end
	
else
	[~,n] = size(V);
	
	crossUV = zeros(3,n);
	
	for i = 1:n
		crossUV(:,i) = cross(U,V(:,i));
	end
	
end

end


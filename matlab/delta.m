function [y] = delta(x,width)
	if nargin==1
		width = eps;
	end
	y = double(abs(x./width)<0.5);
	y(abs(x./width)==0.5) = 0.5;
end
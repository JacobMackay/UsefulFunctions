function [y] = tri(t,style)

	if nargin == 1
		style = 1;
		% Canonical base length = 2
	end

	t(abs(t)>=1./style) = 1./style;
	y = max(1-style*abs(t),0);

end


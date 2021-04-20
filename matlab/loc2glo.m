function [pg] = loc2glo(pl,M,r)
%loc2glo(pl,M,r) Transform a vector in local coordinate frame to global 
%frame
% OUTPUTS:
%	pg:	3xn Vector in global frame
% INPUTS:
%   pl: 3xn Vector in local frame
%	M:	3x3 Rotation matrix
%	r:	3x1 Translation

if nargin == 2
	r = [0;0;0];
end

pg = M*pl+r;

end


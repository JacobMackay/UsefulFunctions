function [pl] = glo2loc(pg,M,r)
%glo2loc(pg,M,r) Transform a point in global coordinate frame to local frame
% OUTPUTS:
%	pl:	3xn Points in global frame
% INPUTS:
%   pg: 3xn Points in local frame
%	M:	3x3 Rotation matrix
%	r:	3x1 Translation

if nargin == 2
	r = [0;0;0];
end

pl = M\(pg-r);

end
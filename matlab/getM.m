function M = getM(forwardVector, unitVector)
%getM Return a rotation matrix that transforms a direction vector into a
%forward vector
%	INPUT
%		forwardVector:	3x1 vector defining the 'forward direction'
%		typically along the xaxis
%		unitVector:		3x1 vector defining the facing direction to find
%		transform of
%	OUTPUT
%		M:	3x3 rotation matrix

	unitVector = unitVector+eps;

	v = cross(forwardVector,unitVector);
	ssc = [0, -v(3), v(2); 
		v(3), 0, -v(1);
		-v(2), v(1), 0];	% Skew symmetric cross product matrix
	M = eye(3) + ssc + ssc^2*(1./(1+dot(forwardVector,unitVector)));

end
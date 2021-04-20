function scat3(points,ptsz)
%scat3 Simple function to 3D scatter plot with colour based on z
%   INPUTS
%	points:	3xn points
%	ptsz:	optional point size

if nargs==1
	ptsz = 1;
end

scatter3(points(1,:),points(2,:),points(3,:),ptsz,points(3,:),'filled')

end


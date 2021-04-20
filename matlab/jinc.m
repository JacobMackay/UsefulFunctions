function [y] = jinc(x)
%jinc Summary of this function goes here
%   Detailed explanation goes here

y = besselj(0,pi*x)./(pi*x);
y(abs(x)<=eps) = 1;

end


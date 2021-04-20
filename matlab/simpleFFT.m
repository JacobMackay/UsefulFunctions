% function [frq, amp, phase] = simpleFFT( signal, ScanRate)
% Purpose: perform an FFT of a real-valued input signal, and generate the single-sided 
% output, in amplitude and phase, scaled to the same units as the input.
%inputs: 
%    signal: the signal to transform
%    ScanRate: the sampling frequency (in Hertz)
% outputs:
%    frq: a vector of frequency points (in Hertz)
%    amp: a vector of amplitudes (same units as the input signal)
%    phase: a vector of phases (in radians)

function [frq, amp, phase] = simpleFFT( signal, ScanRate)

	n = length(signal);														% [samp]
	z = fft(signal, n); %do the actual work									% [in*samp]

	%generate the vector of frequencies
	halfn = floor(n / 2)+1;													% [samp]
	deltaf = 1 / ( n / ScanRate);											% [Hz/samp]
	frq = (0:(halfn-1)) * deltaf;											% [Hz]

	% convert from 2 sided spectrum to 1 sided
	%(assuming that the input is a real signal)
	amp(1) = abs(z(1)) ./ (n);												% [in]
	amp(2:(halfn-1)) = abs(z(2:(halfn-1))) ./ (n / 2);						% [in]
	amp(halfn) = abs(z(halfn)) ./ (n);										% [in]
	phase = angle(z(1:halfn));												% [rad]
	
end
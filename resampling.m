function [vq]=resampling(x,t,tup,Fs,integerUp)

%% INTERPOLATION
F=griddedInterpolant(t,x);
%% UPSAMPLING
vq=F(tup);
%% BANDPASS
[vq,hd]=lowpass(vq,Fs,integerUp*Fs,'Steepness',0.5,'StopbandAttenuation',30);
%% DOWNSAMPLING
vq=downsample(vq,integerUp);
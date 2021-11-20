%This file is used for defining the attenuation functions by users. 
%The output of this file woould be the distance-dependent paraemeters, 
%including geomeritc spreading factor, whole-path anelastic attenuation 
%factor, and path duration factor.
%--------------------------------------------------------------------------
% DO NOT CHANGE ANY PARAMETER SYMBOLS

%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
f=getappdata(0,'F');                                %^^
beta=str2double(get(handles.SourceVs, 'String'));   %^^
R=str2double(get(handles.Distance, 'String'));      %^^
M=str2double(get(handles.Mw, 'String'));            %^^
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
%^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
% Example functions for Switzerland
% Whole-path anelastic attenuation factor

Q=270*f.^0.5; Q=Q.*beta; Q=Q.^(-1);

% Geometric spreading factor

%If there is a finite-fault factor, hFF=10^(-0.405+0.235*M);
% R=sqrt(hFF^2+R^2);

if R<=70
    G=R^(-1.11);
elseif R<=120
    G=(70^(0.41)/(70^1.11))*R^(-0.41);
else
    G=(70^(0.41)/70^(1.11))*(120^(1.38)/120^(0.41))*R^(-1.38);
end

% Path duration
if R<=100
    tdp=0.153*R;
else
    tdp=0.02*(R-100)+15.3;
end

%%
%References
%Edwards,B.,Fäh,D. and Giardini,D.(2011).Attenuation of seismic shear wave
%energy in Switzerland.Geophys.J.Int.185,967–984.

%Edwards,B. and Fäh,D.(2013).A Stochastic Ground-Motion Model for
%Switzerland. Bull.Seismo.Soc.Am.103(1),78-98.
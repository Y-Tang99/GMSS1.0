
%   This script is for stochastic similation of ground motions for a single
%   M-R combination. Plot for each step is provided for illustration or
%   check purpose

%--------------------------------------------------------------------------

% STEP1----Generation of band-limit Gaussian white noise

%--------------------------------------------------------------------------


NS=1;                      % Number of simulations
dt=0.002;                  % Time step, no larger than 0.02 s 

beta=3.5;                  % Source shear wave velocity, km/s
M=6;                       % Moment magnitude
R=30;                      % Distance if no specifically defined
sigma=200;                 % Stress drop, bars
roll=2.8;                  % Source density, g/cm^3 

M0=10^(1.5*M+16.05);       % (Hanks & Kanamori, 1979; Boore et al., 2014)
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));   % corner frequency (Brune,1970)
fa=10^(2.41-0.533*M);
et=10^(2.52-0.637*M);
fb=((fc^2-(1-et)*fa^2)/et)^(0.5); % (Boore et al., 2014).

%Ground motion duration considered in window function
% the weights of the two coener-frequency are 0.5,0.5
%path duration is 0.05*R

td=1/(2*fa)+1/(2*fb)+0.05*R;   

nseed=round(log2(td/dt))+1;

NT=2^nseed;
T=NT*dt;

t=dt:dt:T;
df=1/T;           % Frequency step
fl=1/T;           % low frequency limit
fh=1/(2*dt);      % high frequency limit--Nyquist Frequency
f=(1:1:NT/2)*df;
fbw=fh-fl;        % frequency band
fs=2*fh;          % sampling frequency

nt0(NS,NT)=zeros();
x(NS,NT)=zeros();
hd=design(fdesign.bandpass('N,F3dB1,F3dB2',4,fl,fh,fs),'butter');%butterworth filter

for m=1:1:NS
    x(m,:)=wgn(1,NT,1);             % Gaussian white noise 
    nt0(m,:)=filter(hd,x(m,:));
end

% normalize the noise to Gaussian distribution

nt0=nt0';
std0=std(nt0);
mean0=mean(nt0);
nt=(nt0-ones(NT,1)*mean0)./(ones(NT,1)*std0);  
nt=nt';
% Display STEP 1
figure(1)
subplot(211)
plot(t,nt0)
subplot(212)
plot(t,nt)

%%
%--------------------------------------------------------------------------

% STEP2----% window the white noise to shape a resembled accelerogram

%--------------------------------------------------------------------------
wt=exp((-0.4)*t.*(6/td))-exp((-1.2)*t.*(6/td));   % (Lam et al., 2000)
wts=wt.^2;
wtssum=sum(wts.*dt);
w=sqrt(2*(fbw)/wtssum);     % window scaling factor (Lam et al., 2000)
win=(wt.*w);
          
st(NS,NT)=zeros();          % windowed band-limited white noise
for m=1:1:NS
    st(m,:)=nt(m,:).*win;
end


%Fast Fourier Transform

As(NS,NT)=zeros();
anglek(NS,NT/2)=zeros();
for m=1:1:NS
    As(m,:)=fft(st(m,:));
    anglek(m,:)=angle(As(m,1:NT/2)); % only oomplex number has phase angle
end


Asf=abs(As);         % Fourier amplitude
Asf=Asf(:,1:NT/2)';  % remove the mirrow symmetry in Frequency domain

% normalize the signal by deviding the average value

Asfsum=sum(Asf);
Adjfac=Asfsum/(NT/2);
Adjfac=1./Adjfac;


Adj(NT/2,NS)=zeros();
for m=1:1:NT/2
    Adj(m,:)=Asf(m,:).*Adjfac;
end
Adj=Adj';
% Display STEP 2
figure(2)
subplot(321)
plot(t,wt)
title ('Window function--wt')
xlabel ('Time(sec)');ylabel ('Amplitude')
subplot(322)
plot(t,st)
title ('Windowed band-limited noise--st')
xlabel ('Time(sec)');ylabel ('Amplitude')
subplot(3,2,[3 4])
plot(f,Asf)
title ('Fourier transformed windowed niose--Asf')
xlabel ('Frequency(Hz)');ylabel ('Amplitude')
subplot(3,2,[5 6])
plot(f,Adj)
title ('Adjusted windowed niose--Adj')
xlabel ('Frequency(Hz)');ylabel ('Amplitude')


%%
%--------------------------------------------------------------------------

% STEP3----Target spectrum (seismological model) for a target region

%--------------------------------------------------------------------------
% The target spectrum can be customly defined by users
fm=0.033;                                    % (Atkinson & Boore, 1995)
Vs30=0.76;                                % here the unit is "km/s", NOT "m/s"

Ax=AB95(f,M,R,roll,beta,sigma,Vs30,fm);  % the final unit is cm/s


%Display STEP 3
figure(3)
loglog(f,Ax)
title ('Fourier Spectrum of Acceleration')
xlabel ('Frequency(Hz)');ylabel ('Fourier Amplitude(cm/s)')

%%
%--------------------------------------------------------------------------

% STEP4----obtain the synthetic accelerogram

%--------------------------------------------------------------------------

Aaf(NS,NT/2)=zeros();
for m=1:1:NS
    Aaf(m,:)=Adj(m,:).*Ax;
end

% the following procedure is very important to get the correct amplitude value
 
Aaf=Aaf';
Aa(NT/2,NS)=zeros();
for m=1:1:NT/2
    Aa(m,:)=(Aaf(m,:)/(NT/2)).*(1./(Adjfac));
end
Aa=Aa';

%inverse Fourier transfer into time domain

at00(NS,NT)=zeros();
for m=1:1:NS
    at00(m,:)=ifft(Aa(m,:).*exp(1i*(anglek(m,:))),NT,'nonsymmetric')*NT;
end
at0=real(at00);  % Only the real component is needed --> amplitude


% Following are Time Series Processing Procedures (TSPP)

% (1)Basedline Correction of acceleration

windowWidth=NT-1;    % Frame size, must be odd
polynomialOrder=6;   % Smooth with a Savitzky-Golay sliding polynomial filter
baseline(NS,NT)=zeros();
at1(NS,NT)=zeros();
for m=1:1:NS
    baseline(m,:) = sgolayfilt(at0(m,:), polynomialOrder, windowWidth);
    at1(m,:)=at0(m,:)-baseline(m,:);
end

% add time pad

tpad=20;          % Time pad is added to remove distorsion of 
                  % time-series at long periods
N=NT+2*tpad/dt;                  
T1=T+2*tpad;                   
t1=0:dt:(T1-dt);

% A filter is applied to make sure long period noise is removed
filterts(N)=zeros();
for i=1:1:(tpad+T)/dt
    filterts(i)=1;
end

padsize=[0,tpad/dt];
padval=0; 
direction1='pre';
direction2='post';
at2(NS,NT+tpad/dt)=zeros();
at(NS,N)=zeros();
for m=1:1:NS
    at2(m,:) = padarray(at1(m,:),padsize,padval,direction1); % pre timepad
    at(m,:) = padarray(at2(m,:),padsize,padval,direction2);  % post timepad
     at(m,:)=at(m,:).*filterts;         % filtered acceleration time series
end

pga=max(abs(at'));

% calculation of velocity time series
vt(NS,N)=zeros();
for m=1:1:NS
    vt(m,:)=cumtrapz(t1,at(m,:));    % velocity time history
    vt(m,:)=vt(m,:).*filterts;       % filtered velocity time series
end

pgv=max(abs(vt'));                   % unit "cm/s"


% calculation of displacement time series
Dt(NS,N)=zeros();
for m=1:1:NS
    Dt(m,:)=cumtrapz(t1,vt(m,:));    % displacement time history
    Dt(m,:)=Dt(m,:).*filterts;       % filtered displacement time series
end
pgd=max(abs(Dt'));


% calculation of PGA, PGV and PGD

PGA=geomean(pga/980);      % unit "g"
PGV=geomean(pgv);          % unit "cm/s"
PGD=geomean(pgd*10);       % unit "mm"

% Please note here I use "geomean" function instead of "median" or "mean" function to
% get the median/mean estimates because "geomean" can obtain more smooth curves.
% The above functions can give very close estimates if NS is large enough.

%Display STEP 4
figure(4)
subplot(221)
loglog(f,Aaf,'k',f,Ax)
title('FAS')
xlabel('Frequency(Hz)'); ylabel('Fourier Amplitude(cm/s)');
subplot(222)
plot(t1,at/980)
title('Acceleration')
xlabel('Time(s)'); ylabel('Acceleration(g)');
subplot(223)
plot(t1,vt)
title('Velocity')
xlabel('Time(s)'); ylabel('Velocity(cm/s)');
subplot(224)
plot(t1,Dt*10)
title('Displacement')
xlabel('Time(s)'); ylabel('Displacement (mm)');

%%
% Calculation of Response Spectral Acceleration, Velocity, Displacement

% Tr is the nature time period, it can be customly defined.

Tr=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,...
    0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.30,0.35,0.40,...
    0.45,0.50,0.60,0.70,0.80,0.90,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,...
    2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0];    % natural periods

psi=0.05;                                %viscous damping value

Nr=length(Tr);
RSD(NS,Nr)=zeros();
RSV(NS,Nr)=zeros();
RSA(NS,Nr)=zeros();
for m=1:1:NS  % "parfor" rather than "for" is used for large NS value for parallel computing
    [RSD(m,:),RSV(m,:),RSA(m,:)]=CDM(at(m,:),psi,dt,Tr);
end
RSA=RSA/980;               % unit "g"
RSD=RSD*10;                % unit "mm"

MedianRSD=geomean(RSD);     % median of RSD, unit "mm"

MedianRSV=geomean(RSV);     % median of RSV, unit "cm/s"

MedianRSA=geomean(RSA);     % median of RSA, unit "g"
MedianlnRSA=log(MedianRSA);
StdlnRSA=std(log(RSA));     % standard deviation of lnRSA

% Please note here I use "geomean" function instead of "median" or "mean" function to
% get the median/mean estimates because "geomean" can obtain more smooth curves.
% The above functions can give very close estimates if NS is large enough.


%Display STEP 4
figure(5)
subplot(311)
loglog(Tr,RSD,'k')
hold on 
loglog(Tr,MedianRSD,'r')
title('Response Spectral Displacement')
xlabel('Natural period(s)'); ylabel('Displacement(mm)');
subplot(312)
loglog(Tr,RSV,'k')
hold on 
loglog(Tr,MedianRSV,'r')
title('Response Spectral Velocity')
xlabel('Natural period(s)'); ylabel('Velocity(cm/s)');
subplot(313)
loglog(Tr,RSA,'k')
hold on 
loglog(Tr,MedianRSA,'r')
title('Response Spectral Acceleration')
xlabel('Natural period(s)'); ylabel('Acceleration(g)');

% This is the end of the main program for generating synthetic
% accelerograms

%%
% % The following script is for constructing Conditional Mean Spectrum for ground
% % motion selection and scaling
% % If CMS is required, NS>=2.
% 
% Tc=1.0;     % The conditioning natural period
% RSAc=0.127; % The conditioning RSA, may be retrived from Code Spectrum/ Uniform Hazard Spectrum
% nTr=find(Tr==Tc);  % Find the number in Tr 
% epsilon=(log(RSAc)-MedianlnRSA(nTr))/StdlnRSA(nTr);
% rollc(Nr)=zeros(); % correlation of epsilon at different natural periods
% for i=1:1:Nr
%     rollc(i)=1-cos(pi/2-0.359*log(max(Tc,Tr(i))/min(Tc,Tr(i))));
% end
% CMS=exp(MedianlnRSA+rollc.*StdlnRSA(nTr)*epsilon); % Conditional Mean Spectrum
% figure (6)
% loglog(Tr,MedianRSA,Tr,CMS)

%%
% The following are the functions used in main program

% Fourier Amplitude Spectrum (Target spectrum)

function [ Ax ] = AB95(f,M,R,roll,beta,sigma,Vs30,fm)
%AB95 double corner frequency model (Atkinson & Boore, 1995) 

% the applicable range of this model is: 4.0<=M<=7.25, 10<=R<=500km,
% 0.5<=f<=20Hz

% geometric attenuation function (Trilinear form)

if R<=70
    G=R^(-1);
else
    if R<=130
        G=70^(-1);
    else
        G=70^(-1)*(R/130)^(-0.5);
    end
end


% Source model

fa=10^(2.41-0.533*M);
et=10^(2.52-0.637*M);
M0=10^(1.5*M+16.05);   % (Hanks & Kanamori, 1979; Boore et al., 2014)
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));
fb=((fc^2-(1-et)*fa^2)/et)^(0.5);  % (Boore et al., 2014)


sa=1./(1+((f/fa).^2));
sb=1./(1+((f/fb).^2));
S=C*M0*(sa*(1-et)+sb*et);
S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
end

% Anelastic whole path attenuation 

Q0=680;
n=0.36;

Q=Q0*(f.^(n));
Q=Q.*beta;
Q=Q.^(-1);
An=exp(-pi*f.*Q.*R);

% amplification based on BJ Shear-Wave velocity profiling model for broad use
% (Boore & Joyner, 1997) (Boore, 2016)

%vv=amfBJ(f,beta,roll,Vs30);  

% amplification based on Tang Shear-Wave velocity profiling model for site-specific use
% (Tang, 2019)

% Zs=0.05;
% Zc= 4.0;
% Vszs=1.07;
% Vszc=3.32;
% Vs003=0.94;
% Vs02=0;
% Vs2=0;
% Vs8=3.52;
% vv=amfT(f,beta,roll,Zs,Zc,Vs003,Vszs,Vszc,Vs02,Vs2,Vs8);
vv=amfBJ(f,beta, roll, Vs30);

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s

end

function [ Ax ] = SGD02( f,M,R,roll,beta,sigma,Vs30,fm)
%SGD02  (Silva, et al. 2002) 

% the applicable range of this model is: 4.5<=M<=8.5, 1<=R<=400km,
% 0.1<=f<=100Hz

% geometric attenuation function (Trilinear form)

if R<=80
    G=R^(-(1.0296+(-0.0422)*(M-6.5)));
else
   G=(80^((-0.5)*(1.0296+(-0.0422)*(M-6.5))))*(R^((-0.5)*(1.0296+(-0.0422)*(M-6.5))));
end


% Source model

fa=10^(2.41-0.533*M);
et=10^(2.52-0.637*M);
M0=10^(1.5*M+16.05);   % (Hanks & Kanamori, 1979; Boore et al., 2014)
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));
fb=((fc^2-(1-et)*fa^2)/et)^(0.5);  % (Boore et al., 2014)


sa=1./(1+((f/fa).^2));
sb=1./(1+((f/fb).^2));
S=C*M0*(sa*(1-et)+sb*et);
S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
end
%kappa = 0.0106 * M - 0.012 ; % unit : seconds
%P=exp(-pi*f.*kappa);

% Anelastic whole path attenuation 

Q0=351;
n=0.84;

Q=Q0*(f.^(n));
Q=Q.*beta;
Q=Q.^(-1);
An=exp(-pi*f.*Q.*R);

vv=amfBJ(f,beta,roll,Vs30);

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s
end

function [ Ax ] = A04(f,M,R,roll,beta,sigma,Vs30,fm)
%A04 BCA10a (Atkinson, 2004) 

% the applicable range of this model is: 4.4<=M<=6.8, 10<=R<=800km,
% 0.05<=f<=20Hz

% geometric attenuation function (Trilinear form)

if R<=70
    G=R^(-1.3);
else
    if R<=140
        G=(70^(-0.2)/70^(1.3))*(R^0.2);
    else
        G=(70^(-0.2)/70^(1.3))*(140^(0.5)/140^(-0.2))*(R^(-0.5));
    end
end
%G=R^(-1);  %BCA10d, for CAM developing

% Source model

fa=10^(2.41-0.533*M);
et=10^(2.52-0.637*M);
M0=10^(1.5*M+16.05);   % (Hanks & Kanamori, 1979; Boore et al., 2014)
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));
fb=((fc^2-(1-et)*fa^2)/et)^(0.5);  % (Boore et al., 2014)
%fb=10^(1.43-0.188*M);

sa=1./(1+((f/fa).^2));
sb=1./(1+((f/fb).^2));
S=C*M0*(sa*(1-et)+sb*et);
S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
 %  R1=R;
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
 %  R1=(R-4);
end
%kappa = 0.0106 * M - 0.012 ; % unit : seconds
%P=exp(-pi*f.*kappa);

% Anelastic whole path attenuation 

Q0=893;
n=0.32;

Q1=Q0*(f.^(n));
Q2=1000;
nq=length(Q1);
for i=1:1:nq
    Q=max(Q1(i),Q2);
end

Q=Q.*beta;
Q=Q.^(-1);
An=exp(-pi*f.*Q.*R);

vv=amfBJ(f,beta,roll,Vs30);

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s
end

function [ Ax ] = BCA10d( f,M,R,roll,beta,sigma,Vs30,fm)
%Boore, Campbell, and Atkinson (2010)
% the applicable range of this model is: 4.4<=M<=6.8, 10<=R<=800km,
% 0.05<=f<=20Hz
%   Geometric attenuation function (linear form)

hFF=10^(-0.405+0.235*M);   
Rps=(R^2+hFF^2).^0.5;

G=Rps^(-1);

% Source model

M0=10^(1.5*M+16.05);   % (Hanks & Kanamori, 1979; Boore et al., 2014)
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));   % sigma ranges from 180~250 bars
S=(C*M0)./(1+(f/fc).^2);
S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
end

% Anelastic whole path attenuation 

Q=2850;
Q=Q.*beta;
Q=Q.^(-1);
An=exp(-pi*f.*Q*Rps);

vv=amfBJ(f,beta,roll,Vs30);

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s


end

function [ Ax ] = BS11( f,M,R,roll,beta,sigma,Vs30,fm)
% Boatwright & Seekins (2011)
% the applicable range of this model is: 4.4<=M<=5.0, 23<=R<=602km,
% 0.2<=f<=20Hz

  % geometric attenuation function (Trilinear form)

if R<=50
    G=R^(-1);
else   
    G=50^(0.5)*(R/50)^(-0.5);
end

% Source model

M0=10^(1.5*M+16.05);   % (Hanks & Kanamori, 1979; Boore et al., 2014)
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));
S=(C*M0)./(1+(f/fc).^2);
S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
end
%kappa = 0.0106 * M - 0.012 ; % unit : seconds
%P=exp(-pi*f.*kappa);

% Anelastic whole path attenuation 

Q0=410;
n=0.5;

Q=Q0*(f.^(n));   % suggestive Q0=410, n=0.5, beta=3.5km/s
Q=Q.*beta;
Q=Q.^(-1);
An=exp(-pi*f.*Q.*R);

vv=amfBJ(f,beta,roll,Vs30);

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s

end

function [ Ax ] = AB14(NT,df,f,M,R,roll,beta,sigma,Vs30,fm)
%AB14 ENA seismological model (Atkinson & Boore, 2014)

%the applicable range of this model is: 3.5<=M<=6.0, 10<=R<=500km,
% 0.2<=f<=20Hz

% *% Be careful in all models, R is hypocenter distance, but in this model, R
% should be Rps Rps=(R^2+hFF^2)^(0.5)*


hFF=10^(-0.405+0.235*M);   % (Yenier & Atkinson, 2015a)
Rps=(R^2+hFF^2).^0.5;

%   Geometric attenuation function (Complex form)

h=10;      %focal depth(km)
Tc=zeros(1,NT/2);
nTc=round(1/df);
for i=1:1:nTc
    Tc(i)=1;
end
for i=nTc:1:5*nTc
    Tc(i)=1-1.429*log10(f(i));
end
for i=5*nTc:1:NT/2
    Tc(i)=0;
end

if Rps<=h
    Clf=0.2*cos((0.5*pi)*(Rps-h)/(1-h));
else
    if Rps<=50
        Clf=0.2*cos((0.5*pi)*(Rps-h)/(50-h));
    else
        Clf=0;
    end
end

if Rps<=50
    
    G=(10.^(Tc.*Clf))*Rps^(-1.3);
else
    G=50^(-1.3)*(Rps/50)^(-0.5);
end

% Source model

M0=10^(1.5*M+16.05);   % (Hanks & Kanamori, 1979; Boore et al., 2014)
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));
S=(C*M0)./(1+(f/fc).^2);
S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
end
%kappa = 0.0106 * M - 0.012 ; % unit : seconds
%P=exp(-pi*f.*kappa);

% Anelastic whole path attenuation

Q0=525;
n=0.45;

Q=Q0*(f.^(n));   % suggestive Q0=525, n=0.45, beta=3.7km/s
Q=Q.*beta;
Q=Q.^(-1);
An=exp(-pi*f.*Q.*R);

vv=amfBJ(f,beta,roll,Vs30);

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s

end

% Upper-crust amplification of Boore & Joyner's model

function [vv]=amfBJ(f,beta, roll, Vs30)

H=[0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,...
0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,...
0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,...
0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.104,...
0.126,0.147,0.175,0.202,0.23,0.257,0.289,0.321,0.353,0.385,0.42,0.455,...
0.49,0.525,0.562,0.599,0.637,0.674,0.712,0.751,0.789,0.827,0.866,0.906,...
0.945,0.984,1.02,1.06,1.1,1.14,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.5,...
4,4.5,5,5.5,6,6.5,7,7.5,8,10,15,20,30,50];

% generic very hard rock
V1=[2.768,2.7688,2.7696,2.7704,2.7712,2.772,2.7728,2.7736,2.7744,2.7752...
2.776,2.7776,2.7792,2.7808,2.7824,2.784,2.7856,2.7872,2.7888,2.7904,...
2.792,2.7936,2.7952,2.7968,2.7984,2.8,2.8016,2.8032,2.8048,2.8064,2.808,...
2.80956,2.81112,2.81268,2.81424,2.8158,2.81736,2.81892,2.82048,2.82204,...
2.8236,2.82516,2.82672,2.82828,2.82984,2.8314,2.83296,2.85004,2.86676,...
2.88272,2.9035,2.92344,2.9436,2.9629,2.9853,3.00686,3.06098,3.0821,3.0718,...
3.0941,3.1158,3.1365,3.15796,3.17942,3.20072,3.22048,3.24024,3.260835504,...
3.271637476,3.281964539,3.29211285,3.302087707,3.311425176,3.320409798,...
3.32841313,3.337002316,3.345294257,3.353309507,3.364853485,3.399786096,...
3.430339103,3.457516593,3.482010086,3.50431659,3.510651329,3.516529187,...
3.521980007,3.527062193,3.538443827,3.54833273,3.557078298,3.564919739,...
3.572028075,3.578529853,3.584521359,3.590077571,3.595258021,3.600110775,...
3.616939825,3.647720809,3.669718999,3.700949146,3.740673099];


%generic rock
V2=[0.245,0.245,0.406898105,0.454341586,0.491321564,0.522065927,0.548608647,...
0.572100299,0.593261253,0.612575285,0.630384474,0.662434293,0.690800008,...
0.716351448,0.739672767,0.761177017,0.781168059,0.799876552,0.817482113,...
0.834127602,0.84992869,0.872659052,0.894459078,0.915511227,0.935880677,...
0.955623835,0.974789894,0.993422052,1.011558478,1.02923309,1.046476183,...
1.063314944,1.079773879,1.095875162,1.111638937,1.127083562,1.142225823,...
1.157081117,1.171663602,1.185986333,1.200061379,1.21389992,1.22751234,...
1.240908299,1.254096801,1.26708626,1.279884545,1.409876676,1.524401504,...
1.623105361,1.742468929,1.822101865,1.869784562,1.91154454,1.956709713,...
1.998031044,2.036174042,2.071640608,2.10782397,2.141667263,2.173485515,...
2.203532354,2.23359926,2.262120148,2.289978882,2.315853324,2.341268645,...
2.366247007,2.389604645,2.412077883,2.434298272,2.456270778,2.4769581,...
2.496972474,2.514890987,2.534215818,2.55296508,2.571175941,2.597555276,...
2.678472608,2.750601065,2.815833418,2.875495547,2.930554778,2.981739972,...
3.029614889,3.074625172,3.117129605,3.214232374,3.300788279,3.33118689,...
3.361507951,3.389174372,3.414630616,3.438216903,3.460199618,3.48079135,...
3.500164545,3.567982556,3.694592725,3.787139494,3.921526466,4.097643229];

S1=1./V1;
S2=1./V2;
beta=(1/Vs30-1/0.618)/(1/2.780-1/0.618);
Nh=length(H);
S(Nh)=zeros();
for i=1:1:Nh
    S(i)=(beta)*S1(i)+(1-beta)*S2(i);
end
V=1./S;

thick(Nh-1)=zeros();   %thickness of the layer
wtt(Nh-1)=zeros();     %wave travelling time
acct(Nh-1)=zeros();    %accumulated time
period(Nh-1)=zeros();  %period
avev(Nh-1)=zeros();    %average velocity
fn(Nh-1)=zeros();    %accumulated time

for m=2:1:Nh
    thick(m)=H(m)-H(m-1);
    wtt(m)=thick(m)/((V(m)+V(m-1))/2);
    acct(m)=sum(wtt(2:m));
    period(m)=4*acct(m);
    avev(m)=(H(m)-H(1))/acct(m);
    fn(m)=1/period(m);
end


Density(Nh)=zeros();
Vp(Nh)=zeros();
for i=1:1:Nh
    if V(i)<0.3
        Density(i)=1+(1.53*V(i)^0.85)/(0.35+1.889*V(i)^1.7);
    else
        if V(i)<3.55
            Vp(i)=0.9409+V(i)*2.0947-0.8206*V(i)^2+0.2683*V(i)^3-0.0251*V(i)^4;
            Density(i)=1.74*Vp(i)^0.25;
        else
            Vp(i)=0.9409+V(i)*2.0947-0.8206*V(i)^2+0.2683*V(i)^3-0.0251*V(i)^4;
            Density(i)=1.6612*Vp(i)-0.4721*Vp(i)^2+0.0671*Vp(i)^3-0.0043*Vp(i)^4+0.000106*Vp(i)^5;
        end
    end
end

fn=fn(2:Nh);
avev=avev(2:Nh);   
Density=Density(2:Nh);
            
amp=sqrt((beta*roll)./(avev.*Density.*1));

fn=fliplr(fn);
amp=fliplr(amp);

vn=log(amp);   %using the log amplification-linear frequency interpolation 
vv=interp1(fn,vn,f,'linear','extrap');
vv=exp(vv);
end

% Upper-crust amplificationg of Tang's model

function [vv]=amfT(f,beta,roll,Zs,Zc,Vs003,Vszs,Vszc,Vs02,Vs2,Vs8)

H=[0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,...
0.014,0.016,0.018,0.02,0.022,0.024,0.026,0.028,0.03,0.032,0.034,0.036,...
0.038,0.04,0.042,0.044,0.046,0.048,0.05,0.052,0.054,0.056,0.058,0.06,...
0.062,0.064,0.066,0.068,0.07,0.072,0.074,0.076,0.078,0.08,0.082,0.104,...
0.126,0.147,0.175,0.202,0.23,0.257,0.289,0.321,0.353,0.385,0.42,0.455,...
0.49,0.525,0.562,0.599,0.637,0.674,0.712,0.751,0.789,0.827,0.866,0.906,...
0.945,0.984,1.02,1.06,1.1,1.14,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.5,...
4,4.5,5,5.5,6,6.5,7,7.5,8,10,15,20,30,50];

Nh=length(H);

n=log10(Vszc/Vszs)/log10(Zc/Zs);

if Zs>0.2
    if Zs>=2
        V=Case1(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8);
    else
        if Zc>=2
            V=Case2(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8);
        else
            V=Case3(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8);
        end
    end
else
    if Zc>=2
        V=Case4(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8);
    else
        if Zc<=0.2
            V=Case6(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8);
        else
            V=Case5(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8);
        end
    end
end


thick(Nh-1)=zeros();   %thickness of the layer
wtt(Nh-1)=zeros();     %wave travelling time
acct(Nh-1)=zeros();    %accumulated time
period(Nh-1)=zeros();  %period
avev(Nh-1)=zeros();    %average velocity
fn(Nh-1)=zeros();      %accumulated time

for m=2:1:Nh
    thick(m)=H(m)-H(m-1);
    wtt(m)=thick(m)/((V(m)+V(m-1))/2);
    acct(m)=sum(wtt(2:m));
    period(m)=4*acct(m);
    avev(m)=(H(m)-H(1))/acct(m);
    fn(m)=1/period(m);
end


Density(Nh)=zeros();
Vp(Nh)=zeros();
for i=1:1:Nh
    if V(i)<0.3
        Density(i)=1+(1.53*V(i)^0.85)/(0.35+1.889*V(i)^1.7);
    else
        if V(i)<3.55
            Vp(i)=0.9409+V(i)*2.0947-0.8206*V(i)^2+0.2683*V(i)^3-0.0251*V(i)^4;
            Density(i)=1.74*Vp(i)^0.25;
        else
            Vp(i)=0.9409+V(i)*2.0947-0.8206*V(i)^2+0.2683*V(i)^3-0.0251*V(i)^4;
            Density(i)=1.6612*Vp(i)-0.4721*Vp(i)^2+0.0671*Vp(i)^3-0.0043*Vp(i)^4+0.000106*Vp(i)^5;
        end
    end
end

fn=fn(2:Nh);
avev=avev(2:Nh);   
Density=Density(2:Nh);
            
amp=sqrt((beta*roll)./(avev.*Density.*1));

fn=fliplr(fn);
amp=fliplr(amp);

vn=log(amp);   %using the log amplification-linear frequency interpolation 
vv=interp1(fn,vn,f,'linear','extrap');
vv=exp(vv);


% Input values for example regions

% Melbourne Region

% Zs=0.05 km;
% Zc= 4.0 km;
% Vszs=1.07 km/s;
% Vszc=3.32 km/s;
% Vs003=0.94 km/s;
% Vs8=3.52 km/s;

% HongKong Region

% Zs=0.001 km;
% Zc= 1.0 km;
% Vszs=0.55 km/s;
% Vszc=2.1 km/s;
% Vs2=2.6 km/s;
% Vs8=3.48 km/s;

% Northern Switzerland 

% Zs=0.001 km;
% Zc= 0.62 km;
% Vszs=1.1 km/s;
% Vszc=2.1 km/s;
% Vs2=2.7 km/s;
% Vs8=3.55 km/s;

% The following are functions of SWV profiling model for 6 cases

function Vs=Case1(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8)
Vs(Nh)=zeros();
for i=1:1:Nh
    if H(i)<=0.2
        Vs=Vs003*(H(i)/0.03)^0.3297;
    else
        if H(i)<=2
            Vs=Vs02*(H(i)/0.2)^0.1732;
        else
           if H(i)<=Zs
               Vs=Vs2*(H(i)/2)^0.1667;
           else
              if H(i)<=Zc
                  Vs=Vszc*(H(i)/zc)^n;
              else
                  Vs=Vs8*(H(i)/8)^0.0833;
              end
           end
        end
    end
end

end

function Vs=Case2(H,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8)
Vs(Nh)=zeros();
for i=1:1:Nh
    if H(i)<=0.2
        Vs=Vs003*(H(i)/0.03)^0.3297;
    else
        if H(i)<=Zs
            Vs=Vs02*(H(i)/0.2)^0.1732;
        else
           if H(i)<=Zc
               Vs=Vszc*(H(i)/Zc)^n;
           else
               Vs=Vs8*(H(i)/8)^0.0833;
           end
        end
    end
end

end

function Vs=Case3(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8)
Vs(Nh)=zeros();
for i=1:1:Nh
    if H(i)<=0.2
        Vs(i)=Vs003*(H(i)/0.03)^0.3297;
    else
        if H(i)<=Zs
            Vs(i)=Vs02*(H(i)/0.2)^0.1732;
        else
           if H(i)<=Zc
               Vs(i)=Vszc*(H(i)/Zc)^n;
           else
               if H(i)<=2
                   Vs(i)=Vs2*(H(i)/2)^0.0899;
               else
                   Vs(i)=Vs8*(H(i)/8)^0.0833;
               end
           end
        end
    end
end
end

function Vs=Case4(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8)

  VszI=min(Vs003,Vszs);
  ZI=min(0.03,Zs);
  Vs(Nh)=zeros();
for i=1:1:Nh
    if H(i)<=Zs
        Vs(i)=VszI*(H(i)/ZI)^0.3297;
    else
        if H(i)<=Zc
            Vs(i)=Vszc*(H(i)/Zc)^n;
        else
            Vs(i)=Vs8*(H(i)/8)^0.0833;
        end
    end
end

end

function Vs=Case5(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8)

VszI=min(Vs003,Vszs);
ZI=min(0.03,Zs);
Vs(Nh)=zeros();
for i=1:1:Nh
    if H(i)<=Zs
        Vs(i)=VszI*(H(i)/ZI)^0.3297;
    else
        if H(i)<=Zc
            Vs(i)=Vszc*(H(i)/Zc)^n;
        else
            if H(i)<=2
                Vs(i)=Vs2*(H(i)/2)^0.0899;
            else
                Vs(i)=Vs8*(H(i)/8)^0.0833;
            end
        end
    end
end

end

function Vs=Case6(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8)

VszI=min(Vs003,Vszs);
ZI=min(0.03,Zs);
Vs(Nh)=zeros();
for i=1:1:Nh
    if H(i)<=Zs
        Vs(i)=VszI*(H(i)/ZI)^0.3297;
    else
        if H(i)<=Zc
            Vs(i)=Vszc*(H(i)/Zc)^n;
        else
            if H(i)<=0.2
                Vs(i)=Vs02*(H(i)/0.2)^0.2463;
            else
                if H(i)<=2
                    Vs(i)=Vs2*(H(i)/2)^0.0899;
                else
                    Vs(i)=Vs8*(H(i)/8)^0.0833;
                end
            end
        end
    end
end

end


end

% Central Difference Method for computing response spectra

function [ RSD,RSV,RSA ] = CDM( at,psi,dt,Tr)
% The following is the implementation of time-step integration method
% Central Difference Method
fr=1./Tr;
omega=fr.*(2*pi);
A=-omega.^2+2/(dt^2);
B=psi*omega./dt-1/(dt^2);
C=1/(dt^2)+psi*omega./dt;
Nr=length(Tr);
NT=length(at);
u(Nr,NT)=zeros();
u(Nr,1)=-at(2)*dt^2/2;
for i=1:1:Nr
    for j=2:1:NT
        u(i,j+1)=(-(at(j))+A(i)*u(i,j)+B(i)*u(i,j-1))/C(i);
    end
end
RSD=max(abs(u'));
RSV=omega.*RSD;
RSA=omega.*RSV;


end



%References
%Atkinson, G.M., & Boore, D.M. (1995). Ground-Motion Relations for Eastern North America. Bull. Seism. Soc. Am., 85, 17-30. 
%Atkinson, G.M., & Boore, D.M. (2014). The attenuation of Fourier amplitudes for rock sites in eastern North America. Bull. Seism. Soc. Am, 104, 513-528. 
%Baker, J.W. 2011. Conditional Mean Spectrum: Tool for Ground-Motion Selection, Journal of Structural Engineering, Vol 137(3) 322-331.
%Baker, J.W. and Jayaram, N., (2008) "Correlation of spectral acceleration values from NGA ground motion models," Earthquake Spectra, 24(1), 299-317.
%Boore, D.M., Alessandro, C.D., & Abrahamson, N.A. (2014). A Generalization of the Double-Corner-Frequency Source Spectral Model and Its Use in the SCEC BBP Validation Exercise. Bull. Seismol. Soc. Am., 104(5), 2387-2398. 
%Brune, J. N. (1970). Tectonic stress and the spectra of seismic shear waves from earthquakes. J. Geophys .Res, 75, 4997-5009. 
%Boore, D.M.& Joyner,W.B.(1997). Site Amplifications for Generic Rock Sites. Bull. Seismol. Soc. Am., 87(2), 327-341.
%Boore, D.M.(2016). Short Note: Determining Generic Velocity and Density Models for Crustal Amplification Calculations, with an Update of the Boore and Joyner (1997) Generic Amplification for Vs(Z)=760 m/s. Bull. Seismol. Soc. Am., 106(1), 316-320. 
%Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. J. Geophys .Res, 84(B5), 2348-2350.
%Lam, N. T. K., Wilson, J., & Hutchinson, G. (2000). Generation of Synthetic Earthquake Accelerogram Using Seismological Modelling: A Review. Journal of Earthquake Engineering, 93(6), 2531-2545. 
%Tang,Y.(2019). Seismic Hazard Analysis and Management for Low-to-moderateSeismicity Regions Based on Ground Motion Simulation. Ph.D. Thesis, TheUniversity of Melbourne, Australia.



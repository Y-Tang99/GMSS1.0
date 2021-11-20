function varargout = GMSSPro(varargin)


% Edit the above text to modify the response to help GMSSPro

% Last Modified by GUIDE v2.5 16-Jun-2020 20:12:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GMSSPro_OpeningFcn, ...
                   'gui_OutputFcn',  @GMSSPro_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GMSSPro is made visible.
function GMSSPro_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for GMSSPro
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

initialize_gui(hObject, handles, false);

% --- Outputs from this function are returned to the command line.
function varargout = GMSSPro_OutputFcn(hObject, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;

function Snumber_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Snumber_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function Timestep_Callback(hObject, eventdata, handles)

Timeinterval = str2double(get(hObject, 'String'));
if isnan(Timeinterval)
    set(hObject, 'String',0);
    errordlg('Input must be a number','Error');
end

% --- Executes during object creation, after setting all properties.
function Timestep_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vdamping_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function Vdamping_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Sourcedensity_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Sourcedensity_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function SourceVs_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function SourceVs_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function Stressdrop_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Stressdrop_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Mw_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Mw_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Distance_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Distance_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vs30_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vs30_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function fm_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function fm_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Zs_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Zs_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Zc_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Zc_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vszs_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vszs_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vszc_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vszc_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vs003_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vs003_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vs02_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vs02_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vs2_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vs2_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Vs8_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function Vs8_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function PGA_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function PGA_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PGV_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function PGV_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function PGD_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function PGD_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function Tc1_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function Tc1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function RSAc1_Callback(hObject, eventdata, handles)



% --- Executes during object creation, after setting all properties.
function RSAc1_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

%THE FOLLOWING IS THE MAIN FUNCTION

% --- Executes on button press in RUN.
function RUN_Callback(~, ~, handles)
% hObject    handle to RUN (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

NS=str2double(get(handles.Snumber, 'String'));  
dt=str2double(get(handles.Timestep, 'String'));  
roll=str2double(get(handles.Sourcedensity, 'String'));  
beta=str2double(get(handles.SourceVs, 'String'));  
sigma=str2double(get(handles.Stressdrop, 'String'));  
M=str2double(get(handles.Mw, 'String'));  
R=str2double(get(handles.Distance, 'String'));  
Vs30=str2double(get(handles.Vs30, 'String'));  
fm=str2double(get(handles.fm, 'String'));  
psi=str2double(get(handles.Vdamping, 'String'));
Zs=str2double(get(handles.Zs, 'String'));  
Zc=str2double(get(handles.Zc, 'String')); 
Vszs=str2double(get(handles.Vszs, 'String')); 
Vszc=str2double(get(handles.Vszc, 'String'));  
Vs003=str2double(get(handles.Vs003, 'String'));  
Vs02=str2double(get(handles.Vs02, 'String'));  
Vs2=str2double(get(handles.Vs2, 'String'));  
Vs8=str2double(get(handles.Vs8, 'String'));  
Tc=str2double(get(handles.Tc1, 'String'));  
RSAc=str2double(get(handles.RSAc1, 'String'));  

%Main procedures


%STEP1----Generation of band-limit Gaussian white noise

M0=10^(1.5*M+16.05);   % check page 117 of SMSIM source code
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));   % corner frequency (Brune,1970)
fa=10^(2.41-0.533*M);
et=10^(2.52-0.637*M);
fb=((fc^2-(1-et)*fa^2)/et)^(0.5);  

%Source duration
if get (handles.SCF,'Value')
    tds=1/fc;
else
    tds=1/(2*fa)+1/(2*fb);
end

%Path duration
popup_sel_index = get(handles.FAS, 'Value');
switch popup_sel_index
    case 1
        if R<=70
            tdp=0.16*R;
        else
            if R<=130
                tdp=-0.03*(R-70)+11.2;
            else
                tdp=0.04*(R-130)+9.4;
            end
        end  
    case 2
        tdp=0.05*R;
    case 3
        tdp=0.05*R;
    case 4
        tdp=0.05*R;
    case 5
        tdp=0.05*R;
    case 6
        tdp=0.05*R;
    case 7
        % --- Executes on button press in AttenuFun.
        run(handles.filename) 
end

%Total duration
td=tds+tdp;    

nseed=round(log2(td/dt))+1;

NT=2^nseed;       % Total number of time step
T=NT*dt;          % Total simulation duration

t=dt:dt:T;
df=1/T;
f=(1:1:NT/2)*df;

fl=1/T;                      % low frequency limit
fh=1/(2*dt);  % high frequency limit--Nyquist Frequency
fbw=fh-fl;                   % frequency band
fs=2*fh;                     % sampling frequency

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

%STEP2----% window the white noise to shape a resembled accelerogram


wt=exp((-0.4)*t.*(6/td))-exp((-1.2)*t.*(6/td));   % exponential window function
wts=wt.^2;
wtssum=sum(wts.*dt);
w=sqrt(2*(fbw)/wtssum);     % window scaling factor (Lam, 2000)
win=(wt.*w);

st(NS,NT)=zeros();          % windowed band-limited white noise
for m=1:1:NS
    st(m,:)=nt(m,:).*win;
end
%Forward Fourier Transform

As(NS,NT)=zeros();
anglek(NS,NT/2)=zeros();
for m=1:1:NS
    As(m,:)=fft(st(m,:));
    anglek(m,:)=angle(As(m,1:NT/2)); % only oomplex number has phase angle!
end

Asf=abs(As);   % amplitude value
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


% STEP3----filter function based on seismological model

Ax=FAS(f,M,R,roll,beta,sigma,fm,NT,df,Vs30,Zs,Zc,Vs003,Vszs,Vszc,Vs02,Vs2,Vs8,handles);


% STEP4----obtain the synthetic accelerogram

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
at0=real(at00);  % we just need the real component-->amplitude

% Basedline Correction

windowWidth=NT-1;    % frame size, must be odd
polynomialOrder=6;   % Smooth with a Savitzky-Golay sliding polynomial filter
baseline(NS,NT)=zeros();
at1(NS,NT)=zeros();
for m=1:1:NS
    baseline(m,:) = sgolayfilt(at0(m,:), polynomialOrder, windowWidth);
    at1(m,:)=at0(m,:)-baseline(m,:);
end

% add a pre-event pad 
tpad=20;                     % time pad is 20s
padsize=[0,tpad/dt];  
padval=0; 
N=NT+2*tpad/dt;                  
T1=T+2*tpad; 
t1=0:dt:T1-dt;

% A filter is applied to make sure long period noise is removed
filterts(N)=zeros();
for i=1:1:(tpad+T)/dt
    filterts(i)=1;
end

direction1='pre';
direction2='post';
at2(NS,NT+tpad/dt)=zeros();
at(NS,N)=zeros();
for m=1:1:NS
    at2(m,:) = padarray(at1(m,:),padsize,padval,direction1); 
    at(m,:) = padarray(at2(m,:),padsize,padval,direction2);
    at(m,:)=at(m,:).*filterts;         % filtered acceleration time series
end


pga=max(abs(at'));

vt(NS,N)=zeros();
for m=1:1:NS
    vt(m,:)=cumtrapz(t1,at(m,:));    % velocity time series
    vt(m,:)=vt(m,:).*filterts;       % filtered velocity time series
end

pgv=max(abs(vt'));

Dt(NS,N)=zeros();
for m=1:1:NS
    Dt(m,:)=cumtrapz(t1,vt(m,:));    % displacement time series
    Dt(m,:)=Dt(m,:).*filterts;       % filtered displacement time series
end

pgd=max(abs(Dt'));


% calculation of PGA, PGV and PGD
PGA=geomean(pga/980);      % unit "g"
PGA=roundn(PGA,-2);        
PGV=geomean(pgv);          % unit "cm/s"
PGV=roundn(PGV,-1);
PGD=geomean(pgd*10);       % unit "mm"
PGD=roundn(PGD,-1);

% Please note here I use "geomean" function instead of "median" or "mean" function to
% get the median/mean estimates because "geomean" can obtain more smooth curves.
% The above functions can give very close estimates if NS is large enough.


% Calculation of Response Spectra of Acceleration, Velocity, Displacement

% Tr is the nature time period, it can be customer defined.

Tr=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,...
    0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.30,0.35,0.40,...
    0.45,0.50,0.60,0.70,0.80,0.90,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,...
    2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,4.5,5.0,6.0,7.0,8.0,9.0,10.0];    % natural periods

Nr=length(Tr);
RSD(NS,Nr)=zeros();
RSV(NS,Nr)=zeros();
RSA(NS,Nr)=zeros();
for m=1:1:NS
    [RSD(m,:),RSV(m,:),RSA(m,:)]=CDM(at(m,:),psi,dt,Tr,handles);
end

RSA=RSA/980;               % unit "g"    

MedianRSA=geomean(RSA);    % median of RSA, unit "g"
MedianlnRSA=log(MedianRSA);
StdlnRSA=std(log(RSA));    % standard deviation of lnRSA

% Please note here I use "geomean" function instead of "median" or "mean" function to
% get the median/mean estimates because "geomean" can obtain more smooth curves.
% The above functions can give very close estimates if NS is large enough.


% The following script is for constructing Conditional Mean Spectrum for
% ground motion selection and scaling.


rollc(Nr)=zeros(); % correlation of standard deviation at different natural periods
for i=1:1:Nr
    rollc(i)=1-cos(pi/2-0.359*log(max(Tc,Tr(i))/min(Tc,Tr(i))));
end

if get(handles.CMS1,'Value')
    nTr=find(Tr==Tc);  % Find the number in Tr 
    epsilon=(log(RSAc)-MedianlnRSA(nTr))/StdlnRSA(nTr);
    CMS=exp(MedianlnRSA+rollc.*StdlnRSA(nTr)*epsilon);
else
    CMS=0;
end


%THE FOLLOWING IS TO DISPLAY THE RESULTS

set(handles.PGA,'string',PGA);
set(handles.PGV,'string',PGV);
set(handles.PGD,'string',PGD);

axes(handles.at);  % display acceleration time history
cla;
plot(t1,at(1,:)/980);
grid on
box on
xlabel('Time (s)')
ylabel('Accelrtation (g)')
legend('Acceleration','Location','Northeast','Orientation','horizontal')


axes(handles.vt);  % display velocity time history
cla;
plot(t1,vt(1,:));
grid on
box on
xlabel('Time (s)')
ylabel('Velocity (cm/s)')
legend('Velocity','Location','Northeast','Orientation','horizontal')

axes(handles.FASfig);  % display Fourier amplitude spectrum
cla;
loglog(f,Ax);
grid on
xlabel('Frenquency(Hz)')
ylabel('Fourier Amplitude(cm/s)')
legend('FAS','Location','southwest','Orientation','horizontal')

axes(handles.RSAfig);  % display response spectral acceleration
cla;
if get(handles.CMS1,'Value')
    loglog(Tr,MedianRSA,'b');
    grid on
    hold on
    loglog(Tr,CMS,'-.r');
    xlabel('Tn(s)')
    ylabel('Spectral Acceleration(g)')
    legend('Median RSA','CMS','Location','southwest','Orientation','horizontal')
else
    loglog(Tr,RSA(1,:),'k');
    grid on
    hold on
    loglog(Tr,MedianRSA,'b');
    xlabel('Tn(s)')
    ylabel('RSA(g)')
    legend('RSA','Median RSA','Location','southwest','Orientation','horizontal')
end

%save the data 
setappdata(0,'NT',NT);
setappdata(0,'T',T);
setappdata(0,'F',f);
setappdata(0,'FAS1',Ax);
setappdata(0,'Time1',t1);
setappdata(0,'Acc1',at);
setappdata(0,'Vel1',vt);
setappdata(0,'Dis1',Dt);
setappdata(0,'Tn1',Tr);
setappdata(0,'RSA1',RSA);
setappdata(0,'PGA1',PGA);
setappdata(0,'PGV1',PGV);
setappdata(0,'PGD1',PGD);
setappdata(0,'MedianRSA1',MedianRSA);
setappdata(0,'CMS1',CMS);



% --- Executes on button press in SAVE.
function SAVE_Callback(hObject, eventdata, handles)
% hObject    handle to SAVE (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
NS=str2double(get(handles.Snumber, 'String'));
dt=str2double(get(handles.Timestep, 'String'));  
roll=str2double(get(handles.Sourcedensity, 'String'));  
beta=str2double(get(handles.SourceVs, 'String'));  
sigma=str2double(get(handles.Stressdrop, 'String'));  
M=str2double(get(handles.Mw, 'String'));  
R=str2double(get(handles.Distance, 'String'));  
Vs30=str2double(get(handles.Vs30, 'String'));  
fm=str2double(get(handles.fm, 'String'));  
psi=str2double(get(handles.Vdamping, 'String'));
Zs=str2double(get(handles.Zs, 'String'));  
Zc=str2double(get(handles.Zc, 'String')); 
Vszs=str2double(get(handles.Vszs, 'String')); 
Vszc=str2double(get(handles.Vszc, 'String'));  
Vs003=str2double(get(handles.Vs003, 'String'));  
Vs02=str2double(get(handles.Vs02, 'String'));  
Vs2=str2double(get(handles.Vs2, 'String'));  
Vs8=str2double(get(handles.Vs8, 'String')); 
Tc=str2double(get(handles.Tc1, 'String'));  
RSAc=str2double(get(handles.RSAc1, 'String')); 


PGA=getappdata(0,'PGA1');
PGV=getappdata(0,'PGV1');
PGD=getappdata(0,'PGD1');
fxls=getappdata(0,'F'); fxls=fxls';
FASxls=getappdata(0,'FAS1');FASxls=FASxls';
Time=getappdata(0,'Time1'); Time=Time';
Acc=getappdata(0,'Acc1');Acc=(Acc/980)';   % unit "g"
Vel=getappdata(0,'Vel1');Vel=Vel';         % unit "cm/s"
Dis=getappdata(0,'Dis1');Dis=(Dis*10)';    % unit "mm"
Tn=getappdata(0,'Tn1');Tn=Tn';
RSAxls=getappdata(0,'RSA1');RSAxls=RSAxls';
MedianRSAxls=getappdata(0,'MedianRSA1');MedianRSAxls=MedianRSAxls';
CMSxls=getappdata(0,'CMS1');CMSxls=CMSxls';

memo1={NS;dt;roll;beta;sigma;M;R;Vs30;fm;psi;Zs;Zc;Vszs;Vszc;Vs003;Vs02;Vs2;Vs8;Tc;RSAc};
memo2={PGA;PGV;PGD};

[filename,pathname]=uiputfile('*.xls','Save Results');
if filename==0
    return;
end
[pathstr,filename]=fileparts(filename);
%Inputparameters
xlswrite(filename,{'Simulation No.'},'Sheet1','A1');
xlswrite(filename,{'Time Step(s)'},'Sheet1','A2');
xlswrite(filename,{'Source Density(g/cm^3)'},'Sheet1','A3');
xlswrite(filename,{'Source SWV(km/s)'},'Sheet1','A4');
xlswrite(filename,{'Stress Drop(bar)'},'Sheet1','A5');
xlswrite(filename,{'Magnitude'},'Sheet1','A6');
xlswrite(filename,{'Distance(km)'},'Sheet1','A7');
xlswrite(filename,{'Vs30(km/s)'},'Sheet1','A8');
xlswrite(filename,{'fm(s)'},'Sheet1','A9');
xlswrite(filename,{'Damping Value'},'Sheet1','A10');
xlswrite(filename,{'Zs(km)'},'Sheet1','A11');
xlswrite(filename,{'Zc(km)'},'Sheet1','A12');
xlswrite(filename,{'Vszs(km/s)'},'Sheet1','A13');
xlswrite(filename,{'Vszc(km/s)'},'Sheet1','A14');
xlswrite(filename,{'Vs003(km/s)'},'Sheet1','A15');
xlswrite(filename,{'Vs02(km/s)'},'Sheet1','A16');
xlswrite(filename,{'Vs2(km/s)'},'Sheet1','A17');
xlswrite(filename,{'Vs8(km/s)'},'Sheet1','A18');
xlswrite(filename,{'Tc(s)'},'Sheet1','A19');
xlswrite(filename,{'RSAc(g)'},'Sheet1','A20');


xlswrite(filename,memo1,'Sheet1','B1:B20');

%Peak Ground Motions
xlswrite(filename,{'PGA(g)'},'FAS','A1');
xlswrite(filename,{'PGV(cm/s)'},'FAS','A2');
xlswrite(filename,{'PGD(mm)'},'FAS','A3');
xlswrite(filename,memo2,'FAS','B1:B3');

%Fourier Amplitude Spectrum
xlswrite(filename,{'Frequency(Hz)'},'FAS','D1');
xlswrite(filename,{'Fourier Amplitude(cm/s)'},'FAS','E1');
xlswrite(filename,fxls,'FAS','D2:D20000');
xlswrite(filename,FASxls,'FAS','E2:E20000');

%Time series
xlswrite(filename,{'Time(s)'},'At','A1');
xlswrite(filename,{'Acceleration(g)'},'At','B1');
xlswrite(filename,Time,'At','A2:A50000');
xlswrite(filename,Acc,'At','B2:CW50000');  %maximum 100 time series

xlswrite(filename,{'Time(s)'},'Vt','A1');
xlswrite(filename,{'Velocity(cm/s)'},'Vt','B1');
xlswrite(filename,Time,'Vt','A2:A50000');
xlswrite(filename,Vel,'Vt','B2:CW50000');  %maximum 100 time series

xlswrite(filename,{'Time(s)'},'Dt','A1');
xlswrite(filename,{'Displacement(mm)'},'Dt','B1');
xlswrite(filename,Time,'Dt','A2:A50000');
xlswrite(filename,Dis,'Dt','B2:CW50000');  %maximum 100 time series


% Response spectrum 
xlswrite(filename,{'Natural Periods(s)'},'RSA','A1');
xlswrite(filename,{'RSA(g)'},'RSA','B1');
xlswrite(filename,Tn,'RSA','A2:A150');
xlswrite(filename,RSAxls,'RSA','B2:CW150');

% Median RSAfig & Conditional Mean Spectrum 
xlswrite(filename,{'Natural Periods(s)'},'MedianRSA&CMS','A1');
xlswrite(filename,{'MedianRSA(g)'},'MedianRSA&CMS','B1');
xlswrite(filename,{'CMS(g)'},'MedianRSA&CMS','C1');
xlswrite(filename,Tn,'MedianRSA&CMS','A2:A150');
xlswrite(filename,MedianRSAxls,'MedianRSA&CMS','B2:B150');
xlswrite(filename,CMSxls,'MedianRSA&CMS','C2:C150');





% --- Executes on button press in RESET.
function RESET_Callback(hObject, eventdata, handles)
% hObject    handle to RESET (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

initialize_gui(gcbf, handles, true);


%THE FOLLOWING ARE THE FUNCTIONS ADPOTED IN ABOVE PROGRAMS, PLEASE PUT THE
%EXTRA FUNCTIONS IN THIS PLACE


function [Ax]=FAS(f,M,R,roll,beta,sigma,fm,NT,df,Vs30,Zs,Zc,Vs003,Vszs,Vszc,Vs02,Vs2,Vs8,handles)


fa=10^(2.41-0.533*M);
et=10^(2.52-0.637*M);
M0=10^(1.5*M+16.05);   
C=(0.55*2.0*0.707)/((4*pi*(roll*1000)*(beta*1000)^3)*1000);  % be careful of the unit of C
fc=4.906*(10^6)*beta*((sigma/M0)^(1/3));
fb=((fc^2-(1-et)*fa^2)/et)^(0.5);  

if get (handles.SCF,'Value')
    sc=1./(1+((f/fc).^2));
    S=C*M0*sc;
else
    sa=1./(1+((f/fa).^2));
    sb=1./(1+((f/fb).^2));
    S=C*M0*(sa*(1-et)+sb*et);
end


S=((2*f.*pi).^2).*S;
S=S./10^7;               % convert unit for S into m/s.

% Upper-crust attenuation model

if fm>1
   P=(1+(f/fm).^8).^(-0.5);
else
   kappa=fm;
   P=exp(-pi*f.*kappa);
end

popup_sel_index = get(handles.FAS, 'Value');
switch popup_sel_index
    case 1
        Q=680*f.^0.36;
        Q=Q.*3.8;
        Q=Q.^(-1);
        if R<=70
            G=R^(-1);
        else
            if R<=130
                G=70^(-1);
            else
                G=70^(-1)*(R/130)^(-0.5);
            end
        end
    case 2
        Q=351*f.^0.84;
        Q=Q.*3.52;
        Q=Q.^(-1);
        if R<=80
            G=R^(-(1.0296+(-0.0422)*(M-6.5)));
        else
            G=(80^((-0.5)*(1.0296+(-0.0422)*(M-6.5))))*(R^((-0.5)*(1.0296+(-0.0422)*(M-6.5))));
        end
    case 3
        Q=max(1000,893*f.^0.32);
        Q=Q.*3.7;
        Q=Q.^(-1);
        if R<=70
            G=R^(-1.3);
        else
            if R<=140
                G=(70^(-0.2)/70^(1.3))*(R^0.2);
            else
                G=(70^(-0.2)/70^(1.3))*(140^(0.5)/140^(-0.2))*(R^(-0.5));
            end
        end
    case 4
        Q=2850;
        Q=Q.*3.7;
        Q=Q.^(-1);
        hFF=10^(-0.405+0.235*M);   %finite-fault effect factor (YA15)
        Rps=sqrt(R^2+hFF^2);
        G=Rps^(-1);
    case 5
        Q=410*f.^0.5;
        Q=Q.*3.5;
        Q=Q.^(-1);
        if R<=50
            G=R^(-1);
        else
            G=(50^(0.5)/50)*(R)^(-0.5);
        end
    case 6
        Q=525*f.^0.45;
        Q=Q.*3.7;
        Q=Q.^(-1);
      hFF=10^(-0.405+0.235*M);   %finite-fault effect factor (YA15)
      Rps=sqrt(R^2+hFF^2);
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
    case 7
        % --- Executes on button press in AttenuFun.
        run(handles.filename)
        
end

An=exp(-pi*f.*Q.*R);   % whole path anelastic attenuation

%upper-crustal amplification
if get(handles.AmfBJ1,'Value')
    vv=amfBJ(f,beta, roll, Vs30,handles);
else
    if get(handles.AmfT1,'Value')
        vv=amfT(f,beta,roll,Zs,Zc,Vs003,Vszs,Vszc,Vs02,Vs2,Vs8,handles);
    else
        vv=1;
    end
end

Ax=S.*An.*P.*vv.*G*100;        % convert into cm/s

function FAS_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function FAS_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
set(hObject, 'String', {'AB95', 'SGD02', 'A04', 'BCA10d', 'BS11','AB14','Custom'});


function [ RSD,RSV,RSA ] = CDM( at,psi,dt,Tr,handles)
% The following is the implementation of time-step integration method
% Central Difference Method
dt=0.002;
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
    RSD=max(abs(u'));
    RSV=omega.*RSD;
    RSA=omega.*RSV; 
end


% --- Executes when selected object is changed in uibuttongroup3.
function uibuttongroup3_SelectionChangedFcn(hObject, eventdata, handles)

% Shear-wave velocity profiling model for general broad use
function [vv]=amfBJ(f,beta, roll, Vs30,handles)
% calculating frequency-dependent upper-crustal amplification factor

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

S1=1./V1;    % slowness of hard rock
S2=1./V2;
betavs=(1/Vs30-1/0.618)/(1/2.780-1/0.618);% should be careful here "betavs" is not "beta"
Nh=length(H);
S(Nh)=zeros();
for i=1:1:Nh
    S(i)=(betavs)*S1(i)+(1-betavs)*S2(i);
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

% Shear-wave velocity profiling model for site-specific use
function [vv]=amfT(f,beta,roll,Zs,Zc,Vs003,Vszs,Vszc,Vs02,Vs2,Vs8,handles)

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
        V=Case1(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles);
    else
        if Zc>=2
            V=Case2(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles);
        else
            V=Case3(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles);
        end
    end
else
    if Zc>=2
        V=Case4(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles);
    else
        if Zc<=0.2
            V=Case6(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles);
        else
            V=Case5(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles);
        end
    end
end

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

% The following are functions of SWV profiling model for 6 cases
function Vs=Case1(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles)
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

function Vs=Case2(H,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles)
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

function Vs=Case3(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles)
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

function Vs=Case4(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles)

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

function Vs=Case5(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles)

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

function Vs=Case6(H,Nh,Zs,Zc,Vs003,Vs02,Vs2,Vszs,Vszc,n,Vs8,handles)

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


% --- Executes when selected object is changed in uibuttongroup4.
function uibuttongroup4_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes on button press in SCF.
function SCF_Callback(hObject, eventdata, handles)

% --- Executes on button press in DCF.
function DCF_Callback(hObject, eventdata, handles)


% % --- Executes on button press in AmfBJ1.
function AmfBJ1_Callback(hObject, eventdata, handles)

% --- Executes on button press in AmfT1.
function AmfT1_Callback(hObject, eventdata, handles)

% --- Executes on button press in Amf0.
function Amf0_Callback(hObject, eventdata, handles)


% --- Executes when selected object is changed in uibuttongroup5.
function uibuttongroup5_SelectionChangedFcn(hObject, eventdata, handles)

% --- Executes on button press in CMS1.
function CMS1_Callback(hObject, eventdata, handles)

% --- Executes on button press in CMS0.
function CMS0_Callback(hObject, eventdata, handles)


function FunPath_Callback(hObject, eventdata, handles)

% --- Executes during object creation, after setting all properties.
function FunPath_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function AttenuFun_Callback(hObject, eventdata, handles)

[FileName,PathName] = uigetfile('*.m','Select the attenuation function file');
fullpathname=strcat(PathName,FileName);
set(handles.FunPath,'String',FileName)
handles.filename=fullfile(fullpathname);
guidata(hObject,handles)


function initialize_gui(~, handles, isreset)

if isfield(handles, 'metricdata') && ~isreset
    return;
end

handles.metricdata.Snumber = 100;
handles.metricdata.Timestep  = 0.002;
handles.metricdata.Sourcedensity = 2.8;
handles.metricdata.SourceVs  = 3.5;
handles.metricdata.Stressdrop  = 200;
handles.metricdata.Mw  = 6;
handles.metricdata.Distance  = 30;
handles.metricdata.Vs30  = 0.76;
handles.metricdata.fm  = 0.025;
handles.metricdata.Vdamping  = 0.05;
handles.metricdata.Zs  = 0.05;
handles.metricdata.Zc = 4;
handles.metricdata.Vszs  = 1.33;
handles.metricdata.Vszc  = 3.32;
handles.metricdata.Vs003 = 1.12;
handles.metricdata.Vs02  = 1.78;
handles.metricdata.Vs2  = 2.87;
handles.metricdata.Vs8  = 3.5;
handles.metricdata.PGA  = 0.0;
handles.metricdata.PGV  = 0.0;
handles.metricdata.PGD  = 0.0;
handles.metricdata.FunPath  = 0;
handles.metricdata.Tc1  = 1.0;
handles.metricdata.RSAc1  = 0.127;


set(handles.Snumber, 'String', handles.metricdata.Snumber);
set(handles.Timestep,  'String', handles.metricdata.Timestep);
set(handles.Sourcedensity, 'String', handles.metricdata.Sourcedensity);
set(handles.SourceVs,  'String', handles.metricdata.SourceVs);
set(handles.Stressdrop,  'String', handles.metricdata.Stressdrop);
set(handles.Mw,  'String', handles.metricdata.Mw);
set(handles.Distance,  'String', handles.metricdata.Distance);
set(handles.Vs30,  'String', handles.metricdata.Vs30);
set(handles.fm,  'String', handles.metricdata.fm);
set(handles.Vdamping,  'String', handles.metricdata.Vdamping);
set(handles.Zs,  'String', handles.metricdata.Zs);
set(handles.Zc,  'String', handles.metricdata.Zc);
set(handles.Vszs,  'String', handles.metricdata.Vszs);
set(handles.Vszc,  'String', handles.metricdata.Vszc);
set(handles.Vs003,  'String', handles.metricdata.Vs003);
set(handles.Vs02,  'String', handles.metricdata.Vs02);
set(handles.Vs2,  'String', handles.metricdata.Vs2);
set(handles.Vs8,  'String', handles.metricdata.Vs8);
set(handles.PGA,  'String', handles.metricdata.PGA);
set(handles.PGV,  'String', handles.metricdata.PGV);
set(handles.PGD,  'String', handles.metricdata.PGD);
set(handles.FunPath,  'String', handles.metricdata.FunPath);
set(handles.Tc1,  'String', handles.metricdata.Tc1);
set(handles.RSAc1,  'String', handles.metricdata.RSAc1);


cla(handles.at);
cla(handles.vt);
cla(handles.FASfig);
cla(handles.RSAfig);
% Update handles structure
guidata(handles.figure1, handles);


% --------------------------------------------------------------------
%This is the end.

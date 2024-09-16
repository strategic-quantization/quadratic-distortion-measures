function main
clc;
close all;
clear all;

% parameters
Mval=[4]; % number of levels of quantization
rhoval=[ -0.9 -0.5 0 0.5 0.9]; % correlation

% X distribution
a=-5;
b=5;
mux=0; % mean of source X
sigma_xsq=1; % variance of source X

% theta distribution
at=-5;
bt=5;
mut=0;
sigma_thsq=1;

% discretizing theta (mentioned in Numerical Results; done for
% computational feasibility)
thval1=linspace(at,mut-2*sigma_thsq,1);
thval2=linspace(mut-2*sigma_thsq,mut-sigma_thsq,5);
thval3=linspace(mut-sigma_thsq,mut+sigma_thsq,11);
thval4=linspace(mut+sigma_thsq,mut+2*sigma_thsq,5);
thval5=linspace(mut+2*sigma_thsq,bt,1);
thval=[thval1(2:end) thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)];
thval=[thval1 thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)]'; % \mathcal{T}
nt=length(thval);
% pdf of theta
pth=zeros(1,length(thval));
f12=@(tv) ((1/sqrt(2*pi*sigma_thsq))*exp(-(tv-mut).^2/(2*sigma_thsq)));
sct=integral(f12,at,bt,'ArrayValued',true);
pth(1)=integral(f12,at,thval(1)+(thval(2)-thval(1))/2,'ArrayValued',true)/sct;
for i=2:length(thval)-1
    pth(i)=integral(f12,thval(i)-(thval(i)-thval(i-1))/2,thval(i)+(thval(i+1)-thval(i))/2,'ArrayValued',true)/sct;
end
pth(length(thval))=integral(f12,thval(end)-(thval(end)-thval(end-1))/2,bt,'ArrayValued',true)/sct;

% loading the initializations used for our results
rn=10; % # of initializations
load('data_gaussian.mat');

for M=Mval
for rho=rhoval
% (X,theta) distribution
mux_corr=mux+rho*(sigma_xsq/sigma_thsq)^(1/2)*(thval(:)-mut); % mean of X conditional on theta 
sigma_xsq_corr=(1-rho^2)*sigma_xsq; % variance of X conditional on theta 
f1=@(xv,i) ((1/sqrt(2*pi*sigma_xsq_corr))*exp(-(xv-mux_corr(i)).^2/(2*sigma_xsq_corr)))*pth(i); % pdf of X conditional on theta

xrandinit=reshape(xinitializations(find(M==Mv),find(rho==rhov),:,1:M+1,:),[length(thval),M+1,rn]); % initializations
rn=size(xrandinit,3); % number of initializations
xrm=zeros(length(thval),M+1,rn); % final quantizer values for all initializations
erm=zeros(1,rn); % encoder distortions for all initializations
yrm=zeros(M,rn); % final quantizer representative values for all initializations
drm=zeros(1,rn); % decoder distortions for all initializations
exitflag=zeros(1,rn); % indicating how the algorithm stopped - because of local optima or because the algorithm cannot go further due to constraint violations (quantizer boundaries are increasing in m: x_{theta,m}<x_{theta,m+1})
derend=zeros(length(thval),M-1,rn);

for r=1:rn
flag=1; % indicator of when the gradient descent should stop
xmiter=zeros(length(thval),M+1,100); % quantizer values for each iteration given an initial point
endist=zeros(1,100); % encoder distortions for each iteration given an initial point
frendist=zeros(1,100); % fractional difference in encoder distortions for each iteration given an initial point
dedist=zeros(1,100); % decoder distortions for each iteration given an initial point
derv=zeros(length(thval),M-1,100);
iter=1; % count of iterations 
xm=xrandinit(:,:,r); % initialization
ym=reconstruction(xm,f1,thval);
dist_enc=encoderdistortion(xm,ym,f1,thval);
dist_dec=decoderdistortion(xm,ym,f1,thval);
endist(1)=dist_enc;
dedist(1)=dist_dec;
delta=1; % learning rate: x_{theta',m}\triangleq x_{theta',m} - delta * der
tic
while flag 
for t=1:length(thval)
for i=2:M
der=derivative(xm,ym,f1,i,t,thval); % computing partial D_E/ partial x_{theta',m} in page 4. Here theta' = thval(t), m = i.
derv(t,i-1,iter)=der; % storing derivatives for all iterations
% computing quantizer update
temp=xm(t,i)-delta*der; 
xm1=xm;
xm1(t,i)=temp;
% computing associated reconstruction points and encoder distortion
ym=reconstruction(xm1,f1,thval);
d1=encoderdistortion(xm1,ym,f1,thval);
% checking whether it satisfies constraints and the encoder distortion is decreasing
if (temp>xm(t,i-1) && temp<xm(t,i+1)) && d1<dist_enc 
    xm(t,i)=temp;
else
    [xm]=check(xm,f1,delta,der,dist_enc,i,t,thval); % if either condition is not satisfied, move along the gradient in smaller steps until the constraints are satisfied
end
% computing associated reconstruction points and encoder distortion of the current quantizer iteration
ym=reconstruction(xm,f1,thval);
dist_enc=encoderdistortion(xm,ym,f1,thval);
end
end
% quantizer after gradient descent over x_{theta,m} for all theta \in \mathcal{T} and m \in [1:M-1]
xmtemp=xm;
% computing associated reconstruction points and encoder distortion
ymtemp=reconstruction(xmtemp,f1,thval);
dist_enctemp=encoderdistortion(xmtemp,ymtemp,f1,thval);
% checking for condition to stop gradient descent: encoder distortion remains unchanging
if iter>1
if (endist(iter) == endist(iter-1))
    flag=0; % stop the while loop
    exitflag(r)=2; % indicating that gradient descent stopped because encoder distortion is no longer changing which also be because any further changes causes constraint violation
end
end

if all(abs(derv(:,:,iter)) <10^-7 ) 
    flag=0; % stop the while loop
    exitflag(r)=1; % indicating that the gradient descent loop stopped because derivatives are close to 0, not because of constraint violations 
else 
    iter=iter+1; 
    xm=xmtemp;
    ym=ymtemp;
    xmiter(:,:,iter)=xm; % storing quantizer for this iteration
    dist_enc=dist_enctemp; % storing encoder distortion of this iteration to compare for next iteration
    endist(iter)=dist_enc; % storing encoder distortion for this iteration
    dedist(iter)=decoderdistortion(xm,ym,f1,thval); % storing decoder distortion for this iteration
end
end
toc
% storing values for a given initialization at the end of gradient descent
derend(:,:,r)=derv(:,:,iter); % derivative values partial D_E/ partial x_{theta',m} for theta' \in \mathcal{T}, m \in [1:M]
xrm(:,:,r)=xm; % storing quantizer boundaries
erm(r)=dist_enc; % storing encoder distortion
yrm(:,r)=reconstruction(xm,f1,thval); % storing quantizer reconstruction values
drm(r)=decoderdistortion(xm,yrm(:,r),f1,thval); % storing decoder distortion
exitf=exitflag(r); % storing exit flag value 
% display output
disp(strcat('M = ',num2str(M),', r = ',num2str(r),', rho = ',num2str(rho))) % number of levels of quantization, initialization number, correlation 
exitf % exit flag
xm % quantizer
ym % reconstruction levels
dist_enc % encoder distortion
end

disp(strcat('Results: M = ',num2str(M),', rho = ',num2str(rho)))
[in1 in2]=min(erm); % finding minimum encoder distortion amongst the initializations
xm=xrm(:,:,in2) % corresponding quantizer 
ym=reconstruction(xm,f1,thval) % reconstruction values
dist_enc=encoderdistortion(xm,ym,f1,thval) % encoder distortion
dist_dec=decoderdistortion(xm,ym,f1,thval) % decoder distortion

% save to mat file
save(strcat('xthetaM',num2str(M),'rho',num2str(rho),'varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'noiseless_GD_gaussian.mat'),'xm','ym','dist_enc','dist_dec','erm','xrm','yrm','drm','derend','xrandinit')

end
end

function [xm]=check(xm,f1,delta,der,dist_enc,i,t,thval) 
% called if quantizer after gradient descent violates constraints
while delta>10^-7 % learning rate
    delta=delta/10; % keep decreasing the learning rate until the quantizer update no longer violates constraints
    % computing quantizer update
    temp=xm(t,i)-delta*der;
    xm1=xm;
    xm1(t,i)=temp;
    % computing associated reconstruction points and encoder distortion
    ym=reconstruction(xm1,f1,thval);
    d1=encoderdistortion(xm1,ym,f1,thval);
    % checking whether it satisfies constraints and the encoder distortion is decreasing
    if (temp>xm1(t,i-1) && temp<xm1(t,i+1) ) && d1<dist_enc
        xm(t,i)=temp; % return updated quantizer
        break;
    end
end

function [dist_dec]=decoderdistortion(xthetam,ym,f1,thval) 
% compute decoder distortion
M=size(xthetam,2)-1; % number of levels of quantization
% D_D = \mathbb{E}\{(x-y)^2\}: in Section 3: Problem Formulation
dist_dec=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv-ym(i))^2*f1temp(xv);
        dist_dec=dist_dec+integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
    end
end

function [dist_enc]=encoderdistortion(xthetam,ym,f1,thval)
% compute encoder distortion
M=size(xthetam,2)-1; % number of levels of quantization
% D_E = \mathbb{E}\{(x+theta-y)^2\}: in Section 3 Problem Formulation
dist_enc=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv+thval(k)-ym(i))^2*f1temp(xv);
        dist_enc=dist_enc+integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
    end
end


function [ym]=reconstruction(xthetam,f1,thval)
% compute reconstruction values
M=size(xthetam,2)-1;
% y_i^* = (integral_{\mathcal{T}} integral_{\mathcal{V}_{\theta,m}} x d \mu_{X,\theta})/(integral_{\mathcal{T}} integral_{\mathcal{V}_{\theta,m}} d \mu_{X,\theta})
% in Section 4.1 Main Results, Analysis
ym=zeros(1,M);
for i=1:M
    num=0;
    den=0;
    for j=1:length(thval) % integral_{\mathcal{T}}
        f1temp= @(xv) f1(xv,j);
        f2=@(xv) xv*f1temp(xv);
        % integral_{\mathcal{V}_{\theta,m}}
        num=num+integral(f2,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
        den=den+integral(f1temp,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
    end
    if den~=0
        ym(i)=num/den; % y_i^*
    else
        ym(i)=(1/size(xthetam,1))*sum(xthetam(:,i)); % if den = 0: no region is mapped to quantization level m
    end
end

function [der]=derivative(xm,ym,f1,i,t,thval)
% compute derivative partial D_E/ partial x_{theta',m}
% in Section 4.1 Main Results, Analysis
M=size(xm,2)-1;

der=0;

den1=0;
den2=0;
num=f1(xm(t,i),t);
for th=1:size(thval)
    den1=den1+integral(@(xv) f1(xv,th),xm(th,i-1),xm(th,i));
    den2=den2+integral(@(xv) f1(xv,th),xm(th,i),xm(th,i+1));
end
der=(xm(t,i)+thval(t)-ym(i-1))^2*f1(xm(t,i),t);
der=der-(xm(t,i)+thval(t)-ym(i))^2*f1(xm(t,i),t);
dyixi=(xm(t,i)-ym(i-1))*num/den1;
dyi1xi=-(xm(t,i)-ym(i))*num/den2;
for th=1:length(thval)
    f3_1=@(xv) (xv+thval(th)-ym(i-1))*f1(xv,th);
    f3_2=@(xv) (xv+thval(th)-ym(i))*f1(xv,th);
    if xm(th,i-1)~=xm(th,i)        
        der=der-2*dyixi*integral(f3_1,xm(th,i-1),xm(th,i),'ArrayValued',true);
    end
    if xm(th,i)~=xm(th,i+1)        
        der=der-2*dyi1xi*integral(f3_2,xm(th,i),xm(th,i+1),'ArrayValued',true);
    end
end   

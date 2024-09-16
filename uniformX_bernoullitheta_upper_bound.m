function main
clc
close all
clear all

% parameters
Mval=[2 4 8 16]; % number of levels of quantization

% independent X, theta

% X distribution 
% Uniform [a,b]
% mean mx, var sx
a=0;
b=1;
sx=(b-a)^2/12; % variance of source X
mx=(a+b)/2; % mean of source X
fx1=@(xv) 1/(b-a); % pdf of X

% theta distribution 
% Bernoulli
thval=[-1;1];
pth=[0.5 0.5]; % pmf of theta
st=pth*thval.^2; % variance of theta
mt=pth*thval; % mean of theta

% joint probability distribution of (X,theta)
ftx=@(tv,xv) pth(tv).*fx1(xv);

% computing parameters, eqn 5, 6 in paper
r=st/sx;
A=sqrt(1+4*r);
alpha= (A-1)/(2*r);
kappa=1/(1+alpha^2*r);

% estimation distortion E{(X+theta-hatX)^2} in Theorem 2
d_est=sx*(1+kappa^2-2*kappa)+st*(1+kappa^2*alpha^2-2*kappa*alpha);

% distribution of kappa(X+alpha theta)
ta1=kappa*alpha*thval(1)-kappa*(b-a)/2;
tb1=kappa*alpha*thval(1)+kappa*(b-a)/2;
ta2=kappa*alpha*thval(2)-kappa*(b-a)/2;
tb2=kappa*alpha*thval(2)+kappa*(b-a)/2;
fv=(1/kappa)*1/(b-a);
if tb1<ta2
    fx=@(x) 0.5*fv*trapmf(x,[ta1,ta1,tb1,tb1])+0.5*fv*trapmf(x,[ta2,ta2,tb2,tb2]);
else
    fx=@(x) 0.5*fv*trapmf(x,[ta1,ta1,ta2,ta2])+fv*trapmf(x,[ta2,ta2,tb1,tb1])+0.5*fv*trapmf(x,[tb1,tb1,tb2,tb2]);
end
% range of kappa(X+alpha theta)=[lim1,lim4]
lim1=ta1;
lim4=tb2;

rn=5; % initializations for lloyd max to find classical MSE quantizer of kappa(X+alpha theta) 
for M=Mval
distrn=zeros(rn,1);
xmrn=zeros(M+1,rn);
for r=1:rn
% lloyd max
xmiter=zeros(M+1,1000);
ym=(lim4-lim1)*rand(M,1)+lim1; % initialize reconstruction levels
dist=zeros(1,1000);
frac=1;
iter=1;
while frac>0
    % iterate between minimizing encoder and decoder distortions (both MSE)
    % until convergence
    xm=lloyd_enc(ym,lim1,lim4);
    ym=lloyd_dec(xm,fx);
    if iter>1
        d1=dist_decquant(xm,ym,fx);
        frac=(d-d1)/d;
    end
    d=dist_decquant(xm,ym,fx);
    dist(iter)=d;
    xmiter(:,iter)=xm;
    iter=iter+1;
end
xmrn(:,r)=xm;
distrn(r)=d;
end

% finding minimum of all initializations
[v,ind]=min(distrn);
xm=xmrn(:,ind)'; % Q**
[ym]=reconstruction(xm,fx); % y(Q**)
d_q=encdist(xm,ym,fx); % distortion for Q** quantizer decision and y(Q**) representative levels

d1=thirdterm(xm,ym,ftx,kappa,alpha); % 2E{theta(Xhat-Q**(Xhat))} in Theorem 2

num_quant_levels=M
enc_dist_upperbound=d_q+d_est+d1

save(strcat('discrete_uniformbernoulli_theta_M',num2str(M),'data.mat'),'alpha','ftx','thval','M','xm','ym','enc_dist_upperbound');
end

function [f221]=thirdterm(x,ym,ftx,kappa,alpha)
% 2E{theta(Xhat-Q**(Xhat))} in Theorem 2
M=length(x)-1;
f221=0;
for tv=1:2
for i=1:M
    f221=f221+integral(@(xv) (kappa*(xv+alpha*tv)-ym(i)).*tv.*ftx(tv,xv),x(i),x(i+1)); 
end
end
f221=2*f221;


function [f22]=encdist(x,ym,f12)
M=length(x)-1;
f22=0;
for i=1:M
    f22=f22+integral(@(xv) (xv-ym(i)).^2.*f12(xv),x(i),x(i+1)); 
end

function [ym]=reconstruction(x,f12)
M=length(x)-1;
ym=zeros(1,M);
for i=1:M
    f2=@(xv) xv.*f12(xv);
    num=integral(f2,x(i),x(i+1));
    den=integral(f12,x(i),x(i+1));
    ym(i)=num/den;
end


function d=dist_decquant(xm,ym,fx) % MSE distortion for lloyd max
M=length(ym);
d=0;
for i=1:M
    d=d+integral(@(xv) (xv-ym(i)).^2.*fx(xv),xm(i),xm(i+1));  
end


function xm=lloyd_enc(ym,a,b)
% lloyd max computing quantizer boundaries
M=length(ym);
T1=[a; ym(1:M-1); b];
T2=[a; ym(2:M); b];
xm=(T1+T2)/2;
xm=xm';

function ym=lloyd_dec(xm,fx) 
% lloyd max computing reconstruction levels
M=length(xm)-1;
ym=zeros(M,1);
for i=1:M
    fx1= @(xv) xv.*fx(xv);
        num=integral(fx1,xm(i),xm(i+1),'ArrayValued',true);
        den=integral(@(xv) fx(xv),xm(i),xm(i+1),'ArrayValued',true);
    ym(i)=num/den;
end

% noiseless
% encoder observes (X,theta)
% encoder quantizes h(g(X,theta))=kappa(X+alpha theta) as Q(h(g(X,theta)))
% and sends to the decoder
% decoder estimates X from Q(h(g(X,theta))) 
% quantization is done in non-strategic sense with MSE measure

function main
clc;
close all;
clear all;

Mval=[2 4 8 16 32];

% X distribution parameters
% finite range [a,b]
% mean mx, var sx
a=-5;
b=5;
sx=1;
mx=0;

% theta distribution parameters
% finite range [at,bt]
% mean mt, var st
at=-5;
bt=5;
st=1;
mt=0;

r=st/sx; % r=var(theta)/var(X)

rho=-0.9/sx; % correlation

% computing parameters, eqn 5, 6 in paper
A=sqrt(1+4*(r+rho));
alpha= (A-1)/(2*(r+rho));
kappa=(1+alpha*rho)/(1+alpha^2*r+2*alpha*rho);

% quantizer values for Gaussian from [Max 1960]
[xmall,ymall,dmall,Mv_max]=max_quant();

% distribution of kappa(X+alpha theta)
sigma_xsq=kappa^2*sx+kappa^2*alpha^2*st+2*kappa^2*alpha*rho*sqrt(sx)*sqrt(st);
mux=kappa*mx+kappa*alpha*mt;
a1=kappa*(a+alpha*at);
b1=kappa*(b+alpha*bt);

% estimation distortion E{(X+theta-hatX)^2} in Theorem 2
d_est=sx*(1+kappa^2-2*kappa)+st*(1+kappa^2*alpha^2-2*kappa*alpha)+ rho*(2+2*kappa^2*alpha-2*kappa-2*kappa*alpha)

% joint distribution of Xhat and theta
muxhat_t=@(tv) kappa*(rho*tv+alpha.*tv);
sxhat_t=@(tv) kappa^2*sx*(1-rho^2);
ftxhat=@(tv,xv)  ((1/sqrt(2*pi*st))*exp(-(tv-mt).^2/(2*st))).* ((1./sqrt(2*pi*sxhat_t(tv))).*exp(-(xv-muxhat_t(tv)).^2./(2*sxhat_t(tv))));

for m=1:length(Mval)
m1=find(Mv_max==Mval(m));
% Mval(m) level non-strategic quantizer with MSE distortion for standard normal distribution
xm=xmall(m1,1:Mv_max(m1)-1); 
% quantizer for Gaussian N(mux,sigma_xsq)
xm=sqrt(sigma_xsq)*xm+mux;

x=[a1 xm b1]; % considering range limited to [a1,b1] 
ym=sqrt(sigma_xsq)*ymall(m1,1:Mv_max(m1)); % corresponding decoder actions
d_q=sigma_xsq*dmall(m1); % distortion for x quantizer decision and ym representative levels

d1=thirdterm(x,ym,ftxhat,at,bt); % 2E{theta(Xhat-Q**(Xhat))} in Theorem 2

num_quant_levels=Mval(m)
enc_dist_upperbound=d_est+d_q+d1
save(strcat('a',num2str(a),'b',num2str(b),'contthM',num2str(Mval(m)),'rho',num2str(rho),'varx',num2str(sx),'varth',num2str(st),'.mat'),'xm','enc_dist_upperbound');
end

function [f221]=thirdterm(x,ym,ftxhat,at,bt)
% 2E{theta(Xhat-Q**(Xhat))} in Theorem 2
M=length(x)-1;
f221=0;
for i=1:M
    f221=f221+integral2(@(tv,xv) (xv-ym(i)).*tv.*ftxhat(tv,xv),at,bt,x(i),x(i+1)); 
end
f221=2*f221;

% max quantization table from [Max 1960]
% zero mean, unit variance Gaussian
function [xmall,ymall,dmall,Mval]=max_quant()
clc;
close all;
clear all;
Mval=[2 4 8 16 32];
xmall=zeros(length(Mval),max(Mval)-1);
ymall=zeros(length(Mval),max(Mval));
dmall=zeros(length(Mval),1);
M=2;
xmall(find(M==Mval),1:M-1)=0;
ymall(find(M==Mval),1:M)=[-0.7980 0.7980];
dmall(find(M==Mval))=0.3634;
M=4;
temp=[0.9816];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.4528 1.510];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.1175;
M=8;
temp=[0.5006 1.050 1.748];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.2451 0.7560 1.344 2.152];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.03454;
M=16;
temp=[0.2582 0.5224 0.7996 1.099 1.437 1.844 2.401];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.1284 0.3881 0.6568 0.9424 1.256 1.618 2.069 2.733];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.009497;
M=32;
temp=[0.1320 0.2648 0.3991 0.5359 0.6761 0.8210 0.9718 1.130 1.299 1.482 1.682 1.908 2.174 2.505 2.977];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.06590 0.1981 0.3314 0.4668 0.6050 0.7473 0.8947 1.049 1.212 1.387 1.577 1.788 2.029 2.319 2.692 3.263];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
dmall(find(M==Mval))=0.002499;

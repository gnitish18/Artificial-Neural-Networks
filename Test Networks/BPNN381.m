input=[0.2 0.3 0.4 0.5;0.1 0.2 0.1 0.2];%input('input matrix is : ');
desired=[0.002 0.012 0.004 0.020];%input('desired matrix: ');
npt=4;%input('no.of patterns is: ');
epolim=5400;%input('no.of epochs is : ');
m=2;%input('no.of inputs are : ');
n=2;%input('no,of hidden units are: ');
r=1;%input('no.of output units are: ');
eta=0.5;%input('learning rate is : ');
mom_fac=1;%input('momentum factor is : ');
s=1;%input('slope for sigmiod is : ');
wj=[0.1 0.2;0.3 0.4];%rand(m,n);
bj=[0.5;-0.5];%rand(n,1);
wk=[0.5;0.7];%rand(n,r);
bk=-0.5;%rand(r,1);
for epoch=1:epolim
    for pt=1:npt
        I=input(:,pt);
        D=desired(:,pt);
        sum_hid=zeros(n,1);
        o_h=zeros(n,1);
        sumout=zeros(r,1);
        out=zeros(r,1);
        
        for p=1:n
            for h=1:m
                sum_hid(p)=sum_hid(p)+I(h)*wj(h,p);
            end
            sum_hid(p)=sum_hid(p)+bj(p);
            o_h(p)=1./(1+exp(-s*sum_hid(p)));
        end
        for q=1:r
            for p=1:n
                sumout(q)=sumout(q)+o_h(p)*wk(p,q);
            end
            sumout(q)=sumout(q)+bk(q);
            out(q)=1./(1+exp(-s*sumout(q)));  %sigmoid function 
        end
        %******************************************************************%
        delwk=zeros(n,r);
        delbk=zeros(r,1);
        for q=1:r
            for p=1:n
                f_sigk(q)= 1./(1+exp(-s*sumout(q)));
                f_sig(p)= 1./(1+exp(-s*sum_hid(p)));
                value=(D(q)-f_sigk(q))*f_sigk(q)*(1-f_sigk(q));
                delwk(p,q)=eta*2*s*value*f_sig(p)+mom_fac.*delwk(p,q);
            end
            delbk(q,1)=eta*2*s*value+mom_fac*delbk(q,1);
        end
        
        delwj=zeros(m,n);
        delbj=zeros(n,1);
        for h=1:m
            for p=1:n
                f_sig(p)= 1./(1+exp(-s*sum_hid(p)));
                dsj=0;
                dsjb=0;
                for q=1:r
                     f_sigk(q)= 1./(1+exp(-s*sumout(q)));
                    value=(D(q)-f_sigk(q))*f_sigk(q)*(1-f_sigk(q));
                    dsj=dsj+(2.*s.*value.*wk(p,q).*f_sig(p).*(1-f_sig(p)).*I(h));
                end
                dsjb=dsjb+(2.*s.*value.*wk(p,q).*f_sig(p).*(1-f_sig(p)));
                delwj(h,p)=eta*dsj+mom_fac.*delwj(h,p);
                delbj(p,1)=eta*dsjb+mom_fac.*delbj(p,1);
            end
        end
        wk=wk+delwk;
        bk=bk+delbk;
        wj=wj+delwj;
        bj=bj+delbj;   
    end
end
disp('Testing');
 In=[0.1;0.1];
 sum_hid=zeros(n,1);
 o_h=zeros(n,1);
 sumout=zeros(r,1);
 out=zeros(r,1);
    for p=1:n
            for h=1:m
                sum_hid(p)=sum_hid(p)+In(h)*wj(h,p);
            end
            sum_hid(p)=sum_hid(p)+bj(p);
            o_h(p)=1./(1+exp(-s*sum_hid(p)));
    end  
    for q=1:r
            for p=1:n
                sumout(q)=sumout(q)+o_h(p)*wk(p,q);
            end
            sumout(q)=sumout(q)+bk(q);
            out(q)=1./(1+exp(-s*sumout(q)));
    end
 disp(out);
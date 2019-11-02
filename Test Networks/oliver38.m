X=[54 67;54 62;37 84;41 94;2 99;7 64;25 62;22 60;18 54;4 50;13 40;18 40;24 42;25 38;44 35;41 26;45 21;58 35;62 32;82 7;91 38;83 46;71 44;64 60;68 58;83 69;87 76;74 78;71 71;58 69];
n=30;           %no.of cities
a=1;    
R=0.12;  
distance=zeros(n,n);

for i=1:n
    for j=1:n
        distance(i,j)=((X(i,1)-X(j,1)).^2 + (X(i,2)-X(j,2)).^2).^(0.5) ;  %distance     
    end
end
U=eye(n);             %can use ANY START.
%U=randi([0 1],n,n);
Temp=5000;   %40000
pen=3;      %998
b=1;        %250  %total distance travelled is:503.721

while Temp > 10
    while a < n^2
        I=26;   %random integers
        J=30;
                                     % Wt=wt(i,j,I,J,dis,p,b)
                                     %if(( I==I ) && (J~=J)) || (( I~=I ) && (J==J))
                                     %      Wt =-pen;
                                     % else if (( I==I ) && (J==J))
                                     %     Wt = b;
                                     %else if ((J==(J+1)) ||(J==(J-1)))
                                     %    Wt=-distance(I,I);
                                     %else Wt = 0;
         %initial consesus           %   end
                                     %  end
                                     %end
                                     %Wt=wt(P,Q,P,Q,distance,pen,b);
        Del_Con=(1-2*(U(I,J)))*b;
        for i=1:n
            for j=1:n
               if((i~=I) && (i~=J))
                           if(( i==I ) && (i~=J)) || (( i~=I ) && (i==J))
                               Wt =-p;
                           else if (( i==I ) && (i==J))
                               Wt = b;
                           else if ((i==(J+1)) ||(i==(J-1)))             %wt(i,i,P,Q,distance,pen,b)*U(i,j)
                               Wt=-distance(i,I);
                           else Wt = 0;                           %finding change in consensus.
                               end
                               end
                           end
                    Del_Con=Del_Con + (1-2*(U(I,J)))*Wt;
               end
            end
        end
        A=1/(1+exp(-(Del_Con)/Temp));  %propbability of accepting the change in consensus.
        if R < A                       %accepting the probabilty or rejecting the change based on R.
            U(I,J)=1-U(I,J);           %Low R more fliud and allowing to accept.
            if (J+1)==n
                J=1;
            else if (J+1)==0
                    J=n;
                else 
                    J=J+1;
                end
            end
                
         else
           U(I,J)=U(I,J);
        end
        a=a+1;    
   end
 Temp=0.95*Temp;
end

ks=1;
for a=1:n
    for b=1:n
        if U(b,a)==1
            m(ks)=b;
            ks=ks+1;
        end
    end 
end
%((X(i,1)-X(j,1)).^2 + (X(i,2)-X(j,2)).^2).^(0.5)
di=0;   
for a=1:n
    if a==n
        di=di+((X(m(a),1)-X(m(1),1)).^2 + (X(m(a),2)-X(m(1),2)).^2).^(0.5) ;        % d(m(a),m(1),X);
    else
        di=di+((X(m(a),1)-X(m(a+1),1)).^2 + (X(m(a),2)-X(m(a+1),2)).^2).^(0.5) ;       %d(m(a),m(a+1),X);
    end
end
display('total distance travelled is:');
disp(di);

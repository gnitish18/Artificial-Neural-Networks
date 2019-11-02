in = [1,1,1,-1;-1,-1,-1,1;1,-1,-1,-1;-1,-1,1,1];

m = 2;
mxpt = 4;
epoch = 1000;
R = 0;
n = 4;
eta = 0.9;

W = [0.2,0.8;0.6,0.4;0.5,0.7;0.9,0.3];

for iter = 1:epoch
    for pt = 1:mxpt
        disp('epoch number')
        disp(iter)
        disp('Pattern number')
        disp(pt)
        I=in(pt,:);
        display(in);
        disp(I);
        pause
    end
end
        
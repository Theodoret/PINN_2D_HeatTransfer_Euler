%{
A MATLAB program to calculate 2D Heat Transfer problem. Neural Network
is assigned to solve the Partial Differential Equations in problem.
2D Heat Transfer problem is described as "du/dt = alpha*(d2u/dx2 + d2u/dy2)".

Created by:
Theodoret Putra Agatho
University of Atma Jaya Yogyakarta
Department of Informatics
27/07/2023

Disclaimer:
This program writing is influenced heavily by following code:
Andreas Almqvist (2023). Physics-informed neural network solution of 2nd order ODE:s 
(https://www.mathworks.com/matlabcentral/fileexchange/96852-physics-informed-neural-network-solution-of-2nd-order-ode-s),
MATLAB Central File Exchange. Retrieved May 11, 2023.
%}

clear all; clc; close all;
%% Initialization

% Hyperparameters
alpha = 0.1;

% Time
dt = 0.1;
time = 1.0; % time max
epoch_time = floor(time/dt);
time_step = 2; % graph changing every mod(i,time_step)==0

% Grid 2D
Nx = 21; % N-many elements in x vector
Ny = 11; % N-many elements in x vector
% x-vector
xmin = 0; xmax = 4; % boundary x vector
xline = linspace(xmin,xmax,Nx); % Grid
x = zeros(Ny,Nx);
for i = 1:Ny
    x(i,:) = xline;
end
dx = xline(2) - xline(1);

% y-vector
ymin = 0; ymax = 2; % boundary x vector
yline = linspace(ymin,ymax,Ny); % Grid
y = zeros(Ny,Nx);
for i = 1:Nx
    y(:,i) = yline;
end
dy = yline(2) - yline(1);

% Neural Network
learning_rate = 0.001;
epoch_learning = 512;
batch = 128;
Nn = 32; % N-many neuron
% weightes and biases
min = -2; max = 2; % boundary weights and biases
w0x = min + rand(Nn,1)*(abs(min)+abs(max));
w0y = min + rand(Nn,1)*(abs(min)+abs(max));
b0 = min + rand(Nn,1)*(abs(min)+abs(max));
w1 = (min + rand(Nn,1)*(abs(min)+abs(max)));
b1 = (min + rand(1)*(abs(min)+abs(max)));
params = [w0x;w0y;b0;w1;b1];
% initial_params = params;

% Initial values
U = 100.*ones(Ny,Nx)./300; % consider "U" as current y and "y" as next y
U(1,:) = 300.*ones(1,Nx)./300;
U(end,:) = 300.*ones(1,Nx)./300;


%% Other solutions
% Analytical (/exact) Solution
L = ymax;
Uetemp = zeros(Ny,Nx);

mmax = 201;
for m = 1:mmax
    Uetemp = Uetemp + exp(-(m*pi/L).^2.*alpha.*time).*((1-(-1).^m)./(m.*pi)).*sin(m*pi*y/L);
end
Ue = (300 + 2*(100-300).*Uetemp)./300;

% Numerical solution - Finite Difference Method
Ufd = U(2:Ny-1,2:Nx-1);
sx = alpha*dt/dx^2;
sy = alpha*dt/dy^2;

Ni = (Ny-2)*(Nx-2);
A = eye(Ni,Ni);
for i = 1:Ni
    for j = 1:Ni
        if A(i,j) == 1
            A(i,j) = 1 + 2*sx + 2*sy;
            if j > 1 && mod(j,(Ny-2)) ~= 1
                A(i,j-1) = -sx;
            end
            if j < Ni && mod(j,(Ny-2)) ~= 0
                A(i,j+1) = -sx;
            end
            if j > (Ny-2)
                A(i,j-(Ny-2)) = -sy;
            end
            if j <= Ni-(Ny-2)
                A(i,j+(Ny-2)) = -sy;
            end
        end
    end
end

% Boundary
By = zeros(Ny-2,Nx-2);
Bx = zeros(Ny-2,Nx-2);
Bx(end,:) = 300.*ones(1,Nx-2)./300;
Bx(1,:) = 300.*ones(1,Nx-2)./300;
By(1,:) = 300.*ones(1,Nx-2)./300;
By(end,:) = 300.*ones(1,Nx-2)./300;
By = reshape(By,[],Ni);
Bx = reshape(Bx,[],Ni);
Ufd = reshape(Ufd,[],Ni);

t0fd = tic; % stopwatch start
for i = 1:epoch_time
    Bx = Ufd;
    Bx = reshape(Bx,[Ny-2,Nx-2]);
    Bx(:,2:end-1) = zeros(Ny-2,Nx-4);
    Bx = reshape(Bx,[],Ni);
    Ufd = (Ufd + Bx*sx + By*sy)/A;
end
tfd = toc(t0fd); % stopwatch end

Ufd = reshape(Ufd,[Ny-2,Nx-2]);
Utemp = 100.*ones(Ny,Nx)./300;
Utemp(2:Ny-1,2:Nx-1) = Ufd;
Utemp(end,:) = 300.*ones(1,Nx)./300;
Utemp(1,:) = 300.*ones(1,Nx)./300;
Utemp(:,1) = Utemp(:,2);
Utemp(:,end) = Utemp(:,end-1);
Ufd = Utemp;



%% Neural Network training
t0nn = tic; % stopwatch start
for i = 1:epoch_time
    disp(i)
    for j = 1:epoch_learning
        params = training(params,U,x,y,Nn,alpha,dt,batch,learning_rate);
    end
    [u,~,~] = predict(params,x,y,Nn);
    U = u;
    if mod(i,time_step)==0
        figure(gcf);surf(x,y,U);%ylim([0 1]);
    end
end
tnn = toc(t0nn); % stopwatch end
% U = U.*300;
surf(x,y,U);title('Neural Network');


%% Results and Visualisations
% Visualisations
figure(2);surf(x,y,Ufd);title('Finite Difference');

% Error
error_nn = mean((U-Ue).^2,'all');
error_fd = mean((Ufd-Ue).^2,'all');
figure(3);surf(x,y,U-Ue);title('Neural Network Error');
figure(4);surf(x,y,Ufd-Ue);title('Finite Difference Error');

% mean((U(end,:)-1).^2+(U(1,:)-1).^2+(U(:,1)-U(:,2)).^2+(U(:,end)-U(:,end-1)).^2,'all')



%% Functions (Neural Network)
function y = mySigmoid(x)
    y = 1./(1 + exp(-x));
end

function params = training(params,U,x,y,Nn,alpha,dt,batch,learning_rate)
    w0x = params(1:Nn);
    w0y = params(Nn+1:Nn*2);
    b0 = params(Nn*2+1:Nn*3);
    w1 = params(Nn*3+1:end-1);
    b1 = params(end);

    Ny = size(y,1);
    Nx = size(x,2);
    btop = 300.*ones(1,Nx)./300;
    bbottom = 300.*ones(1,Nx)./300;

    % Boundary weights and biases
    w0xbx = repmat(w0x,1,Ny); w0ybx = repmat(w0y,1,Ny);
    w0xby = repmat(w0x,1,Nx); w0yby = repmat(w0y,1,Nx);

    xleft = repmat(x(:,1).',Nn,1) ; xright  = repmat(x(:,end).',Nn,1);
    xnleft = repmat(x(:,2).',Nn,1) ; xnright  = repmat(x(:,end-1).',Nn,1);
    xtop  = repmat(x(1,:),Nn,1)   ; xbottom = repmat(x(end,:),Nn,1);

    yleft = repmat(y(:,1).',Nn,1) ; yright  = repmat(y(:,end).',Nn,1);
    ynleft = repmat(y(:,2).',Nn,1) ; ynright  = repmat(y(:,end-1).',Nn,1);
    ytop  = repmat(y(1,:),Nn,1)   ; ybottom = repmat(y(end,:),Nn,1);
    for k = 1:batch
%         disp(k)
        % Pick a random data point for current batch
        j = randi(Nx);
        i = randi(Ny);
        xi = x(i,j);
        yi = y(i,j);
        Ui = U(i,j);

        % Sigmoid function derivatives with w0, b0, and xi inputs
        s = mySigmoid(w0x*xi + w0y*yi + b0);
        sp = s.*(1 - s);
        spp = sp.*(1 - 2*s);
        sppp = spp.*(1 - 2*s) - 2*sp.^2;

        % at boundary
        sleft = mySigmoid(w0xbx.*xleft + w0ybx.*yleft + b0); % [Nn,Ny]
        sright = mySigmoid(w0xbx.*xright + w0ybx.*yright + b0);
        stop = mySigmoid(w0xby.*xtop + w0yby.*ytop + b0); % [Nn,Nx]
        sbottom = mySigmoid(w0xby.*xbottom + w0yby.*ybottom + b0);

        snleft = mySigmoid(w0xbx.*xnleft + w0ybx.*ynleft + b0); % [Nn,Ny]
        snright = mySigmoid(w0xbx.*xnright + w0ybx.*ynright + b0);

        % Prediction current batch
        v = sum(w1.*s) + b1;
        % dvdx = sum(w0x.*w1.*sp); % unused
        d2vdx2 = sum(w0x.^2.*w1.*spp);
        % dvdy = sum(w0y.*w1.*sp); % unused
        d2vdy2 = sum(w0y.^2.*w1.*spp);
        % at boundary
        vleft = sum(w1.*sleft) + b1; % [1,Ny]
        vright = sum(w1.*sright) + b1;
        vtop = sum(w1.*stop) + b1; % [1,Nx]
        vbottom = sum(w1.*sbottom) + b1;

        bleft = sum(w1.*snleft) + b1; % [1,Ny]
        bright = sum(w1.*snright) + b1;


        % y derivatives to weights and biases
        dvdw0x = w1.*sp.*xi;
        dvdw0y = w1.*sp.*yi;
        dvdb0 = w1.*sp;
        dvdw1 = s;
        dvdb1 = 1;

        % dvdx derivatives to weights and biases - unused
        % dvpdw0x = w1.*sp + w0x.*w1.*spp.*xi;
        % dvpdw0y = w0x.*w1.*spp.*yi;
        % dvpdb0 = w0x.*w1.*spp;
        % dvpdw1 = w0x.*sp;
        % dvpdb1 = 0;

        % d2vdx2 derivatives to weights and biases
        dvxppdw0x = 2*w0x.*w1.*spp + w0x.^2.*w1.*sppp.*xi; % ypp = d2ydx2
        dvxppdw0y = w0x.^2.*w1.*sppp.*yi;
        dvxppdb0 = w0x.^2.*w1.*sppp;
        dvxppdw1 = w0x.^2.*spp;
        dvxppdb1 = 0;

        % d2vdy2 derivatives to weights and biases
        dvyppdw0x = w0y.^2.*w1.*sppp.*xi; % ypp = d2ydx2
        dvyppdw0y = 2*w0y.*w1.*spp + w0y.^2.*w1.*sppp.*yi;
        dvyppdb0 = w0y.^2.*w1.*sppp;
        dvyppdw1 = w0y.^2.*spp;
        dvyppdb1 = 0;

        dsldx = sleft.*(1 - sleft); % [Nn,Ny]
        % vleft derivatives to weights and biases
        dvldw0x = dsldx.*w1.*xleft; % [Nn,Ny]
        dvldw0y = dsldx.*w1.*yleft;
        dvldb0 = w1.*dsldx;
        dvldw1 = sleft;
        dvldb1 = 1;

        dsrdx = sright.*(1 - sright);
        % vright derivatives to weights and biases
        dvrdw0x = dsrdx.*w1.*xright;
        dvrdw0y = dsrdx.*w1.*yright;
        dvrdb0 = w1.*dsrdx;
        dvrdw1 = sright;
        dvrdb1 = 1;

        dstdx = stop.*(1 - stop); % [Nn,Nx]
        % vleft derivatives to weights and biases
        dvtdw0x = dstdx.*w1.*xtop; % [Nn,Nx]
        dvtdw0y = dstdx.*w1.*ytop; 
        dvtdb0 = w1.*dstdx;
        dvtdw1 = stop;
        dvtdb1 = 1;

        dsbdx = sbottom.*(1 - sbottom);
        % vleft derivatives to weights and biases
        dvbdw0x = dsbdx.*w1.*xbottom;
        dvbdw0y = dsbdx.*w1.*ybottom;
        dvbdb0 = w1.*dsbdx;
        dvbdw1 = sbottom;
        dvbdb1 = 1;


        dslndx = snleft.*(1 - snleft); % [Nn,Ny]
        % vleft derivatives to weights and biases
        dvlndw0x = dslndx.*w1.*xnleft; % [Nn,Ny]
        dvlndw0y = dslndx.*w1.*ynleft;
        dvlndb0 = w1.*dslndx;
        dvlndw1 = snleft;
        dvlndb1 = 1;

        dsrndx = snright.*(1 - snright);
        % vright derivatives to weights and biases
        dvrndw0x = dsrndx.*w1.*xnright;
        dvrndw0y = dsrndx.*w1.*ynright;
        dvrndb0 = w1.*dsrndx;
        dvrndw1 = snright;
        dvrndb1 = 1;
        

        % Weights and biases update
        % l = mean((y - dt*alpha*d2udx2*d2udy2 - Ui).^2) + (y0)^2 + (y1)^2; % loss function
        b1 = b1 - learning_rate*(2*(v - dt*alpha*(d2vdx2+d2vdy2) - Ui)* ...
            (dvdb1 - dt*alpha*(dvxppdb1+dvyppdb1)) + sum(2*(vleft-bleft).*(dvldb1-dvlndb1)) + ...
            sum(2*(vright-bright).*(dvrdb1-dvrndb1)) + sum(2*(vtop-btop).*dvtdb1) + sum(2*(vbottom-bbottom).*dvbdb1));
        vleft = repmat(vleft,Nn,1); vright = repmat(vright,Nn,1);
        vtop = repmat(vtop,Nn,1); vbottom = repmat(vbottom,Nn,1);
        w0x = w0x - learning_rate*(2*(v - dt*alpha*(d2vdx2+d2vdy2) - Ui)* ...
            (dvdw0x - dt*alpha*(dvxppdw0x+dvyppdw0x)) + sum(2*(vleft-bleft).*(dvldw0x-dvlndw0x),2) + ...
            sum(2*(vright-bright).*(dvrdw0x-dvrndw0x),2) + sum(2*(vtop-btop).*dvtdw0x,2) + sum(2*(vbottom-bbottom).*dvbdw0x,2));
        w0y = w0y - learning_rate*(2*(v - dt*alpha*(d2vdx2+d2vdy2) - Ui)* ...
            (dvdw0y - dt*alpha*(dvxppdw0y+dvyppdw0y)) + sum(2*(vleft-bleft).*(dvldw0y-dvlndw0y),2) + ...
            sum(2*(vright-bright).*(dvrdw0y-dvrndw0y),2) + sum(2*(vtop-btop).*dvtdw0y,2) + sum(2*(vbottom-bbottom).*dvbdw0y,2));
        b0 = b0 - learning_rate*(2*(v - dt*alpha*(d2vdx2+d2vdy2) - Ui)* ...
            (dvdb0 - dt*alpha*(dvxppdb0+dvyppdb0)) + sum(2*(vleft-bleft).*(dvldb0-dvlndb0),2) + ...
            sum(2*(vright-bright).*(dvrdb0-dvrndb0),2) + sum(2*(vtop-btop).*dvtdb0,2) + sum(2*(vbottom-bbottom).*dvbdb0,2));
        w1 = w1 - learning_rate*(2*(v - dt*alpha*(d2vdx2+d2vdy2) - Ui)* ...
            (dvdw1 - dt*alpha*(dvxppdw1+dvyppdw1)) + sum(2*(vleft-bleft).*(dvldw1-dvlndw1),2) + ...
            sum(2*(vright-bright).*(dvrdw1-dvrndw1),2) + sum(2*(vtop-btop).*dvtdw1,2) + sum(2*(vbottom-bbottom).*dvbdw1,2));
    end
    params = [w0x;w0y;b0;w1;b1];
end

function [y,dydx,d2ydx2] = predict(params,x,y,Nn)
    w0x = params(1:Nn);
    w0y = params(Nn+1:Nn*2);
    b0 = params(Nn*2+1:Nn*3);
    w1 = params(Nn*3+1:end-1);
    b1 = params(end);

    Nx = size(x,2);
    Ny = size(y,1);
    Ni = Nx*Ny;
    w0x = repmat(w0x,1,Ni);
    w0y = repmat(w0y,1,Ni);
    b0 = repmat(b0,1,Ni);
    w1 = repmat(w1,1,Ni);
    x = repmat(reshape(x,[1,Ni]),Nn,1);
    y = repmat(reshape(y,[1,Ni]),Nn,1);

    % Sigmoid function derivative with w0, b0, and x inputs
    s = mySigmoid(w0x.*x + w0y.*y + b0);
    dsdx = s.*(1 - s);
    d2sdx2 = dsdx.*(1 - 2*s);

    % Prediction
    y = sum(w1.*s) + b1;
    dydx = sum(w0x.*w1.*dsdx);
    d2ydx2 = sum(w0x.^2.*w1.*d2sdx2);
    y = reshape(y,[Ny,Nx]);
end
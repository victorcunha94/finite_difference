
% poisson2.m  -- solve the Poisson problem u_{xx} + u_{yy} = f(x,y)
% on [a,b] x [a,b].  
% 
% The 5-point Laplacian is used at interior grid points.
% This system of equations is then solved using backslash.
% 
% From  http://www.amath.washington.edu/~rjl/fdmbook/chapter3  (2007)
clear all
close all
clc

a = 0; 
b = 1; 
mvec = [16,32,64,128,256,512,1024];
hvec = []; errovec = [];

for m = mvec

    h = (b-a)/(m+1);
    hvec = [hvec,h];

    x = linspace(a,b,m+2);   % grid points x including boundaries
    y = linspace(a,b,m+2);   % grid points y including boundaries
    
    
    [X,Y] = meshgrid(x,y);      % 2d arrays of x,y values
    X = X';                     % transpose so that X(i,j),Y(i,j) are
    Y = Y';                     % coordinates of (i,j) point
    
    Iint = 2:m+1;              % indices of interior points in x
    Jint = 2:m+1;              % indices of interior points in y
    Xint = X(Iint,Jint);       % interior points
    Yint = Y(Iint,Jint);
    
    f = @(x,y) 1.25*exp(x+y/2);         % f(x,y) function
    
    rhs = f(Xint,Yint);        % evaluate f at interior points for right hand side
                   % rhs is modified below for boundary conditions.
    
    utrue = exp(X+Y/2);        % true solution for test problem
    
    % set boundary conditions around edges of usoln array:
    
    usoln = utrue;              % use true solution for this test problem
                    % This sets full array, but only boundary values
                    % are used below.  For a problem where utrue
                    % is not known, would have to set each edge of
                    % usoln to the desired Dirichlet boundary values.
    
    
    % adjust the rhs to include boundary terms:
    rhs(:,1) = rhs(:,1) - usoln(Iint,1)/h^2;
    rhs(:,m) = rhs(:,m) - usoln(Iint,m+2)/h^2;
    rhs(1,:) = rhs(1,:) - usoln(1,Jint)/h^2;
    rhs(m,:) = rhs(m,:) - usoln(m+2,Jint)/h^2;
    
    
    % convert the 2d grid function rhs into a column vector for rhs of system:
    F = reshape(rhs,m*m,1);
    
    % form matrix A:
    I = speye(m);
    e = ones(m,1);
    T = spdiags([e -4*e e],[-1 0 1],m,m);
    S = spdiags([e e],[-1 1],m,m);
    A = (kron(I,T) + kron(S,I)) / h^2;
    
    
    % Solve the linear system:
    uvec = A\F;  
    
    % reshape vector solution uvec as a grid function and 
    % insert this interior solution into usoln for plotting purposes:
    % (recall boundary conditions in usoln are already set) 
    
    usoln(Iint,Jint) = reshape(uvec,m,m);
    
    % assuming true solution is known and stored in utrue:
    err     = max(max(abs(usoln-utrue)));  
    errovec = [errovec,err];
    fprintf('Error relative to true solution of PDE = %10.3e \n',err)

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculo da ordem p
p_m = zeros(1,length(hvec)-1);

for i = 1:length(hvec)-1
  p_m(1,i)=log((errovec(i+1)/errovec(i)))/(log((hvec(1,i+1)/hvec(1,i))));
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Escrevendo as informaçoes relevantes na tela  
format short
fprintf('Aproximação da ordem p\n')
p_m


%% Plot results:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clf
hold on

% plot grid:
% plot(X,Y,'g');  plot(X',Y','g')

% plot solution:
figure(1)
contour(X,Y,usoln,30,'k')

axis([a b a b])
daspect([1 1 1])
title('Contour plot of computed solution')
hold off

figure(2)

loglog(hvec,errovec,'ko--','LineWidth',1,'MarkerSize',10)
title('Teste ordem escala loglog')
hold on
grid on
loglog(hvec,hvec, 'bo-','LineWidth',2,'MarkerFaceColor','b')
loglog(hvec,hvec.*hvec,'rd-','LineWidth',2,'MarkerFaceColor','r')
xlabel('h','FontSize',16)
ylabel('Erro Númerico','FontSize',16)
hl = legend('show');
set(hl, 'Interpreter','latex')
legend('Location','southeast')
legend('$\parallel u - U \parallel _{\infty }$','Reta $O(h)$ ','Reta $O(h^{2})$')
legend('FontSize',13)
set(gca, 'FontSize', 12);
xlim([hvec(end)*0.8 (hvec(1)*1.2)])


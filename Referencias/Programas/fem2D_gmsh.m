% 2_D-Fem for Laplace-Operator
% Malla generada con GMSH, almacenado como .msh
% Initialisation

% u_exact_fun = @(x,y) sin(pi*x-pi/3).*cos(pi*y);
u_exact_fun = @(x,y) sin(pi*x).*y;

% malla1=load_gmsh('cuadroD.msh');
malla1=load_gmsh('cuadroDN.msh');

nel=malla1.nbTriangles;
nver=malla1.nbNod;
Coordinates=malla1.POS(:,1:2);
Elements3=malla1.TRIANGLES(:,1:3);
LadosFrontera=malla1.LINES;

% unique(LadosFrontera(:,3))

Dirichlet=find(LadosFrontera(:,3)==12);
Dirichlet=unique(LadosFrontera(Dirichlet,1:2));

Neumann=LadosFrontera(find(LadosFrontera(:,3)==13),1:2);

Freenodes=setdiff(1:nver,unique(Dirichlet));
A=sparse(nver,nver);
b=sparse(nver,1);

% Assembly and Volume Forces
for j=1:nel
    coord = Coordinates(Elements3(j,:),:)';
    areaT = abs(det([ones(1,3);coord]))/2; 
    G = [ones(1,3);coord] \ [zeros(1,2);eye(2)];
    M =  areaT * G * G';
    A(Elements3(j,:),Elements3(j,:))=A(Elements3(j,:),Elements3(j,:)) + M;
    b(Elements3(j,:))=b(Elements3(j,:))+areaT*f(sum(Coordinates(Elements3(j,:),:))/3)/3;
end

%Neumann conditions  
if ~isempty(Neumann)
  for j=1:size(Neumann,1)
        tang = Coordinates(Neumann(j,1),:)-Coordinates(Neumann(j,2),:);
        vecnor = [-tang(2),tang(1)];
        puntoMedio = sum(Coordinates(Neumann(j,:),:))/2;
        b(Neumann(j,:))=b(Neumann(j,:))+norm(tang)*g(puntoMedio,vecnor)/2;
    end
end

%Dirichlet Conditions
u=sparse(size(Coordinates,1),1);
coordD = Coordinates(unique(Dirichlet),:)';
u(unique(Dirichlet))=u_d(coordD);
b=b-A*u;

%Computation of the solution
u(Freenodes)=A(Freenodes,Freenodes)\b(Freenodes);

%graphic representation
trisurf(Elements3,Coordinates(:,1),Coordinates(:,2),full(u));
title('Solucion aproximada');
colorbar

figure
u_exact = u_exact_fun(Coordinates(:,1), Coordinates(:,2));
trisurf(Elements3, Coordinates(:,1), Coordinates(:,2), u_exact);
title('Solucion exacta');
colorbar

% Error L^2
eL2=0;
slnL2 = 0;
for j=1:nel
    coord = Coordinates(Elements3(j,:),:)';
    areaT = abs(det([1,1,1;coord]))/2;
%     bary = sum(coord,2)/3;
%     Ue = u_exact_fun(bary(1),bary(2));
%     Ua = sum(u(Elements3(j,:)))/3;
%     eL2 = eL2 + mT*(Ue-Ua)^2;
%     slnL2 = slnL2 + mT*Ue^2;
    ptm = coord*[1/2 0 1/2;1/2 1/2 0;0 1/2 1/2];
    Ue = u_exact_fun(ptm(1,:),ptm(2,:));
    Ua = u(Elements3(j,:))'*[1/2 0 1/2;1/2 1/2 0;0 1/2 1/2];
    eL2 = eL2 + areaT*sum((Ue-Ua).^2)/3;
    slnL2 = slnL2 + areaT*sum(Ue.^2)/3;
end
[eL2 slnL2]
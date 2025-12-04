function Stress = g(x,n)
%G   Data on the Neumann boundary
%   Y = G(X) returns values of the normal-derivative at N discrete points on
%   the Neumann boundary. This input data has to be choosen by the user. X
%   has dimension N x 2 and Y has dimension N x 1.
%
%
%   See also FEM2D, F, and U_D.
%

%    J. Alberty, C. Carstensen and S. A. Funken  02-11-99
%    File <g.m> in $(HOME)/acf/fem2d/
%    This Neumann boundary data is used to compute Fig. 3 in 
%    "Remarks around 50 lines of Matlab: Short finite element 
%    implementation"

% Stress = zeros(size(x,1),1);
if abs(n(1))>0
    Stress = pi*cos(pi*x(1))*x(2);
%     Stress = pi*cos(pi*x(1)-pi/3)*cos(pi*x(2));
else
    Stress = sin(pi*x(1));
%     Stress = -pi*sin(pi*x(1)-pi/3)*sin(pi*x(2));
end

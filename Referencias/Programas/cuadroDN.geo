lc = 0.2;
Point(1) = {0.0,0.0,0.0,lc};
Point(2) = {2,0.0,0.0,lc};
Point(3) = {2,2,0.0,lc};
Point(4) = {0,2,0.0,lc};

Line(1) = {4,3};
Line(2) = {3,2};
Line(3) = {2,1};
Line(4) = {1,4};

Line Loop(9) = {2,3,4,1};
Plane Surface(11) = {9};

Physical Line(13)={1,2};
Physical Line(12)={3,4};

Physical Surface(14)={11};


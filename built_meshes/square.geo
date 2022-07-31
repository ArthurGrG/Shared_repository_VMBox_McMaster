L = 1.0;
H = 1.0;

Point(1) = {-0.5*L, 0.5*H, 0, 0.025};
Point(2) = {-0.5*L, -0.5*H, 0, 0.025};
Point(3) = {0.5*L, -0.5*H, 0, 0.025};
Point(4) = {0.5*L, 0.5*H, 0, 0.025}; 

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {4, 4};
Line(6) = {3, 3}; 
Line(7) = {2, 2};
Line(8) = {1, 1}; 

Curve Loop(9) = {1, 2, 3, 4};

Plane Surface(1) = {9}; 

Physical Curve("Bottom", 8) = {2};
Physical Curve("Top", 9) = {4};
Physical Curve("Left", 10) = {1};
Physical Curve("Right", 11) = {3};

Physical Curve("PointTopRight", 12) = {5};
Physical Curve("PointBottomRigth", 13) = {6};
Physical Curve("PointBottomLeft", 14) = {7};
Physical Curve("PointTopLeft", 15) = {8};

Physical Surface("UnitSquare", 2) = {1}; 
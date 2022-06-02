Point(1) = {-Sqrt(3)/2, -0.5, 0, 0.05}; 
Point(2) = {0, -1, 0, 0.05}; 
Point(3) = {Sqrt(3)/2, -0.5, 0, 0.05}; 
Point(4) = {Sqrt(3)/2, 0.5, 0, 0.05}; 
Point(5) = {0, 1, 0, 0.07}; 
Point(6) = {-Sqrt(3)/2, 0.5, 0, 0.05};

Line(1) = {1, 2}; 
Line(2) = {2, 3}; 
Line(3) = {3, 4}; 
Line(4) = {4, 5}; 
Line(5) = {5, 6}; 
Line(6) = {6, 1};

Curve Loop(7) = {1, 2, 3, 4, 5, 6};

Plane Surface(1) = {7}; 

Physical Surface("UnitHexagonal", 2) = {1}; 


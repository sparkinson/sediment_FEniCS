edgeLength = 0.025;

Point(1) = {0, 0, 0, 1.0};
Extrude {0, 0.4, 0} {
  Point{1}; Layers{0.4/edgeLength};
}
Extrude {0.2, 0.0, 0.0} {
  Line{1}; Layers{0.2/edgeLength};
}
Extrude {0.8, 0.0, 0.0} {
  Line{2}; Layers{0.8/edgeLength};
}
Physical Line(1) = {3, 7}; //bottom
Physical Line(2) = {1, 6}; //sides
Physical Line(3) = {4, 8}; //top
Physical Surface(1) = {9, 5};

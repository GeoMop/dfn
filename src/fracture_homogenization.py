"""
1. DONE, gmsh reference simplex
2. generate square fracture, side a = 1,
   uniform random orientation f(theta) = sin(theta)
   centre in cube (-a sqrt(2)) (1,1,1) : (1+a sqrt(2))(1,1,1)
3. if positive intersetion area -> accept
4. generate mesh
   random corsssection (0.01, 0.1)*a
5. compute three fluxes in single flow123d calculation (three rotations of the simplex + fracture)
6. write down:
   fracture angles, conductivity angles
   conductivity eigenvalues
   fracture size, crossection
   intersection area
   intersection center of mass

7. generate minimum 10 000 samples
8.
"""




import gmsh

geo = gmsh.Geometry('occ', "three_frac_symmetric", verbose=True)
ss = [geo.make_simplex(dim=d).translate([2*d, 0, 0]) for d in  range(4) ]


el_size = 0.1
mesh = gmsh.MeshOptions()
mesh.CharacteristicLengthMin = el_size
mesh.CharacteristicLengthMax = el_size
geo.make_mesh(ss)

geo.show()
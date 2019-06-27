from gmsh_api import gmsh
import fracture_generator as fg
import numpy as np

geo = gmsh.Geometry('occ', "three_frac_symmetric", verbose=False)
geopt = gmsh.GeometryOptions()
geopt.ToleranceBoolean = 1e-2
geopt.MatchMeshTolerance = 1e-2
geopt.Tolerance = 1e-2


def p32_over_p30(exp, size_range):
    return -exp * (size_range[1] ** (2 - exp) - size_range[0] ** (2 - exp)) / (2 - exp) / (size_range[1] ** (-exp) - size_range[0] ** (-exp))

def p30_scale(exp, orig_range, new_range):
    return (new_range[1] ** (-exp) - new_range[0] ** (-exp)) / (orig_range[1] ** (-exp) - orig_range[0] ** (-exp))

size_range = [0.1, 100]
cube_side = 5
# SKB HZ
power_law_exp = 2.1
power_law = fg.PowerLawSize(power=power_law_exp, size_min=0.038, size_max=564)
fischer = fg.FisherOrientation(270, 0, 15.2)
p32_intensity = 0.543
p30_intensity = p32_intensity / p32_over_p30(power_law_exp, [0.038, 564])
p30_intensity = p30_intensity * p30_scale(power_law_exp, [0.038, 564], size_range)
volume = cube_side ** 3
intensity = volume * p30_intensity
n_fractures = np.random.poisson(lam=intensity, size=1)[0]


l_rmin, l_rmax = cube_side * np.log(size_range[0]), cube_side * np.log(size_range[1])
b_sides = [l_rmax - l_rmin, cube_side, cube_side]
box = geo.box(b_sides, [0.5*(l_rmax + l_rmin), 0.5 * cube_side, 0.5 * cube_side])
fr_base = geo.rectangle()

fractures = []
for i in range(n_fractures):
    if i % 1000 == 0:
        print('done ... {}% ({}/{})'.format(100*i/n_fractures, i, n_fractures))
    fr_a = fischer.sample_axis_angle()[0]
    axis, angle = fr_a[:3], fr_a[3]
    size = power_law.sample(size_min=size_range[0], size_max=size_range[1])


    # random position
    pt_x = np.log(size) * cube_side
    pt_y, pt_z = cube_side * np.random.rand(2)
    print(i, size, pt_x, pt_y, pt_z)
    # axial rotation
    normal_angle = 2 * np.pi * np.random.uniform()
    fr = fr_base.copy().scale([size, size, size])
    fr = fr.rotate([0, 0, 1], normal_angle).rotate(axis, angle).translate([pt_x, pt_y, pt_z])
    fr = fr.intersect(box)
    fractures.append(fr)

mesh = gmsh.MeshOptions()
mesh.ToleranceInitialDelaunay = 1e-3
mesh.CharacteristicLengthMin = size_range[0]
mesh.CharacteristicLengthMax = size_range[1]
geo.make_mesh(fractures, dim=2)
geo.show()

print(p30_intensity, intensity, n_fractures)

# shape = fr_base.scale([fr.rx, fr.ry, 1]) \
#     .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
#     .translate(fr.centre) \
#     .set_region(fr.region)
# shapes.append(shape)
#
# fracture_fragments = self.fragment(*shapes)
# return fracture_fragments
#
# geo.make_fractures()
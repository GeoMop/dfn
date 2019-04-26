from fracture_type import FractureType
from fracture_generator import FractureGenerator
from gmsh_api import gmsh
from math import *
#from random_frac import *


r_min = 0.038
r_max = 564
SKB_DATA = [{"name": 'NS', "trend": 292, "plunge": 1,  "strike": 202, "dip": 89, "k": 17.8, "r_0": 0.038, "kappa": 2.5,
             "r_min": r_min, "r_max": r_max, "p_32": 0.073},
            {"name": 'NE', "trend": 326, "plunge": 2,  "strike": 236, "dip": 88, "k": 14.3, "r_0": 0.038, "kappa": 2.7,
             "r_min": r_min, "r_max": r_max, "p_32": 0.319},
            {"name": 'NW', "trend": 60,  "plunge": 6,  "strike": 330, "dip": 84, "k": 12.9, "r_0": 0.038, "kappa": 3.1,
             "r_min": r_min, "r_max": r_max, "p_32": 0.107},
            {"name": 'EW', "trend": 15,  "plunge": 2,  "strike": 285, "dip": 88, "k": 14.0, "r_0": 0.038, "kappa": 3.1,
             "r_min": r_min, "r_max": r_max, "p_32": 0.088},
            {"name": 'HZ', "trend": 5,   "plunge": 86, "strike": 275, "dip": 4,  "k": 15.2, "r_0": 0.038, "kappa": 2.38,
             "r_min": r_min, "r_max": r_max, "p_32": 0.543}]

nf = 20       # number of fractures
mind = 0.05    # minimal distance from boundary
minr = 0.0004     # minimal fracture radius
maxr = 0.5     # maximal fracture radius
file_name = "test_api.msh2"


def create_fractures():
    fractures_data = []
    for fracture_type in SKB_DATA:
        fr_type = FractureType(**fracture_type)
        fr_type.set_n_fractures(volume=1, use_poisson=True)

        fr_generator = FractureGenerator(fr_type)
        fracture_data = fr_generator.generate_fractures(mind, minr, maxr)
        fractures_data.extend(fracture_data)

    fr_generator.write_fractures(fractures_data, file_name)
    fractures_data = fr_generator.read_fractures(file_name)

    generate_mesh(fractures_data, max_el_size=1, file_name=file_name, verbose=1, shape="circle")


def generate_mesh(fractures_data, max_el_size, file_name, verbose=0, shape="circle"):
    r""" Create mesh and write it to a file.

    Parameters
    ----------
    fractures_data : list of FractureData
      Array of objects defining fractures.
    max_el_size : double
      Maximal size of mesh element.
    file_name : str
      File name to write mesh into.
    verbose : {0, 1}
      If set to 1, messages during mesh generation will be printed.
    """
    model = gmsh.model
    factory = model.occ
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", verbose)
    model.add("test_api")

    # set options
    gmsh.option.setNumber("Geometry.Tolerance", 0)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 0)
    gmsh.option.setNumber("Geometry.MatchMeshTolerance", 0)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-12)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max_el_size*0.01)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_el_size)
    # gmsh.option.setNumber("Mesh.MinimumCurvePoints", 5)

    # Generate fractures
    if shape == "circle":
        fractures = generate_disks(factory, fractures_data)
    # elif shape == "rectangle":
    #     fractures = generate_rectangles(factory, fractures_data)
    # elif shape == "ellipse":
    #     fractures = generate_ellipses(factory, fractures_data)

    # set physical id and name of fractures
    fractures_phys_id = model.addPhysicalGroup(2, [x[1] for x in fractures], -1)
    model.setPhysicalName(2, fractures_phys_id, "fractures")

    # define 3d volume and embed fractures into it
    box = factory.addBox(0, 0, 0, 1, 1, 1)

    factory.synchronize()
    # Embed the model entities of dimension 'dim', and tags 'tag' in the other model entity which is given by
    # dimension (3) and tag (box)
    model.mesh.embed(2, [tag for dim, tag in fractures], 3, box)

    # define physical id of volume and its boundaries
    box_phys_id = model.addPhysicalGroup(3, [box], -1)
    model.setPhysicalName(3, box_phys_id, "box")
    # box_bdry = model.getBoundary([(3,box)], True)
    # box_bdry_left_phys_id = model.addPhysicalGroup(2, [box_bdry[0][1]], -1)
    # box_bdry_right_phys_id = model.addPhysicalGroup(2, [box_bdry[1][1]], -1)
    # model.setPhysicalName(2, box_bdry_left_phys_id, ".left")
    # model.setPhysicalName(2, box_bdry_right_phys_id, ".right")

    # Define fields for mesh refinement
    # Compute the distance from curves, each curve is replaced by NNodesByEdge equidistant nodes
    # and the distance from those nodes is computed.
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumber(1, "NNodesByEdge", 100)
    frac_bdry = model.getBoundary(fractures, False, False, False)
    # Set the numerical list option "EdgesList" to list of "frac_bdry" tags for field tag 1.
    model.mesh.field.setNumbers(1, "EdgesList", [tag for dm, tag in frac_bdry if dm == 1])

    # Threshold
    # F = LCMin if Field[IField] <= DistMin,
    # F = LCMax if Field[IField] >= DistMax,
    # F = interpolation between LcMin and LcMax if DistMin < Field[IField] < DistMax
    model.mesh.field.add("Threshold", 2)
    # Index of the field to evaluate (in this case 1 is "Distance")
    model.mesh.field.setNumber(2, "IField", 1)
    # Element size inside DistMin
    model.mesh.field.setNumber(2, "LcMin", max_el_size * 0.03)
    # Element size outside DistMax
    model.mesh.field.setNumber(2, "LcMax", max_el_size)
    # Distance from entity up to which element size will be LcMin, it should depend on particular model
    model.mesh.field.setNumber(2, "DistMin", 0.001)
    # Distance from entity after which element size will be LcMax, it should depend on particular model
    model.mesh.field.setNumber(2, "DistMax", 0.5)

    # Set threshold as the background mesh size field.
    model.mesh.field.setAsBackgroundMesh(2)

    # generate mesh, write to file and output number of entities that produced error
    factory.synchronize()
    gmsh.write(file_name + '.brep')
    model.mesh.generate(3)
    bad_entities = model.mesh.getLastEntityError()
    gmsh.write(file_name)
    gmsh.finalize()

    return len(bad_entities)


def generate_disks(factory, fractures_data):
    """
    Generate disks fractures
    :param factory: gmsh.model.occ
    :param fractures_data: list of FractureData
    :return: tags
    """
    disks = []
    tags = []
    for f in fractures_data:
        pc = factory.addPoint(f.centre[0], f.centre[1], f.centre[2], f.r)

        p0 = factory.addPoint(f.centre[0] + f.r,               f.centre[1],                     f.centre[2], f.r)
        p1 = factory.addPoint(f.centre[0] + f.r * cos(pi*2/3), f.centre[1] + f.r * sin(pi*2/3), f.centre[2], f.r)
        p2 = factory.addPoint(f.centre[0] + f.r * cos(pi*4/3), f.centre[1] + f.r * sin(pi*4/3), f.centre[2], f.r)
        factory.rotate([(0, p0), (0, p1), (0, p2)], f.centre[0], f.centre[1], f.centre[2], f.rotation_axis[0],
                       f.rotation_axis[1], f.rotation_axis[2], f.rotation_angle)

        c1 = factory.addCircleArc(p0, pc, p1)
        c2 = factory.addCircleArc(p1, pc, p2)
        c3 = factory.addCircleArc(p2, pc, p0)
        cl = factory.addCurveLoop([c1, c2, c3])
        d = factory.addPlaneSurface([cl])

        disks.append((2, d))
        tags.append(f.tag)

    # fragment fractures
    fractures, fractures_map = factory.fragment(disks, [])
    assert len(fractures_map) == len(disks)
    return fractures

#
# def generate_rectangles(factory, fractures_data):
#     """
#     Generate rectangular fractures
#     :param factory: gmsh.model.occ
#     :param fractures_data: list of FractureData
#     :return: tags
#     """
#     rectangles = []
#     tags = []
#
#     for f in fractures_data:
#         axis, angle = f.get_rotation_axis_angle()
#         # Vertices of the rectangle
#         p0 = factory.addPoint(f.centre[0] + f.r1, f.centre[1] - f.r2, f.centre[2])
#         p1 = factory.addPoint(f.centre[0] + f.r1, f.centre[1] + f.r2, f.centre[2])
#         p2 = factory.addPoint(f.centre[0] - f.r1, f.centre[1] + f.r2, f.centre[2])
#         p3 = factory.addPoint(f.centre[0] - f.r1, f.centre[1] - f.r2, f.centre[2])
#
#         factory.rotate([(0, p0), (0, p1), (0, p2), (0, p3)], f.centre[0], f.centre[1], f.centre[2],
#                        axis[0], axis[1], axis[2], angle)
#
#         # Create lines
#         l1 = factory.addLine(p0, p1)
#         l2 = factory.addLine(p1, p2)
#         l3 = factory.addLine(p2, p3)
#         l4 = factory.addLine(p3, p0)
#
#         cl = factory.addCurveLoop([l1, l2, l3, l4])
#         d = factory.addPlaneSurface([cl])
#         rectangles.append((2, d))
#         tags.append(f.tag)
#
#     # fragment fractures
#     fractures, fractures_map = factory.fragment(rectangles, [])
#     assert len(fractures_map) == len(rectangles)
#     return fractures
#
#
# def generate_ellipses(factory, fractures_data):
#     """
#     Generate ellipses fractures
#     :param factory: gmsh.model.occ
#     :param fractures_data: list of FractureData
#     :return: tags
#     """
#     ellipses = []
#     tags = []
#
#     for f in fractures_data:
#         axis, angle = f.get_rotation_axis_angle()
#
#         ellipse = factory.addEllipse(f.centre[0], f.centre[1], f.centre[2], f.r1, f.r2)
#
#         factory.rotate([(1, ellipse)], f.centre[0], f.centre[1], f.centre[2],
#                        axis[0], axis[1], axis[2], angle)
#
#         cl = factory.addCurveLoop([ellipse])
#         d = factory.addPlaneSurface([cl])
#         ellipses.append((2, d))
#         tags.append(f.tag)
#
#     # fragment fractures
#     fractures, fractures_map = factory.fragment(ellipses, [])
#
#     assert len(fractures_map) == len(ellipses)
#     return fractures


if __name__ == "__main__":
    create_fractures()

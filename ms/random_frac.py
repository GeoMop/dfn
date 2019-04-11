from math import *
import numpy as np
from random import *
import gmsh_api.gmsh as gmsh


class FractureData:

    def __init__(self, x, y, z, r1, r2, nx, ny, nz, tag):
        self.centre = np.array([x, y, z])
        self.r1, self.r2 = r1, r2
        self.normal = np.array([nx, ny, nz])
        self.tag = tag

    def get_rotation_axis_angle(self, from_vec=np.array([0, 0, 1])):
        axis = np.cross(from_vec, self.normal)
        a_sin = np.linalg.norm(axis)
        a_cos = np.dot(from_vec, self.normal)
        angle = atan2(a_sin, a_cos)
        return axis, angle


def write_fractures(fracture_data, file_name):
    f = open(file_name, 'w')
    for d in fracture_data:
      f.write("%f %f %f %f %f %f %f %f %d\n" % (d.centre[0], d.centre[1], d.centre[2], d.r1, d.r2, d.normal[0],
                                                d.normal[1], d.normal[2], d.tag))
    f.close()


def read_fractures(file_name):
    data = []
    f = open(file_name, 'r')
    for l in f:
        x, y, z, r1, r2, nx, ny, nz = [float(i) for i in l.split(' ')[:-1]]
        tag = int(l.split(' ')[-1])
        d = FractureData(x, y, z, r1, r2, nx, ny, nz, tag)
        data.append(d)
    f.close()
    return data


def generate_fractures(n_fractures, min_distance, min_radius, max_radius):
    fractures = []
    for i in range(n_fractures):
        x = uniform(2*min_distance, 1-2*min_distance)
        y = uniform(2*min_distance, 1-2*min_distance)
        z = uniform(2*min_distance, 1-2*min_distance)
        r1 = uniform(min_radius, max_radius)
        r1 = min(max(0, x-min_distance), max(0, 1-x-min_distance), max(0, y-min_distance), max(0, 1-y-min_distance),
                 max(0, z-min_distance), max(0, 1-z-min_distance), r1)
        r2 = uniform(min_radius, r1)
        r2 = min(max(0, x-min_distance), max(0, 1-x-min_distance), max(0, y-min_distance), max(0, 1-y-min_distance),
                 max(0, z-min_distance), max(0, 1-z-min_distance), r2)
        nx, ny, nz = random(), random(), random()
        fractures.append(FractureData(x, y, z, r1, r2, nx, ny, nz, i*100))

    return fractures


def calculate_area(model, tag):
    jacs, dets, pts = model.mesh.getJacobians(2, "Gauss1", tag) # 2 = triangle
    return sum(dets)


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
    elif shape == "rectangle":
        fractures = generate_rectangles(factory, fractures_data)
    elif shape == "ellipse":
        fractures = generate_ellipses(factory, fractures_data)

    # set physical id and name of fractures
    fractures_phys_id = model.addPhysicalGroup(2, [x[1] for x in fractures], -1)
    model.setPhysicalName(2, fractures_phys_id, "fractures")

    # define 3d volume and embed fractures into it
    box = factory.addBox(0, 0, 0, 1, 1, 1)

    factory.synchronize()
    model.mesh.embed(2, [x[1] for x in fractures], 3, box)

    # define physical id of volume and its boundaries
    box_phys_id = model.addPhysicalGroup(3, [box], -1)
    model.setPhysicalName(3, box_phys_id, "box")
    # box_bdry = model.getBoundary([(3,box)], True)
    # box_bdry_left_phys_id = model.addPhysicalGroup(2, [box_bdry[0][1]], -1)
    # box_bdry_right_phys_id = model.addPhysicalGroup(2, [box_bdry[1][1]], -1)
    # model.setPhysicalName(2, box_bdry_left_phys_id, ".left")
    # model.setPhysicalName(2, box_bdry_right_phys_id, ".right")

    # define fields for mesh refinement
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumber(1, "NNodesByEdge", 100)
    frac_bdry = model.getBoundary(fractures, False, False, False)
    model.mesh.field.setNumbers(1, "EdgesList", [tag for (dm, tag) in frac_bdry if dm == 1])
    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "IField", 1)
    model.mesh.field.setNumber(2, "LcMin", max_el_size*0.03)
    model.mesh.field.setNumber(2, "LcMax", max_el_size)
    model.mesh.field.setNumber(2, "DistMin", 0.001)
    model.mesh.field.setNumber(2, "DistMax", 0.5)
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
    # create fractures (currently we ignore r2 and create circular disks)
    disks = []
    tags = []
    for f in fractures_data:
        pc = factory.addPoint(f.centre[0], f.centre[1], f.centre[2], f.r1)
        axis, angle = f.get_rotation_axis_angle()
        p0 = factory.addPoint(f.centre[0] + f.r1,               f.centre[1],                      f.centre[2], f.r1)
        p1 = factory.addPoint(f.centre[0] + f.r1 * cos(pi*2/3), f.centre[1] + f.r1 * sin(pi*2/3), f.centre[2], f.r1)
        p2 = factory.addPoint(f.centre[0] + f.r1 * cos(pi*4/3), f.centre[1] + f.r1 * sin(pi*4/3), f.centre[2], f.r1)
        factory.rotate([(0, p0), (0, p1), (0, p2)], f.centre[0], f.centre[1], f.centre[2], axis[0], axis[1], axis[2], angle)
        c1 = factory.addCircleArc(p0, pc, p1)
        c2 = factory.addCircleArc(p1, pc, p2)
        c3 = factory.addCircleArc(p2, pc, p0)
        cl = factory.addCurveLoop([c1, c2, c3])
        d = factory.addPlaneSurface([cl])

        disks.append((2,d))
        tags.append(f.tag)

    # fragment fractures
    fractures, fractures_map = factory.fragment(disks, [])
    assert len(fractures_map) == len(disks)
    return fractures


def generate_rectangles(factory, fractures_data):
    """
    Generate rectangular fractures
    :param factory: gmsh.model.occ
    :param fractures_data: list of FractureData
    :return: tags
    """
    rectangles = []
    tags = []

    for f in fractures_data:
        axis, angle = f.get_rotation_axis_angle()
        # Vertices of the rectangle
        p0 = factory.addPoint(f.centre[0] + f.r1, f.centre[1] - f.r2, f.centre[2])
        p1 = factory.addPoint(f.centre[0] + f.r1, f.centre[1] + f.r2, f.centre[2])
        p2 = factory.addPoint(f.centre[0] - f.r1, f.centre[1] + f.r2, f.centre[2])
        p3 = factory.addPoint(f.centre[0] - f.r1, f.centre[1] - f.r2, f.centre[2])

        factory.rotate([(0, p0), (0, p1), (0, p2), (0, p3)], f.centre[0], f.centre[1], f.centre[2],
                       axis[0], axis[1], axis[2], angle)

        # Create lines
        l1 = factory.addLine(p0, p1)
        l2 = factory.addLine(p1, p2)
        l3 = factory.addLine(p2, p3)
        l4 = factory.addLine(p3, p0)

        cl = factory.addCurveLoop([l1, l2, l3, l4])
        d = factory.addPlaneSurface([cl])
        rectangles.append((2, d))
        tags.append(f.tag)

    # fragment fractures
    fractures, fractures_map = factory.fragment(rectangles, [])
    assert len(fractures_map) == len(rectangles)
    return fractures


def generate_ellipses(factory, fractures_data):
    """
    Generate ellipses fractures
    :param factory: gmsh.model.occ
    :param fractures_data: list of FractureData
    :return: tags
    """
    ellipses = []
    tags = []

    for f in fractures_data:
        axis, angle = f.get_rotation_axis_angle()

        ellipse = factory.addEllipse(f.centre[0], f.centre[1], f.centre[2], f.r1, f.r2)

        factory.rotate([(1, ellipse)], f.centre[0], f.centre[1], f.centre[2],
                       axis[0], axis[1], axis[2], angle)

        cl = factory.addCurveLoop([ellipse])
        d = factory.addPlaneSurface([cl])
        ellipses.append((2, d))
        tags.append(f.tag)

    # fragment fractures
    fractures, fractures_map = factory.fragment(ellipses, [])
    assert len(fractures_map) == len(ellipses)
    return fractures

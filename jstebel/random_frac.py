from math import *
import numpy as np
from random import *
import gmsh


class FractureData:

    def __init__(self, x, y, z, r1, r2, nx, ny, nz, tag):
        self.centre = np.array([x, y, z])
        self.r1, self.r2 = r1, r2
        self.normal = np.array([nx, ny, nz])
        self.tag = tag

    def get_rotation_axis_angle(self, from_vec = np.array([0,0,1])):
        axis = np.cross(from_vec, self.normal)
        a_sin = np.linalg.norm(axis)
        a_cos = np.dot(from_vec, self.normal)
        angle = atan2(a_sin, a_cos)
        return axis, angle


def write_fractures(fracture_data, file_name):
    f = open(file_name, 'w')
    for d in fracture_data:
        f.write("%f %f %f %f %f %f %f %f %d\n" % (d.centre[0], d.centre[1], d.centre[2], d.r1, d.r2, d.normal[0], d.normal[1], d.normal[2], d.tag))
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
    jacs, dets, pts = model.mesh.getJacobians(2, "Gauss1", tag)  # 2 = triangle
    return sum(dets)


def generate_mesh(fractures_data, max_el_size, file_name, verbose=0):
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

    # Compute mesh element sizes from values given at geometry points, default value: 1
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    # Automatically compute mesh element sizes from curvature, respects surface curvature, default value: 0
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Element size constraint options
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max_el_size*0.01)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_el_size)
    # gmsh.option.setNumber("Mesh.MinimumCurvePoints", 5)

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

        disks.append((2, d))
        tags.append(f.tag)

    # fragment fractures
    fractures, fractures_map = factory.fragment(disks, [])
    print('disks = ', disks)
    # List of fracture tuples (dim, tag)
    print('fractures = ', fractures)
    # List of lists of tuples, each sublist contains fragments (dim, tag) from particular fracture
    print('fractures_map = ', fractures_map)
    assert len(fractures_map) == len(disks)

    # set physical id and name of fractures
    fractures_phys_id = model.addPhysicalGroup(2, [id for dim, id in fractures])
    model.setPhysicalName(2, fractures_phys_id, "fractures")

    # define 3d volume and embed fractures into it
    box = factory.addBox(0, 0, 0, 1, 1, 1)

    factory.synchronize()
    # Embed the model entities of dimension 'dim', and tags 'tag', ...] in the other model entity which is given by
    # dimension (3) and tag (box).
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
    model.mesh.field.setNumber(2, "LcMin", max_el_size*0.03)
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




import attr
from typing import TypeVar, Union, Tuple, Optional
import numpy as np

from gmsh_api import gmsh
"""
Structure:
gmsh
gmsh.option
gmsh.model
gmsh.model.mesh
gmsh.model.geo
gmsh.model.occ
gmsh.view
gmsh.plugin
gmsh.logger
gmsh.fltk
gmsh.onelab
"""
#from math import *


@attr.s(auto_attribs=True)
class FractureData:
    r: float
    # Size of the fracture, laying in XY plane
    centre: np.array
    # location of the barycentre of the fracture
    rotation_axis: np.array
    # axis of rotation
    rotation_angle: float
    # angle of rotation around the axis (?? counterclockwise with axis pointing up)
    region: Union[str, int]
    # name or ID of the physical group
    aspect: float = 1
    # aspect ratio of the fracture, y_length : x_length, x_length == r

    @property
    def rx(self):
        return self.r

    @property
    def ry(self):
        return self.r * self.aspect



GMSHmodel = TypeVar('GMSHmodel', gmsh.model.occ, gmsh.model.geo)

class GMSHFactory:
    def __init__(self, gmsh_model, **kwargs):
        gmsh.initialize()
        print("GMSH initialized")
        self.model = gmsh_model
        gmsh.option.setNumber("General.Terminal", kwargs.get('verbose', False))

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

    def object(self, dim, tag):
        return GMSHobject(self, [(dim, tag)])

    #def objects(self, ):

    def rectangle(self, xy_sides=[1,1], center=[0,0,0]):
        xy_sides.append(0)
        corner = np.array(center)  - np.array(xy_sides) / 2
        return self.object(2, self.model.addRectangle(*corner.tolist(), *xy_sides[0:2]))

    def box(self, sides, center=[0,0,0] ):
        corner = np.array(center) - np.array(sides) / 2
        box_tag = self.model.addBox(*corner, *sides)
        return self.object(3, box_tag)

    def cylinder(self, r=1, axis=[0,0,1], center=[0,0,0] ):
        cylinder_tag = self.model.addCylinder(*center, *axis, r)
        return self.object(3, cylinder_tag)

    def synchronize(self):
        """
        Not clear which actions requires synchronization. Seems that it should be called after calculation of
        new shapes and before new shapes are added explicitely.
        """
        self.model.synchronize()

    def group(self, obj_list):
        if type(obj_list) is GMSHobject:
            return obj_list
        all_dim_tags = [ dim_tag
                            for obj in obj_list
                                for dim_tag in obj.dim_tags ]
        return GMSHobject(self, all_dim_tags)

    def make_rectangle(self, scale) -> int:
        # Vertices of the rectangle
        shifts = np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)])
        corners = shifts[:, :] * scale[None, :]
        point_tags = [self.model.addPoint(*corner) for corner in corners]
        lines = [self.model.addLine(point_tags[i-1], point_tags[i]) for i in range(4)]
        cl = self.model.addCurveLoop(lines)
        return self.model.addPlaneSurface([cl])


    def make_fractures(self, fractures, base_shape: 'GMSHobject'):
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        shapes = {}
        for fr in fractures:
            shape = base_shape.copy()
            shape = shape.scale([fr.rx, fr.ry, 1])\
                         .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle)\
                         .translate(fr.centre)
            shapes[fr.region] = shape

        # break fractures by their intersections
        # returns fragments and map orig_fracture_tag -> list of fracture fragment tags
        fr_dimtags = [shp.dim_tags[0] for shp in shapes.values()]
        _, fragments = self.model.fragment(fr_dimtags, [])
        fracture_fragments = {shape_tag: GMSHobject(self, fragment_list) for shape_tag, fragment_list in zip(shapes, fragments)}
        for tag, fr_frags in fracture_fragments.items():
            fr_frags.assign_physical_group(tag)
        return fracture_fragments

RegionType = Tuple[int, int, str]

@attr.s(auto_attribs=True)
class GMSHobject:
    factory: GMSHFactory
    dim_tags: int
    region: Optional[RegionType] = None

    def translate(self, vector):
        """"""
        self.factory.model.translate(self.dim_tags, *vector)
        return self

    def rotate(self, axis, angle, center=[0,0,0]):
        self.factory.model.rotate(self.dim_tags, *center, *axis, angle)
        return self

    def scale(self, scale_vector, center=[0,0,0]):
        self.factory.model.dilate(self.dim_tags, *center, *scale_vector)
        return self

    def copy(self) -> 'GMSHobject':
        return GMSHobject(self.factory, self.factory.model.copy(self.dim_tags))

    def get_boundary(self):
        return GMSHobject(self.factory, gmsh.model.getBoundary(self.dim_tags, combined=False, oriented=False))

    def get_boundary_combined(self):
        GMSHobject(self.factory, gmsh.model.getBoundary(self.dim_tags, combined=True, oriented=False))

    def common_dim(self, dim_tags=None):
        if dim_tags is None:
            dim_tags = self.dim_tags
        assert self.dim_tags
        dim = self.dim_tags[0][0]
        for d, tag in self.dim_tags:
            if d != dim:
                return None
        return dim

    def assign_pysical_group(self, tag):
        if type(tag) is str:
            id, name  = -1, tag
        else:
            assert type(tag) is int
            id, name = tag, None
        common_dim = self.common_dim()
        id = gmsh.model.addPhysicalGroup(common_dim,  [tag for dim, tag in self.dim_tags], tag=id)
        if name:
            gmsh.model.setPhysicalName(common_dim, id, name)
        self.region = (common_dim, id, name)
        return self

    def embed(self, lower_dim_objects):
        """
        Embedding is not managed at object level, in particular it does not produce
        a new object that is result of embedding. Lower dim objects can not poke out of
        the higher object, not clear if they can reach its boundary.
        Possibly it should be avoided and possibly not be part of GMSHobject.
        """
        all_dim_tags = self.factory.group(lower_dim_objects)
        lower_dim = all_dim_tags.common_dim()
        assert len(self.dim_tags) == 1
        higher_dim, higher_tag = self.dim_tags[0]
        assert  lower_dim < higher_dim
        gmsh.mesh.embed(lower_dim, [tag for dim, tag in all_dim_tags.dim_tags], higher_dim, higher_tag)

    def cut(self, tool_objects, remove_object=True, remove_tool=True):
        tool_objects = self.factory.group(tool_objects)
        print('cut:', self.dim_tags, tool_objects.dim_tags)
        new_tags = self.factory.model.cut(self.dim_tags, tool_objects.dim_tags, -1, remove_object, remove_tool)
        return GMSHobject(self.factory, new_tags)



def generate_mesh():
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

    factory = GMSHFactory(gmsh.model.occ)
    file_name = "box_wells"
    gmsh.model.add(file_name)
    #box_size = 2000
    #box = factory.box(3 * [box_size]) #.assign_pysical_group("box")
    #box1 = factory.box(3 * [box_size/2])

    box = gmsh.model.occ.addBox(-1000, -1000, -1000, 2000, 2000, 2000)
    #box1 = gmsh.model.occ.addCylinder(0, 0, -500, 0, 0, 1000, 500)

    rec1 = gmsh.model.occ.addRectangle(-800, -800, 0, 1600, 1600)
    rec2 = gmsh.model.occ.addRectangle(-1200, -1200, 0, 2400, 2400)
    rec1_dt = (2, rec1)
    rec2_dt = (2, rec2)
    gmsh.model.occ.rotate([rec2_dt], 0,0,0,  0,1,0, np.pi/2)
    rectangle, map = gmsh.model.occ.fragment([rec1_dt, rec2_dt], [])

    #box, box1 = box.dim_tags[0][1], box1.dim_tags[0][1]
    #print( box, box1)

    # well_shift = 500
    # well_diam = 10
    # well_length = 100
    # well_z_shift = -well_length/2
    # left_center = [-well_shift, 0, 0]
    # right_center = [well_shift, 0, 0]
    # left_well = factory.cylinder(well_diam, axis=[0,0,well_length])#\
    #                 #.translate([0,0,well_z_shift]).translate(left_center)
    #     #\
    #     #            .assign_pysical_group("left_well")
    # right_well = factory.cylinder(well_diam, axis=[0,0,well_length]) #\
    #                 #.translate([0, 0, well_z_shift]).translate(right_center)
    #     #\
    #     #            .assign_pysical_group("right_well")

    box = [(3,box)]
    #box, map = gmsh.model.occ.cut([(3,box)], [(3,box1)])
    #print('cut:', [(3, box)], [(3, box1)])


    box_copy = gmsh.model.occ.copy(box)
    dim_tags, map = gmsh.model.occ.intersect(rectangle, box_copy)
    gmsh.model.occ.synchronize()

    dim_tags_copy = gmsh.model.occ.copy(dim_tags)
    box_copy = gmsh.model.occ.copy(box)
    box_cut, map = gmsh.model.occ.fragment(box_copy, dim_tags_copy)
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()


    b = gmsh.model.addPhysicalGroup(3, [tag for dim, tag in box_cut])
    gmsh.model.setPhysicalName(3, b, "box")

    rect = gmsh.model.addPhysicalGroup(2, [tag for dim, tag in dim_tags])
    gmsh.model.setPhysicalName(2, rect, "rectangle")

    bc_tags = gmsh.model.getBoundary(dim_tags, combined=False, oriented=False)
    bc_nodes = gmsh.model.getBoundary(dim_tags, combined=False, oriented=False, recursive=True)
    bc_box_nodes = gmsh.model.getBoundary(box_cut, combined=False, oriented=False, recursive=True)
    bc_rect = gmsh.model.addPhysicalGroup(1, [tag for dim, tag in bc_tags])
    gmsh.model.setPhysicalName(1, bc_rect, ".rectangle")
    gmsh.model.occ.setMeshSize(bc_nodes, 50)




    #gmsh.model.mesh.embed(2, [tag for dim, tag in dim_tags], 3, box_copy[0][1])
    #factory.synchronize()



    model = gmsh.model

    # generate mesh, write to file and output number of entities that produced error
    factory.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 100)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 300)

    gmsh.write(file_name + '.brep')
    gmsh.write(file_name + '.geo')
    model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()
    bad_entities = model.mesh.getLastEntityError()
    print(bad_entities)
    gmsh.write(file_name + ".msh")
    gmsh.finalize()

    return len(bad_entities)












if __name__ == "__main__":
    generate_mesh()
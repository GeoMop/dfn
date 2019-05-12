
import attr
from typing import TypeVar, Union, Tuple, Optional, List
import numpy as np
import field
from collections import defaultdict

from gmsh_api import gmsh
"""
Structure:
gmsh
gmsh.option
gmsh.model
gmsh.model.mesh
gmsh.model.mesh.field
gmsh.model.geo
gmsh.model.geo,mesh
gmsh.model.occ
gmsh.view
gmsh.plugin
gmsh.graphics
gmsh.fltk
gmsh.onelab
gmsh.logger
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




@attr.s(auto_attribs=True)
class Region:
    dim: Optional[int]
    id: int
    name: str
    _boundary_region: 'Region' = None
    _max_reg_id = 99999

    @classmethod
    def get_region_id(cls):
        cls._max_reg_id += 1
        return cls._max_reg_id

    @classmethod
    def get(cls, name, dim=None):
        """
        Return a unique possibly uncomplete region.
        """
        return Region(dim, cls.get_region_id(), name)


    def get_boundary(self):
        b_reg = Region(self.dim - 1, self.get_region_id(), "." + self.name)
        return b_reg


    def complete(self, dim):
        """
        Check dimension match and complete the region.
        """
        if self.dim is None:
            self.dim = dim
        else:
            assert self.dim == dim, (self.dim, dim)
        return self

    def set_unique_name(self, idx):
        self.name = "{}_{}".format(self.name, idx)

Region.default_region = [Region.get("default_{}d".format(dim), dim) for dim in range(4)]






GMSHmodel = TypeVar('GMSHmodel', gmsh.model.occ, gmsh.model.geo)
DimTag = Tuple[int, int]

class GMSHFactory:
    def __init__(self, gmsh_model, model_name, **kwargs):
        self.model_name = model_name
        gmsh.initialize()
        gmsh.model.add(model_name)
        print("GMSH initialized")
        self.model = gmsh_model
        gmsh.option.setNumber("General.Terminal", kwargs.get('verbose', False))

        # set options
        gmsh.option.setNumber("Geometry.Tolerance", 0)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 0)
        gmsh.option.setNumber("Geometry.MatchMeshTolerance", 0)
        gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-12)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", max_el_size*0.01)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_el_size)
        # gmsh.option.setNumber("Mesh.MinimumCurvePoints", 5)

    def object(self, dim, tag):
        return GMSHobject(self, [(dim, tag)], [Region.default_region[dim]])

    #def objects(self, ):

    def rectangle(self, xy_sides=[1,1], center=[0,0,0]):
        xy_sides.append(0)
        corner = np.array(center)  - np.array(xy_sides) / 2
        return self.object(2, self.model.addRectangle(*corner.tolist(), *xy_sides[0:2]), )

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
        if len(obj_list) == 1:
            return obj_list[0]
        all_dim_tags = [ dim_tag
                            for obj in obj_list
                                for dim_tag in obj.dim_tags ]
        regions = [ reg
                            for obj in obj_list
                                for reg in obj.regions ]
        return GMSHobject(self, all_dim_tags, regions)

    def make_rectangle(self, scale) -> int:
        # Vertices of the rectangle
        shifts = np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)])
        corners = shifts[:, :] * scale[None, :]
        point_tags = [self.model.addPoint(*corner) for corner in corners]
        lines = [self.model.addLine(point_tags[i-1], point_tags[i]) for i in range(4)]
        cl = self.model.addCurveLoop(lines)
        return self.model.addPlaneSurface([cl])


    def make_fractures(self, fractures, base_shape: 'GMSHobject', set_name="fractures"):
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        shapes = []
        for fr in fractures:
            fr.region = Region.get(set_name)
            shape = base_shape.copy()
            shape = shape.scale([fr.rx, fr.ry, 1])\
                         .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle)\
                         .translate(fr.centre)\
                         .set_region(fr.region)
            shapes.append(shape)

        fracture_fragments = self.group(shapes).fragment([])
        return fracture_fragments

    def _assign_physical_groups(self, obj):
        reg_to_tags = {}
        reg_names = defaultdict(set)

        # collect tags of regions
        for reg, dimtag in obj.regdimtag():
            dim, tag = dimtag
            reg.complete(dim)
            reg_to_tags.setdefault(reg.id, (reg, []) )
            reg_to_tags[reg.id][1].append(tag)
            reg_names[reg.name].add(reg.id)

        # make used region names unique
        for id_set in reg_names.values():
            if len(id_set) > 1:
                for i, id in enumerate(sorted(id_set)):
                    reg_to_tags[id][0].set_unique_name(i)

        # set physical groups
        for reg, tags in reg_to_tags.values():
            reg._gmsh_id = gmsh.model.addPhysicalGroup(reg.dim, tags, tag=-1)
            gmsh.model.setPhysicalName(reg.dim, reg._gmsh_id, reg.name)

    def make_mesh(self, objects: List['GMSHobject']) -> None:
        """
        Mesh given objects.
        :param objects:
        :return:
        """

        self._assign_physical_groups(self.group(objects))

        self.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()
        bad_entities = gmsh.model.mesh.getLastEntityError()
        print("Bad entities:", bad_entities)


    def write_brep(self, filename=None):
        if filename is None:
            filename = self.model_name
        gmsh.write(filename + '.brep')


    def write_mesh(self, filename=None):
        if filename is None:
            filename = self.model_name
        gmsh.write(filename + '.msh')

    def __del__(self):
        gmsh.finalize()



class GMSHobject:
    def __init__(self, factory: GMSHFactory, dim_tags:List[DimTag], regions: List[Region]) -> None:
        self.factory = factory
        self.dim_tags = dim_tags
        if len(regions)==1:
            self.regions = [regions[0] for _ in dim_tags]
        else:
            assert(len(regions) == len(dim_tags))
            self.regions = regions


    def set_region(self, region):
        if isinstance(region, str):
            region = Region.get(region)
        self.regions = [region.complete(dim) for dim, tag in self.dim_tags]
        return self


    def translate(self, vector):
        self.factory.model.translate(self.dim_tags, *vector)
        return self


    def rotate(self, axis, angle, center=[0,0,0]):
        self.factory.model.rotate(self.dim_tags, *center, *axis, angle)
        return self


    def scale(self, scale_vector, center=[0,0,0]):
        self.factory.model.dilate(self.dim_tags, *center, *scale_vector)
        return self


    def copy(self) -> 'GMSHobject':
        return GMSHobject(self.factory, self.factory.model.copy(self.dim_tags), self.regions)


    def get_boundary(self, combined = False):
        """
        derive_regions - if combined True, make derived boundary regions, other wise default regions are used
        combined=True ... omit fracture intersetions (boundary of combined object)
        combined=False ... give also intersections (boundary of indiviual objects)

        TODO: some support for oriented=True (returns signed tag according to its orientation)
              recursive=True (seems to provide boundary nodes)
        """
        dimtags = gmsh.model.getBoundary(self.dim_tags, combined=combined, oriented=False)
        regions = [Region.default_region[dim] for dim, tag in dimtags]
        return GMSHobject(self.factory, dimtags, regions)


    def common_dim(self, dim_tags=None):
        if dim_tags is None:
            dim_tags = self.dim_tags
        assert dim_tags
        dim = dim_tags[0][0]
        for d, tag in dim_tags:
            if d != dim:
                return None
        return dim

    def regdimtag(self):
        assert len(self.regions) == len(self.dim_tags)
        return zip(self.regions, self.dim_tags)

    # def assign_physical_group(self, tag):
    #     if type(tag) is str:
    #         id, name  = -1, tag
    #     else:
    #         assert type(tag) is int
    #         id, name = tag, None
    #     common_dim = self.common_dim()
    #     id = gmsh.model.addPhysicalGroup(common_dim,  [tag for dim, tag in self.dim_tags], tag=id)
    #     if name:
    #         gmsh.model.setPhysicalName(common_dim, id, name)
    #     self.regions = [(common_dim, id, name) for _ in self.dim_tags]
    #     return self

    # def _set_regions(self, regions):
    #     """
    #     Internal method to assign regions to fresh dimtags.
    #     :param regions:
    #     :return:
    #     """
    #     self.regions = []
    #     for reg, dimtag in zip(regions, self.dim_tags):
    #         dim, tag = dimtag
    #         if reg:
    #             reg_dim, reg_id, reg_name = reg
    #             assert reg_dim == dim
    #             gmsh.model.addPhysicalGroup(dim, [tag], tag=reg_id)
    #         self.regions.append(reg)

    # def embed(self, lower_dim_objects):
    #     """
    #     Embedding is not managed at object level, in particular it does not produce
    #     a new object that is result of embedding. Lower dim objects can not poke out of
    #     the higher object, not clear if they can reach its boundary.
    #     Possibly it should be avoided and possibly not be part of GMSHobject.
    #     """
    #     all_dim_tags = self.factory.group(lower_dim_objects)
    #     lower_dim = all_dim_tags.common_dim()
    #     assert len(self.dim_tags) == 1
    #     higher_dim, higher_tag = self.dim_tags[0]
    #     assert  lower_dim < higher_dim
    #     gmsh.mesh.embed(lower_dim, [tag for dim, tag in all_dim_tags.dim_tags], higher_dim, higher_tag)

    def _apply_operation(self, tool_objects, operation):
        tool_objects = self.factory.group(tool_objects)
        new_tags, old_tags_map = operation(self.dim_tags, tool_objects.dim_tags)

        # assign regions
        assert len(self.regions) == len(self.dim_tags), (len(self.regions),len(self.dim_tags))
        old_tags_objects = [GMSHobject(self.factory, new_subtags, [reg])
                            for reg, new_subtags in zip(self.regions, old_tags_map[:len(self.dim_tags)])]
        new_obj = self.factory.group(old_tags_objects)

        # store auxiliary information
        # TODO: remove, should not be necessary
        # new_obj._previous_obj = self
        # new_obj._previous_dim_tags = self.dim_tags
        # new_obj._previous_map = old_tags_map

        # invalidate original objects
        self.invalidate()
        tool_objects.invalidate()
        return new_obj


    def cut(self, tool_objects):
        return self._apply_operation(tool_objects, self.factory.model.cut)


    def intersect(self, tool_objects):
        return self._apply_operation(tool_objects, self.factory.model.intersect)


    def fragment(self, tool_objects):
        return self._apply_operation(tool_objects, self.factory.model.fragment)


    def invalidate(self):
        self.factory = None
        self.dim_tags = None
        self.regions = None



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

    factory = GMSHFactory(gmsh.model.occ, "three_frac_symmetric")
    box_size = 2000
    box = factory.box(3 * [box_size]).set_region("box")

    well_shift = 500
    well_radius = 50
    well_length = 3000
    well_z_shift = -well_length/2
    left_center = [-well_shift, 0, 0]
    right_center = [well_shift, 0, 0]
    left_well = factory.cylinder(well_radius, axis=[0, 0, well_length])\
                    .translate([0,0,well_z_shift]).translate(left_center)\
                    .set_region("left_well")
    right_well = factory.cylinder(well_radius, axis=[0, 0, well_length])\
                    .translate([0, 0, well_z_shift]).translate(right_center)\
                    .set_region("right_well")

    fractures = [
        FractureData(r, centre, axis, angle, tag) for r, centre, axis, angle, tag in
        [
            (1300, left_center,  [0, 1, 0], np.pi/6, 'left_fr'),
            (1300, right_center, [0, 1, 0], np.pi/6, 'right_fr'),
            (900, [0,0,0],      [0, 1, 0], np.pi/2, 'center_fr')
        ]]
    all_fractures = factory.make_fractures(fractures, factory.rectangle())



    box_without_wells = box.cut([left_well, right_well])
    cut_fractures = all_fractures.intersect(box_without_wells.copy())
    box_without_wells = box_without_wells.fragment(cut_fractures.copy())

    #left_well.get_boundary()

    # define 3d volume and embed fractures into it
    factory.synchronize()
    # Embed the model entities of dimension 'dim', and tags 'tag' in the other model entity which is given by
    # dimension (3) and tag (box)
    #box.embed(fr_fractures.values())
    #model.mesh.embed(2, [tag for dim, tag in fractures], 3, box)

    # define physical id of volume and its boundaries
    # box_bdry = model.getBoundary([(3,box)], True)
    # box_bdry_left_phys_id = model.addPhysicalGroup(2, [box_bdry[0][1]], -1)
    # box_bdry_right_phys_id = model.addPhysicalGroup(2, [box_bdry[1][1]], -1)
    # model.setPhysicalName(2, box_bdry_left_phys_id, ".left")
    # model.setPhysicalName(2, box_bdry_right_phys_id, ".right")

    # Define fields for mesh refinement
    # Compute the distance from curves, each curve is replaced by NNodesByEdge equidistant nodes
    # and the distance from those nodes is computed.
    min_el_size = well_radius
    fracture_el_size = box_size / 20
    fracture_size = 1000
    max_el_size = box_size / 5

    frac_bdry = cut_fractures.get_boundary()
    distance_field = field.distance_edges([tag for dm, tag in frac_bdry.dim_tags if dm == 1])
    threshold = field.threshold(distance_field, (0, fracture_el_size), (fracture_size, max_el_size))
    field.set_mesh_step_field(threshold)

    factory.write_brep()
    factory.make_mesh([box_without_wells, cut_fractures])
    factory.write_mesh()
















if __name__ == "__main__":
    generate_mesh()
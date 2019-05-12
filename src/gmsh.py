import attr
import enum
from typing import TypeVar, Tuple, Optional, List
from collections import defaultdict
from gmsh_api import gmsh
import numpy as np
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

class Options:
    def __init__(self, gmsh_class):
        self.gmsh_class
        self.names_map = {}

    def add(self, gmsh_name, default):
        if isinstance(default, type):
            option_type = default
        else:
            option_type = type(default)
        self.names_map[gmsh_name] = (gmsh_name, option_type)
        self.__setattr__(gmsh_name, default)

    # TODO: set option type in 'add' through default value ar type
    def __setattr__(self, key, value):
        assert key in self.names_map
        gmsh_name, option_type = self.names_map[key]
        assert type(value) is option_type
        full_name = self.prefix + gmsh_name
        if isinstance(value, (int, float, bool)):
            gmsh.option.setNumber(full_name, value)
        elif isinstance(value, str):
            gmsh.option.setString(full_name, value)
        else:
            raise ValueError("Unsupported value type {} for GMSH option type.")


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


class Geometry:
    """
    Interface to gmsh_api.
    Single instance is allowed.
    TODO: add documented support of geometry and meshing parameters.
    """
    _have_instance = False

    def __init__(self, model_str, model_name, **kwargs):
        if model_str == 'occ':
            self.model = gmsh.model.occ
        elif model_str == 'geo':
            self.model = gmsh.model.geo
        else:
            raise ValueError

        if self._have_instance:
            raise Exception("Only single instance of GMSHFactory is allowed.")
        else:
            self._have_instance = False

        self.model_name = model_name
        gmsh.initialize()
        gmsh.model.add(model_name)
        print("GMSH initialized")

        self._region_names = {}


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
        return ObjectSet(self, [(dim, tag)], [Region.default_region[dim]])

    # def objects(self, ):

    def rectangle(self, xy_sides=[1, 1], center=[0, 0, 0]):
        xy_sides.append(0)
        corner = np.array(center) - np.array(xy_sides) / 2
        return self.object(2, self.model.addRectangle(*corner.tolist(), *xy_sides[0:2]), )

    def box(self, sides, center=[0, 0, 0]):
        corner = np.array(center) - np.array(sides) / 2
        box_tag = self.model.addBox(*corner, *sides)
        return self.object(3, box_tag)

    def cylinder(self, r=1, axis=[0, 0, 1], center=[0, 0, 0]):
        cylinder_tag = self.model.addCylinder(*center, *axis, r)
        return self.object(3, cylinder_tag)

    def synchronize(self):
        """
        Not clear which actions requires synchronization. Seems that it should be called after calculation of
        new shapes and before new shapes are added explicitely.
        """
        self.model.synchronize()

    def group(self, obj_list):
        if type(obj_list) is ObjectSet:
            return obj_list
        if len(obj_list) == 1:
            return obj_list[0]
        all_dim_tags = [dim_tag
                        for obj in obj_list
                        for dim_tag in obj.dim_tags]
        regions = [reg
                   for obj in obj_list
                   for reg in obj.regions]
        return ObjectSet(self, all_dim_tags, regions)

    def make_rectangle(self, scale) -> int:
        # Vertices of the rectangle
        shifts = np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)])
        corners = shifts[:, :] * scale[None, :]
        point_tags = [self.model.addPoint(*corner) for corner in corners]
        lines = [self.model.addLine(point_tags[i - 1], point_tags[i]) for i in range(4)]
        cl = self.model.addCurveLoop(lines)
        return self.model.addPlaneSurface([cl])

    def make_fractures(self, fractures, base_shape: 'ObjectSet', set_name="fractures"):
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        shapes = []
        for fr in fractures:
            fr.region = Region.get(set_name)
            shape = base_shape.copy()
            shape = shape.scale([fr.rx, fr.ry, 1]) \
                .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
                .translate(fr.centre) \
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
            reg_to_tags.setdefault(reg.id, (reg, []))
            reg_to_tags[reg.id][1].append(tag)
            reg_names[reg.name].add(reg.id)

        # make used region names unique
        for id_set in reg_names.values():
            if len(id_set) > 1:
                for i, id in enumerate(sorted(id_set)

                                       ):
                    reg_to_tags[id][0].set_unique_name(i)

        # set physical groups
        for reg, tags in reg_to_tags.values():
            reg._gmsh_id = gmsh.model.addPhysicalGroup(reg.dim, tags, tag=-1)
            gmsh.model.setPhysicalName(reg.dim, reg._gmsh_id, reg.name)

    def make_mesh(self, objects: List['ObjectSet']) -> None:
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

    def get_region_name(self, name):
        region = self._region_names.get(name, Region.get(name))
        self._region_names[name] = region
        return region

    def __del__(self):
        gmsh.finalize()




class Mesh:
    class Algo2d(enum.IntEnum):
        MeshAdapt = 1
        Automatic = 2
        Delaunay = 5
        FrontalDelaunay = 6
        BAMG = 7
        FrontalDelaunayQuads = 8
        ParalelogramsPacking = 9

    class Algo3d(enum.IntEnum):
        Delaunay = 1
        Frontal = 4
        MMG3D = 7
        RTree = 9
        HXT = 10



    def __init__(self):
        self.option = Options('Mesh.')

        self.option.add('Algorithm', self.Algo2d.Automatic)
        # 2D mesh algorithm
        self.option.add('Algorithm3D', self.Algo3d.Delaunay)
        # 3D mesh algorithm
        self.option.add('ToleranceInitialDelaunay', 1e-12)
        # Tolerance for initial 3D Delaunay mesher
        self.option.add('CharacteristicLengthFromPoints', True)
        # Compute mesh element sizes from values given at geometry points
        self.option.add('CharacteristicLengthFromCurvature', True)
        # Automatically compute mesh element sizes from curvature (experimental)
        self.option.add('CharacteristicLengthExtendFromBoundary', int)
        # Extend computation of mesh element sizes from the boundaries into the interior
        # (for 3D Delaunay, use 1: longest or 2: shortest surface edge length)
        self.option.add('CharacteristicLengthMin', float)
        # Minimum mesh element size
        self.option.add('CharacteristicLengthMax', float)
        # Maximum mesh element size
        self.option.add('CharacteristicLengthFactor', float)
        # Factor applied to all mesh element sizes
        self.option.add('MinimumCurvePoints', 6)
        # Minimum number of points used to mesh a (non-straight) curve


class ObjectSet:
    def __init__(self, factory: Geometry, dim_tags: List[DimTag], regions: List[Region]) -> None:
        self.factory = factory
        self.dim_tags = dim_tags
        if len(regions) == 1:
            self.regions = [regions[0] for _ in dim_tags]
        else:
            assert (len(regions) == len(dim_tags))
            self.regions = regions

    def set_region(self, region):
        """
        Set given region to all self.dimtags.
        Create a new region if just a string is given.
        :return: self
        """
        if isinstance(region, str):
            region = self.factory.get_region_name(region)
        self.regions = [region.complete(dim) for dim, tag in self.dim_tags]
        return self

    def prefix_regions(self, prefix: str) -> None:
        """
        Set given region to all self.dimtags.
        Create a new region if just a string is given.
        :return: self
        """
        regions = []
        for region in self.regions:
            new_name = prefix + region.name
            new_region = Region.get(new_name)
            regions.append(new_region)
        self.regions = regions
        return self

    def translate(self, vector):
        self.factory.model.translate(self.dim_tags, *vector)
        return self

    def rotate(self, axis, angle, center=[0, 0, 0]):
        self.factory.model.rotate(self.dim_tags, *center, *axis, angle)
        return self

    def scale(self, scale_vector, center=[0, 0, 0]):
        self.factory.model.dilate(self.dim_tags, *center, *scale_vector)
        return self

    def copy(self) -> 'ObjectSet':
        return ObjectSet(self.factory, self.factory.model.copy(self.dim_tags), self.regions)

    def get_boundary(self, combined=False):
        """
        derive_regions - if combined True, make derived boundary regions, other wise default regions are used
        combined=True ... omit fracture intersetions (boundary of combined object)
        combined=False ... give also intersections (boundary of indiviual objects)

        TODO: some support for oriented=True (returns signed tag according to its orientation)
              recursive=True (seems to provide boundary nodes)
        """
        dimtags = gmsh.model.getBoundary(self.dim_tags, combined=combined, oriented=False)
        regions = [Region.default_region[dim] for dim, tag in dimtags]
        return ObjectSet(self.factory, dimtags, regions)

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
        assert len(self.regions) == len(self.dim_tags), (len(self.regions), len(self.dim_tags))
        old_tags_objects = [ObjectSet(self.factory, new_subtags, [reg])
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

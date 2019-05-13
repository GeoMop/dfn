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

gmsh_api, issues:
- terrible interface to fields
- get_boundary not part of geometry model (occ/geo), need lot of synchronizations
- all existing dimtags are meshed not only those with assigned physical groups
- physical groups are not assigned to the objects, but groups are formed from objects, 
  possible error having single object in more physical groups
- Mesh.Format option - doc do not support gmsh 2.0 format
  gmsh.write function seems to ignore the format and use extensions which are not documented
"""


class MeshFormat(enum.IntEnum):
    msh = 1
    unv = 2
    msh2 = 3    # only for extension, code unknown
    auto = 10
    vtk = 16
    vrml = 19
    mail = 21
    pos_stat = 26
    stl = 27
    p3d = 28
    mesh = 30
    bdf = 31
    cgns = 32
    med = 33
    diff = 34
    ir3 = 38
    inp = 39
    ply2 = 40
    celum = 41
    su2 = 42
    tochnog = 47
    neu = 49
    matlab = 50


class Algorithm2d(enum.IntEnum):
    MeshAdapt = 1
    Automatic = 2
    Delaunay = 5
    FrontalDelaunay = 6
    BAMG = 7
    FrontalDelaunayQuads = 8
    ParalelogramsPacking = 9


class Algorithm3d(enum.IntEnum):
    Delaunay = 1
    Frontal = 4
    MMG3D = 7
    RTree = 9
    HXT = 10


class Options:
    """
    Auxiliary class to set GMSH options as a class attributes:

    my_options.option_name = value

    Valid option names are defined in the constructor of the derived class as
    their own attributes. After that call of 'finish_init' will:
    1. collect existing attributes
    2. set appropriate GMSH options to default values
    3. set __setattr__ method so that furhter assignements to attribute are translated to
       setting the GMSH option.

    """
    def __init__(self, prefix):
        self.prefix = prefix
        # Prefix of the GMSH option, e.g. 'Mesh.'
        self.names_map = {}
        # Dictionary of valid options: option_name -> type
        self.__setattr__ = self.init_setattr


    def finish_init(self):
        self.__setattr__ = self.instance_setattr


    def _add(self, gmsh_name, default):
        """
        Define new option with name 'gmsh_name'.
        :param default: either initial value or just type (enum, bool, float, int, str)

        If default value is provided it is passed to GMSH immediately.
        """
        if isinstance(default, type):
            option_type = default
        else:
            option_type = type(default)
            self.instance_setattr(gmsh_name, default)
        self.names_map[gmsh_name] = (gmsh_name, option_type)


    def init_setattr(self, key, value):
        """
        Syntactic sugar for _add.
        """
        self._add(key, value)


    def instance_setattr(self, key, value):
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
        self._need_synchronize = False


        gmsh.option.setNumber("General.Terminal", kwargs.get('verbose', False))

        # set options
        gmsh.option.setNumber("Geometry.Tolerance", 0)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 0)
        gmsh.option.setNumber("Geometry.MatchMeshTolerance", 0)



    def get_region_name(self, name):
        region = self._region_names.get(name, Region.get(name))
        self._region_names[name] = region
        return region


    def object(self, dim, tag):
        return ObjectSet(self, [(dim, tag)], [Region.default_region[dim]])

    # def objects(self, ):

    def rectangle(self, xy_sides=[1, 1], center=[0, 0, 0]):
        xy_sides.append(0)
        corner = np.array(center) - np.array(xy_sides) / 2
        rec_tag = self.model.addRectangle(*corner.tolist(), *xy_sides[0:2])
        self._need_synchronize = True
        return self.object(2, rec_tag)

    def box(self, sides, center=[0, 0, 0]):
        corner = np.array(center) - np.array(sides) / 2
        box_tag = self.model.addBox(*corner, *sides)
        self._need_synchronize = True
        return self.object(3, box_tag)

    def cylinder(self, r=1, axis=[0, 0, 1], center=[0, 0, 0]):
        cylinder_tag = self.model.addCylinder(*center, *axis, r)
        self._need_synchronize = True
        return self.object(3, cylinder_tag)

    def synchronize(self):
        """
        Not clear which actions requires synchronization. Seems that it should be called after calculation of
        new shapes and before new shapes are added explicitely.
        """
        if self._need_synchronize:
            self.model.synchronize()
            self._need_synchronize = False

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
        self._need_synchronize = True
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
        self.synchronize()
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

    def make_mesh(self, objects: List['ObjectSet'], dim=3) -> None:
        """
        Generate mesh for given objects.
        :param dim: Set highest dimension to mesh.
        """

        self._assign_physical_groups(self.group(objects))
        self.synchronize()

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()
        bad_entities = gmsh.model.mesh.getLastEntityError()
        if bad_entities:
            print("Bad entities:", bad_entities)



    def write_brep(self, filename=None):
        if filename is None:
            filename = self.model_name
        gmsh.write(filename + '.brep')



    def write_mesh(self, filename: Optional[str] = None, format:MeshFormat = MeshFormat.auto) -> None:
        """
        Write a mesh generated by 'make_mesh' to the file 'filename'.
        Format is given by extension (see MeshFormat for supported formats)
        If 'filename' is not provided it is determined by the model name given in constructor.
        In such case the format is given by the 'format' parameter.
        """
        if filename is None:
            gmsh.option.setNumber('Mesh.Format', format)
            extension = format.name
            filename = "{}.{}".format(self.model_name, extension)
        gmsh.write(filename)


    def show(self):
        gmsh.fltk.run()




    def __del__(self):
        gmsh.finalize()




class MeshOptions(Options):

    def __init__(self):
        super().__init__('Mesh.')

        self.Algorithm = Algorithm2d.Automatic
        # 2D mesh algorithm
        self.Algorithm3D = Algorithm3d.Delaunay
        # 3D mesh algorithm
        self.ToleranceInitialDelaunay = 1e-12
        # Tolerance for initial 3D Delaunay mesher
        self.CharacteristicLengthFromPoints = True
        # Compute mesh element sizes from values given at geometry points
        self.CharacteristicLengthFromCurvature = True
        # Automatically compute mesh element sizes from curvature (experimental)
        self.CharacteristicLengthExtendFromBoundary = int
        # Extend computation of mesh element sizes from the boundaries into the interior
        # (for 3D Delaunay, use 1: longest or 2: shortest surface edge length)
        self.CharacteristicLengthMin = float
        # Minimum mesh element size
        self.CharacteristicLengthMax = float
        # Maximum mesh element size
        self.CharacteristicLengthFactor = float
        # Factor applied to all mesh element sizes
        self.MinimumCurvePoints = 6
        # Minimum number of points used to mesh a (non-straight) curve
        self.finish_init()

class BoolOperationError(Exception):
    pass

class GetBoundaryError(Exception):
    pass

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
        self.factory._need_synchronize = True
        return self

    def rotate(self, axis, angle, center=[0, 0, 0]):
        self.factory.model.rotate(self.dim_tags, *center, *axis, angle)
        self.factory._need_synchronize = True
        return self

    def scale(self, scale_vector, center=[0, 0, 0]):
        self.factory.model.dilate(self.dim_tags, *center, *scale_vector)
        self.factory._need_synchronize = True
        return self

    def copy(self) -> 'ObjectSet':
        copy_tags = self.factory.model.copy(self.dim_tags)
        self.factory._need_synchronize = True
        return ObjectSet(self.factory, copy_tags, self.regions)

    def get_boundary(self, combined=False):
        """
        Get the boundary of the model entities dimTags.
        Return in outDimTags the boundary of the individual entities
        (if combined is false) or the boundary of the combined geometrical shape
        formed by all input entities (if combined is true).
        Return tags multiplied by the sign of the boundary entity if oriented is true.
        Apply the boundary operator recursively down to dimension 0 (i.e. to points) if recursive is true.

        derive_regions - if combined True, make derived boundary regions, other wise default regions are used
        combined=True ... omit fracture intersetions (boundary of combined object)
        combined=False ... give also intersections (boundary of indiviual objects)

        TODO: some support for oriented=True (returns signed tag according to its orientation)
              recursive=True (seems to provide boundary nodes)
        """
        self.factory.synchronize()
        try:
            dimtags = gmsh.model.getBoundary(self.dim_tags, combined=combined, oriented=False)
        except ValueError as err :
            message = "\nobj dimtags: {}".format(str(self.dim_tags[:10]))
            raise GetBoundaryError(message) from err
        regions = [Region.default_region[dim] for dim, tag in dimtags]
        return ObjectSet(self.factory, dimtags, regions)

    def split_by_region(self):
        """
        Split objects in ObjectSet into ObjectSets one per region.
        :return: list of ObjectSets
        """
        reg_to_tags = {}
        # collect tags of regions
        for reg, dimtag in self.regdimtag():
            dim, tag = dimtag
            reg.complete(dim)
            reg_to_tags.setdefault(reg.id, (reg, []))
            reg_to_tags[reg.id][1].append(dimtag)
        reg_sets = [ObjectSet(self.factory, dimtags, [reg]) for reg, dimtags in reg_to_tags.values()]
        return reg_sets


    def get_boundary_per_region(self):
        reg_sets = self.split_by_region()
        b_sets = []
        for rset in reg_sets:
            reg = rset.regions[0]
            b_reg = reg.get_boundary()
            boundary = rset.get_boundary(combined=True).set_region(b_reg)
            b_sets.append(boundary)
        return b_sets

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
        try:
            new_tags, old_tags_map = operation(self.dim_tags, tool_objects.dim_tags, removeObject=True, removeTool=True)
        except ValueError as err :
            message = "\nobj dimtags: {}\ntool dimtags: {}".format(str(self.dim_tags[:10]), str(tool_objects.dim_tags[:10]))
            raise BoolOperationError(message) from err

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
        self.factory._need_synchronize = True
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

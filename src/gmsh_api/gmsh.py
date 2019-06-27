import attr
from typing import TypeVar, Tuple, Optional, List
import gmsh
import numpy as np
import itertools
from collections import defaultdict


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
- terrible interface to fields (resolved by field.py)

- get_boundary not part of geometry model (occ/geo), need lot of synchronizations
  (resolved by automatic synchronizations)
- all existing dimtags are meshed not only those with assigned physical groups
  (resolved by a step before meshing that removes all objects without associated physical group)
- physical groups are not assigned to the objects, but groups are formed from objects, 
  possible error having single object in more physical groups
  (Not sure what happens in the case of two groups for single object, but in general the concept is a valid option.)
- Mesh.Format option - doc do not support gmsh 2.0 format
  gmsh.write function seems to ignore the format and use extensions which are not documented
  (Resolved. Version can be set by different option,)
- gmsh.model.occ.setMeshSize - seems have no effect, in particular in combination with getBoundary
  (Confirmed, replaced by similar function in other module)
- no constant field
  (Resolved by the shpere field)
- gmsh.model.occ.removeAllDuplicates ... doesn't work
  (No sure, it works at least partialy.) 
- seems that occ.copy() doesn't preserve boundaries, so boundary dim tags are copied twice
  (It does exactly what it is asked for just copy the given shapes)
(Problem resolved by introduction of select_by_intersection)
"""

class BoolOperationError(Exception):
    pass

class GetBoundaryError(Exception):
    pass



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


# Initialize class attribute
Region.default_region = [Region.get("default_{}d".format(dim), dim) for dim in range(4)]
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


DimTag = Tuple[int, int]





class GeometryOCC:
    """
    Interface to gmsh_api.
    Single instance is allowed.
    TODO: add documented support of geometry and meshing parameters.
    """
    _have_instance = False


    # def addPoint(self, x, y, z, size):
    #     return self.object(0, self.model.addPoint(x, y, z, size))
    #
    # def addLine(self, start, end):
    #     return self.object(1, self.model.addLine(start.point(), end.point()))
    #
    # def addCircleArc(self, start, center, end):
    #     return self.object(1, self.model.addCircleArc(start.point(), center.point(), end.point()))
    #
    # def addEllipseArc(self, start, center, end):
    #     return self.object(1, self.model.addEllipseArc(start.point(), center.point(), end.point()))
    #
    # def addSpline(self, points):
    #     points = [pt_obj.point() for pt_obj in points]
    #     return self.object(1, self.model.addSpline(points))
    #
    # def addBSpline(self, points):
    #     points = [pt_obj.point() for pt_obj in points]
    #     return self.object(1, self.model.addBSpline(points))
    #
    # def addBezier(self, points):
    #     points = [pt_obj.point() for pt_obj in points]
    #     return self.object(1, self.model.addBezier(points))
    #
    # def addCurveLoop(self):
    #     pass
    #
    # def addPlaneSurface(self):
    #     pass
    #
    # addSurfaceFilling
    #
    # addSurfaceLoop
    #
    # addVolume
    #
    # extrude
    #
    # revolve
    #
    #
    # twist
    #
    # translate
    #
    # rotate
    # dilate
    #
    # symmetrize
    #
    # copy
    #
    # remove
    #
    # removeAllDuplicates
    #
    # synchronize


    def __init__(self, model_name, model_str='occ', **kwargs):
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

    def reinit(self):
        gmsh.clear()

    def get_region_name(self, name):
        region = self._region_names.get(name, Region.get(name))
        self._region_names[name] = region
        return region

    def object(self, dim, tag):
        return ObjectSet(self, [(dim, tag)], [Region.default_region[dim]])

    # def objects(self, ):

    def make_simplex(self, dim=3):
        """
        Make reference simplex
        TODO: use own methods for construction of geometries (combine with BSplines lib.)
        :return:
        """
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        lines = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
        faces = [(0, 1, 2), (0, 3, 4), (2, 3, 5), (1, 4, 5)]
        if dim == 0:
            res = self.model.addPoint(*points[0])
        elif dim == 1:
            point_ids = [self.model.addPoint(*p) for p in points[:2]]
            res = self.model.addLine(*point_ids)
        elif dim == 2:
            point_ids = [self.model.addPoint(*p) for p in points[:3]]
            line_ids = [self.model.addLine(*[point_ids[p] for p in l]) for l in lines[:3]]
            loop = self.model.addCurveLoop(line_ids)
            res = self.model.addPlaneSurface([loop])
        elif dim == 3:
            point_ids = [self.model.addPoint(*p) for p in points[:4]]
            line_ids = [self.model.addLine(*[point_ids[p] for p in l]) for l in lines[:6]]
            loop_ids = [self.model.addCurveLoop([line_ids[l] for l in f]) for f in faces[:4]]
            face_ids = [self.model.addPlaneSurface([loop]) for loop in loop_ids]
            surf_loop = self.model.addSurfaceLoop(face_ids)
            res = self.model.addVolume([surf_loop])
        self._need_synchronize = True
        return self.object(dim, res)

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
        new shapes and before new shapes are added explicitly.
        """
        if self._need_synchronize:
            self.model.synchronize()
            self._need_synchronize = False

    def group(self, *obj_list: 'ObjectSet') -> 'ObjectSet':
        """
        Group any number of ObjectSets into a single one.
        :param obj_list:
        :return:
        """
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

    def fragment(self, *object_sets: 'ObjectSet') -> List['ObjectSet']:
        """
        Fragment given objects mutually return list of fragmented objects.
        :param object_sets:
        :return:
        """
        cumulsizes = list(itertools.accumulate((o.size for o in object_sets)))
        all_dimtags = list(itertools.chain(*[o.dim_tags for o in object_sets]))

        try:
            new_tags, tags_map = self.model.fragment(all_dimtags, [], removeObject=True, removeTool=True)
        except ValueError as err:
            message = "\nall dimtags: {}, ...".format(str(all_dimtags[:20]))
            raise BoolOperationError(message) from err

        # assign regions
        new_sets = []
        begin = 0
        assert cumulsizes[-1] == len(tags_map), str(tags_map[cumulsizes[-1]:])
        for o, end in zip(object_sets, cumulsizes):
            dim_tag_map = tags_map[begin:end]
            newset = [ObjectSet(self, new_subtags, [reg])
                        for reg, new_subtags in zip(o.regions, dim_tag_map)]
            newset = self.group(*newset)
            new_sets.append(newset)
            begin = end
            o.invalidate()

        self._need_synchronize = True
        return new_sets

    def make_fractures(self, fractures, base_shape: 'ObjectSet'):
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        shapes = []
        for fr in fractures:
            shape = base_shape.copy()
            shape = shape.scale([fr.rx, fr.ry, 1]) \
                .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
                .translate(fr.centre) \
                .set_region(fr.region)

            shapes.append(shape)

        fracture_fragments = self.fragment(*shapes)
        return fracture_fragments

    def _assign_physical_groups(self, obj):
        self.synchronize()
        reg_to_tags = {}
        reg_names = defaultdict(set)

        # collect tags of regions
        for dimtag, reg in obj.dimtagreg():
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

    def make_mesh(self, objects: List['ObjectSet'], dim=3, eliminate=True) -> None:
        """
        Generate mesh for given objects.
        :param dim: Set highest dimension to mesh.
        """
        group = self.group(*objects)
        self._assign_physical_groups(group)
        if eliminate:
            self.keep_only(group)
        self.synchronize()
        gmsh.model.mesh.generate(dim)
        gmsh.model.mesh.removeDuplicateNodes()
        bad_entities = gmsh.model.mesh.getLastEntityError()
        if bad_entities:
            print("Bad entities:", bad_entities)

    def write_brep(self, filename=None):
        self.synchronize()
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

    def remove_duplicate_entities(self):
        self.synchronize()
        self.model.removeAllDuplicates()
        self._need_synchronize = True

    def keep_only(self, *object_sets):
        self.synchronize()
        if object_sets:
            group_dimtags = self.group(*object_sets).dim_tags
        else:
            group_dimtags = []
        all_dimtags = set(gmsh.model.getEntities())
        remove_dimtags = all_dimtags.difference(set(group_dimtags))
        try:
            self.model.remove(list(remove_dimtags), recursive=False)
        except ValueError:
            pass

    def all_entities(self):
        self.synchronize()
        return gmsh.model.getEntities()

    def show(self):
        gmsh.fltk.run()

    def __del__(self):
        gmsh.finalize()






class ObjectSet:
    def __init__(self, factory: GeometryOCC, dim_tags: List[DimTag], regions: List[Region]) -> None:
        self.factory = factory
        self.dim_tags = dim_tags
        if len(regions) == 1:
            self.regions = [regions[0] for _ in dim_tags]
        else:
            assert (len(regions) == len(dim_tags))
            self.regions = regions

    @property
    def tags(self):
        return [tag for dim, tag in self.dim_tags]

    @property
    def size(self):
        return len(self.dim_tags)

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

    def modify_regions(self, format: str) -> None:
        """
        For every of object's regions create a new region with a name given by the 'format'
        and original region name.
        : param format: a string format with single placeholder.
        E.g. to prefix all region names by 'XY_' use format "XY_{}".

        TODO: allow to include: dim, tag, entity type into the format string through named placehodders
        """
        regions = []
        for region in self.regions:
            new_name = format.format(region.name)
            new_region = self.factory.get_region_name(new_name)
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
        TODO: Return Group
        """
        reg_to_tags = {}
        # collect tags of regions
        for dimtag, reg in self.dimtagreg():
            dim, tag = dimtag
            reg.complete(dim)
            reg_to_tags.setdefault(reg.id, (reg, []))
            reg_to_tags[reg.id][1].append(dimtag)
        reg_sets = [ObjectSet(self.factory, dimtags, [reg]) for reg, dimtags in reg_to_tags.values()]
        return reg_sets

    def split_by_dimension(self):
        """
        Split objects in ObjectSet into ObjectSets of same dimansion.
        :return: list of ObjectSets
        TODO: Return Group
        """
        dimtags = [[], [], [], []]
        regions = [[], [], [], []]
        for dimtag, reg in self.dimtagreg():
            dim, tag = dimtag
            reg.complete(dim)
            dimtags[dim].append(dimtag)
            regions[dim].append(reg)
        sets = [ObjectSet(self.factory, dimtags, regs) for regs, dimtags in zip(regions, dimtags)]
        return sets

    def get_boundary_per_region(self, format=".{}"):
        """
        Split object by regions, call get_boundary for individual region subobjects and assign
        related boundary regions.
        :return:
        TODO: Return Group
        """
        reg_sets = self.split_by_region()
        b_sets = []
        for rset in reg_sets:
            reg = rset.regions[0]
            b_reg_name = format.format(reg.name)
            b_reg = Region.get(b_reg_name, dim=reg.dim - 1)
            #self.factory.get_region_name()

            boundary = rset.get_boundary(combined=True).set_region(b_reg)
            b_sets.append(boundary)
        return b_sets

    def have_common_dim(self, dim_tags=None):
        if dim_tags is None:
            dim_tags = self.dim_tags
        assert dim_tags
        dim = dim_tags[0][0]
        for d, tag in dim_tags:
            if d != dim:
                return None
        return dim

    def dimtagreg(self):
        assert len(self.regions) == len(self.dim_tags)
        return zip(self.dim_tags, self.regions)

    def set_mesh_step(self, step):
        """
        Set mesh step 'step' to all nodes of the objects in the ObejctSet.
        """
        # Get boundary resursive to obtain nodes
        self.factory.synchronize()
        try:
            dimtags = gmsh.model.getBoundary(self.dim_tags, combined=False, oriented=False, recursive=True)
        except ValueError as err :
            message = "\nobj dimtags: {}".format(str(self.dim_tags[:10]))
            raise GetBoundaryError(message) from err
        nodes = [(dim, tag) for dim, tag in dimtags if dim == 0]
        gmsh.model.mesh.setSize(nodes, step)

    def select_by_intersect(self, *tool_objects: 'ObjectSet') -> 'ObjectSet':
        """
        Make intersection with copy of the object
        :param tool_objects:
        :return:
        """
        sc = self.copy()
        tool = self.factory.group(*tool_objects).copy()
        objs, map = self.factory.model.intersect(sc.dim_tags, tool.dim_tags)
        tool.invalidate()
        sc.invalidate()

        isec = []
        for dimtag_map, dimtagreg in zip(map, self.dimtagreg()):
            if len(dimtag_map) > 1:
                raise BoolOperationError("Can not select by intersect, insufficient fragmentation:\n{}".format(self.dim_tags))
            if len(dimtag_map) == 1:
                isec.append(dimtagreg)
        if isec:
            dimtags, regs = zip(*isec)
        else:
            return ObjectSet(self.factory, [], [])
        return ObjectSet(self.factory, dimtags, regs)

    def split_by_cut(self, *tool_objects: 'ObjectSet') -> Tuple['ObjectSet', 'ObjectSet', 'ObjectSet', 'ObjectSet']:
        """
        Cut self object and return both cut object and the remainder object.
        Doesn't work preprely for boundaries due to a bug i OCC.

        :param tool_objects: any number of ObjectSet
        :return: cut objectset, intersection objectset, tool remainder objectset
        TODO: Return Group
        """
        factory = self.factory
        tool_objects = self.factory.group(*tool_objects)
        new_obj, new_tool = self.factory.fragment(self, tool_objects)
        dict_obj = dict(new_obj.dimtagreg())
        dict_tool = dict(new_tool.dimtagreg())
        cut_obj = {k: dict_obj[k] for k in set(dict_obj) - set(dict_tool)}
        cut_tool = {k: dict_tool[k] for k in set(dict_tool) - set(dict_obj)}
        isec_set = set(dict_obj) & set(dict_tool)
        isec_obj = {k: dict_obj[k] for k in isec_set}
        isec_tool = {k: dict_tool[k] for k in isec_set}

        out_objs = [ ObjectSet(factory, list(d.keys()), list(d.values()))
                        for d in [cut_obj, cut_tool, isec_obj, isec_tool] ]
        return out_objs

    def set_region_from_dimtag(self):
        """
        Mainly for debugging purposes. Set new regions for every dimtag.
        :return:
        """
        regions = []
        for dim, tag in self.dim_tags:
            name = "{}_{}".format(dim, tag)
            regions.append(self.factory.get_region_name(name))
        self.regions = regions

    def _apply_operation(self, tool_objects, operation):
        tool_objects = self.factory.group(*tool_objects).copy()
        try:
            new_tags, old_tags_map = operation(self.dim_tags, tool_objects.dim_tags, removeObject=True, removeTool=True)
        except ValueError as err :
            message = "\nobj dimtags: {}\ntool dimtags: {}".format(str(self.dim_tags[:10]), str(tool_objects.dim_tags[:10]))
            raise BoolOperationError(message) from err

        # assign regions
        assert len(self.regions) == len(self.dim_tags), (len(self.regions), len(self.dim_tags))
        old_tags_objects = [ObjectSet(self.factory, new_subtags, [reg])
                            for reg, new_subtags in zip(self.regions, old_tags_map[:len(self.dim_tags)])]
        new_obj = self.factory.group(*old_tags_objects)

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

    def cut(self, *tool_objects) -> 'ObjectSet':
        """
        Cut self object with 'tool_objects'.
        Returns the cut object, self is destroyed, tool_objects are preserved (we use their copy).
        Regions set on self are transfered to the result.
        """
        return self._apply_operation(tool_objects, self.factory.model.cut)

    def intersect(self, *tool_objects) -> 'ObjectSet':
        """
        Intersect self object with 'tool_objects'.
        Returns the intersected object, self is destroyed, tool_objects are preserved (we use their copy).
        Regions set on self are transfered to the result.
        """
        return self._apply_operation(tool_objects, self.factory.model.intersect)

    def fragment(self, *tool_objects) -> 'ObjectSet':
        """
        Fragment self object with 'tool_objects'.
        Returns the fragmented object, self is destroyed, tool_objects are preserved (we use their copy).
        Regions set on self are transfered to the result.
        """
        return self._apply_operation(tool_objects, self.factory.model.fragment)

    def invalidate(self):
        self.factory = None
        self.dim_tags = None
        self.regions = None

    def mass(self):
        return sum((self.factory.model.getMass(dimtag) for dimtag in self.dim_tags))

    def center_of_mass(self):
        center = np.zeros(3)
        mass_total = 0
        for dimtag in self.dim_tags:
            mass = self.factory.model.getMass(*dimtag)
            center += mass*np.array(self.factory.model.getCenterOfMass(*dimtag))
            mass_total += mass
        if mass_total > 0.0:
            return center/mass_total, mass_total
        else:
            return 0, 0

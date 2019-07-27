import gmsh
from typing import TypeVar, List
gmsh_field = gmsh.model.mesh.field

Field = int

def set_mesh_step_field(field: Field) -> None:
    gmsh_field.setAsBackgroundMesh(field)


def distance_nodes(nodes, coordinate_fields=None):
    """
    Distance from a set of 'nodes' given by their tags.
    Optional coordinate_fields = ( field_x, field_y, field_z),
    gives fields used as X,Y,Z coordinates (not clear how exactly these curved coordinates are used).
    """
    id = gmsh_field.add('Distance')
    gmsh_field.setNumbers(id, "NodesList", nodes)
    if coordinate_fields:
        fx, fy, fz = coordinate_fields
        gmsh_field.setNumber(id, "FieldX", fx)
        gmsh_field.setNumber(id, "FieldY", fy)
        gmsh_field.setNumber(id, "FieldZ", fz)
    return id


def distance_edges(curves, nodes_per_edge=8, coordinate_fields=None) -> Field:
    """
    Distance from a set of curves given by their tags. Curves are replaced by 'node_per_edge' nodes
    and DistanceNodes is applied.
    Optional coordinate_fields = ( field_x, field_y, field_z),
    gives fields used as X,Y,Z coordinates (not clear how exactly these curved coordinates are used).
    """
    id = gmsh_field.add('Distance')
    gmsh_field.setNumbers(id, "EdgesList", curves)
    gmsh_field.setNumber(id, "NNodesByEdge", nodes_per_edge)
    if coordinate_fields:
        fx, fy, fz = coordinate_fields
        gmsh_field.setNumber(id, "FieldX", fx)
        gmsh_field.setNumber(id, "FieldY", fy)
        gmsh_field.setNumber(id, "FieldZ", fz)
    return id


def threshold(field, lower_bound, upper_bound=None, sigmoid=False) -> Field:
    """
    field_min, threshold_min = lower_bound
    field_max, threshold_max = lower_bound

    threshold = threshold_min IF field <= field_min
    threshold = threshold_max IF field >= field_max
    interpolation otherwise.

    upper_bound = None is equivalent to field_max = infinity
    Linear interpolation is used unless 'sigmoid' is set.
    """
    id = gmsh_field.add('Threshold')

    gmsh_field.setNumber(id, "IField", field)
    field_min, threshold_min = lower_bound
    gmsh_field.setNumber(id, "DistMin", field_min)
    gmsh_field.setNumber(id, "LcMin", threshold_min)
    if upper_bound:
        field_max, threshold_max = lower_bound
        gmsh_field.setNumber(id, "DistMax", field_max)
        gmsh_field.setNumber(id, "LcMax", threshold_max)
    else:
        gmsh_field.setNumber(id, "StopAtDistMax", True)

    if sigmoid:
        gmsh_field.setNumber(id, "Sigmoid", True)
    return id


def min(fields: List[Field]) -> Field:
    """
    Field that is minimum of other fields.
    :param fields: list of fields
    :return: field
    """
    id = gmsh_field.add('Min')
    gmsh_field.setNumbers(id, "FieldsList", fields)
    return id


def max(fields: List[Field]) -> Field:
    """
    Field that is maximum of other fields.
    :param fields: list of fields
    :return: field
    """
    id = gmsh_field.add('Max')
    gmsh_field.setNumbers(id, "FieldsList", fields)
    return id


def box(pt_min, pt_max, v_in, v_out=1e300):
    """
    The value of this field is VIn inside the box, VOut outside the box. The box is given by

    pt_a <= (x, y, z) <= pt_b

    Can be used instead of a constant field.
    """
    id = gmsh_field.add('Box')
    gmsh_field.setNumber(id, "VIn", v_in)
    gmsh_field.setNumber(id, "VOut", v_out)
    gmsh_field.setNumber(id, "XMax", pt_max[0])
    gmsh_field.setNumber(id, "XMin", pt_min[0])
    gmsh_field.setNumber(id, "YMax", pt_max[1])
    gmsh_field.setNumber(id, "YMin", pt_min[1])
    gmsh_field.setNumber(id, "ZMax", pt_max[2])
    gmsh_field.setNumber(id, "ZMin", pt_min[2])
    return id


def constant(value, radius):
    """
    Make a field with constant value = value.
    Emulated using a box field with box containing shpere in origin with given 'readius'.
    """
    return box((-radius, -radius, -radius), (radius, radius, radius), v_in=value)

def restrict(field:Field, *object_sets, add_boundary=False):
    """
    Restrict the application of a 'field' to a given list of geometrical entities: points, curves, surfaces or volumes.
    Entities are given as object sets.
    """
    if not object_sets:
        return

    factory = object_sets[0].factory
    factory.synchronize()
    group = factory.group(*object_sets)
    if add_boundary:
        b_group = group.get_boundary(combined=False)
        group = factory.group(group, b_group)
    points, edges, faces, volumes = group.split_by_dimension()
    id = gmsh_field.add('Restrict')
    gmsh_field.setNumber(id, "IField", field)
    gmsh_field.setNumbers(id, "VerticesList", points.tags)
    gmsh_field.setNumbers(id, "EdgesList", edges.tags)
    gmsh_field.setNumbers(id, "FacesList", faces.tags)
    gmsh_field.setNumbers(id, "RegionsList", volumes.tags)
    return id
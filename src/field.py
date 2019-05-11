from gmsh_api import gmsh
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


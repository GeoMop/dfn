import gmsh_io
import heal_mesh
import numpy as np
import pytest

# def test_check_flat_element():
#     h = 0.01
#     nodes = np.array([[0,0,0], [1,1,0], [0,1,h], [1,0,h]], dtype=float)
#     ref_nodes = np.array([[0,0,h/2], [1,1,h/2], [0,1, h/2], [1,0,h/2]], dtype=float)
#     result = heal_mesh.check_flat_element(nodes, 0.05)
#     edge, new_nodes, isec = result
#     assert edge == [0,1,2,3]
#     assert np.allclose(new_nodes, ref_nodes)
#     assert np.allclose(isec, np.array([0.5, 0.5, h/2]))
#
#     nodes = np.array([[0,0,0], [0,1,h], [1,1,0], [1,0,h]], dtype=float)
#     ref_nodes = np.array([[0,0,h/2], [0,1, h/2], [1,1,h/2], [1,0,h/2]], dtype=float)
#     result = heal_mesh.check_flat_element(nodes, 0.05)
#     edge, new_nodes, isec = result
#     assert edge == [0,2,1,3]
#     assert np.allclose(new_nodes, ref_nodes)
#     assert np.allclose(isec, np.array([0.5, 0.5, h/2]))

@pytest.mark.skip
def test_tet_common_normal():
    h = 0.01
    nodes = np.array([[0, 0, 0], [1, 1, 0], [0, 1, h], [1, 0, h]], dtype=float)
    shape = heal_mesh.Tetrahedron(nodes)
    common_norm = shape.common_normal()
    assert np.allclose(np.zeros(6), shape.edge_vectors @ common_norm, atol=2*h)

    nodes = np.array([[0, 0, h], [1, 0, 0], [0, 1, 0], [0.2, 0.2, h]], dtype=float)
    shape = heal_mesh.Tetrahedron(nodes)
    common_norm = shape.common_normal()
    assert np.allclose(np.zeros(6), shape.edge_vectors @ common_norm, atol=2*h)

def test_check_flat_quad():
    h = 0.005
    nodes = np.array([[0, 0, 0], [1, 1, 0], [0, 1, h], [1, 0, h]], dtype=float)
    mesh_io = gmsh_io.GmshIO()
    for inn, n in enumerate(nodes):
        mesh_io.nodes[inn] = n
    mesh_io.elements[0] = (4, (1,2,3), [0,1,2,3])

    hm = heal_mesh.HealMesh(mesh_io)
    hm._check_flat_tetra(0, 0.01)

def test_check_flat_quad_degen():
    h = 0.005
    nodes = np.array([[0, 0, 0], [0.501, 0.501, 0], [0, 1, h], [1, 0, h]], dtype=float)
    mesh_io = gmsh_io.GmshIO()
    for inn, n in enumerate(nodes):
        mesh_io.nodes[inn] = n
    mesh_io.elements[0] = (4, (1,2,3), [0,1,2,3])

    hm = heal_mesh.HealMesh(mesh_io)
    hm._check_flat_tetra(0, 0.01)
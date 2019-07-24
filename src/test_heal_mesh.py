import heal_mesh
import numpy as np

def test_check_flat_element():
    h = 0.01
    nodes = np.array([[0,0,0], [1,1,0], [0,1,h], [1,0,h]], dtype=float)
    ref_nodes = np.array([[0,0,h/2], [1,1,h/2], [0,1, h/2], [1,0,h/2]], dtype=float)
    result = heal_mesh.check_flat_element(nodes, 0.05)
    edge, new_nodes, isec = result
    assert edge == [0,1,2,3]
    assert np.allclose(new_nodes, ref_nodes)
    assert np.allclose(isec, np.array([0.5, 0.5, h/2]))

    nodes = np.array([[0,0,0], [0,1,h], [1,1,0], [1,0,h]], dtype=float)
    ref_nodes = np.array([[0,0,h/2], [0,1, h/2], [1,1,h/2], [1,0,h/2]], dtype=float)
    result = heal_mesh.check_flat_element(nodes, 0.05)
    edge, new_nodes, isec = result
    assert edge == [0,2,1,3]
    assert np.allclose(new_nodes, ref_nodes)
    assert np.allclose(isec, np.array([0.5, 0.5, h/2]))
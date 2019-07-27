import pytest
import heal_mesh
import os

mesh_files=[
    "random_fractures_01.msh",
    "random_fractures_02.msh",
    "random_fractures_03.msh"
]
@pytest.mark.parametrize("mesh", mesh_files)
@pytest.mark.parametrize("tol", [0.003, 0.01, 0.03])
def test_on_mesh_samples(mesh, tol):
    mesh_path = os.path.join("meshes", mesh)
    hm = heal_mesh.HealMesh.read_mesh(mesh_path, node_tol=tol*0.01)
    hm.heal_mesh(gamma_tol=tol)
    hist, bins, bad_els = hm.quality_statistics(bad_el_tol=tol)
    for name, h in hist.items():
        hm.print_stats(h, bins, name)
        print("# bad els: ", len(bad_els[name]))

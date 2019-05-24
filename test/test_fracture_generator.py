import numpy as np
import fracture_generator as fg

def test_fisher_orientation():
    #normals = np.random.rand(10, 3)
    normals = [[0,0,1], [0,1,0], [1,0,0], [0.01, 0, 1]]

    axis_angle = fg.FisherOrientation.normal_to_axis_angle(np.array(normals, dtype=float))
    # print(axis_angle)
    z_axis = np.array([0,0,1])
    for aa, nn in zip(axis_angle, normals):
        normal = fg.FisherOrientation.rotate(z_axis, axis_angle=aa)
        assert np.allclose(normal, nn, rtol=1e-4)

    fr = fg.FisherOrientation(45, 60, np.inf)
    sin_pi_4 = np.sin(np.pi / 4)
    sin_pi_3 = np.sin(np.pi / 3)
    normal = fr.sample_normal()
    assert np.allclose([0.5*sin_pi_4, 0.5*sin_pi_4, sin_pi_3], normal)
    aa = fr.sample_axis_angle()
    assert np.allclose([-sin_pi_4, sin_pi_4, 0, np.pi/6], aa)





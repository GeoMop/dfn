import numpy as np
import pytest
import src.fracture as frac

from collections import defaultdict


def test_fisher_orientation():
    normals = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0.01, 0, 1]]

    axis_angle = frac.FisherOrientation.normal_to_axis_angle(np.array(normals, dtype=float))
    # print(axis_angle)
    z_axis = np.array([0,0,1])
    for aa, nn in zip(axis_angle, normals):
        normal = frac.FisherOrientation.rotate(z_axis, axis_angle=aa)
        assert np.allclose(normal, nn, rtol=1e-4)

    fr = frac.FisherOrientation(45, 60, np.inf)
    sin_pi_4 = np.sin(np.pi / 4)
    sin_pi_3 = np.sin(np.pi / 3)
    normal = fr.sample_normal()
    assert np.allclose([0.5*sin_pi_4, 0.5*sin_pi_4, sin_pi_3], normal)
    aa = fr.sample_axis_angle()
    assert np.allclose([-sin_pi_4, sin_pi_4, 0, np.pi/6], aa)


@pytest.mark.parametrize("volume, intensity, size_range, kappa", [pytest.param(1, 3, [1, 10], 2.1),
                                                                  pytest.param(5, 3, [2, 10], 7)])
def test_power_law(volume, intensity, size_range, kappa):
    """
    Test power law size
    :param volume: Cube volume
    :param intensity: Number of fractures
    :param size_range:
    :param kappa: Power param
    :return:
    """
    power_law_size = frac.PowerLawSize(power=kappa, diam_range=size_range, intensity=intensity,
                                       sample_range=size_range)

    power_law_size.set_sample_range(None)
    mean_size = power_law_size.mean_size(volume)

    samples_length = []
    for i in range(200):
        samples_length.append(len(power_law_size.sample(volume)))

    assert np.isclose(np.mean(samples_length), mean_size, 0.2)


def test_fracture_population():
    """
    Test base sample structures
    :return: None
    """
    volume = 1
    pop = frac.Population.load("test_skb_data.json", volume)
    samples = pop.sample()

    for sample in samples:
        assert sample.r > 0
        assert len(sample.rotation_axis) == 3
        assert len(sample.centre) == 3
        assert sample.rotation_angle > 0


def test_intensity_p_32():
    """
    Test fracture intensity (P30) and total fractures size per volume unit (P32)
    :return: None
    """
    rep = 10000  # Number of repetitions
    volume = 1

    families = defaultdict(list)
    pop = frac.Population.load("test_skb_data.json", volume)

    for _ in range(rep):
        samples = pop.sample()
        for sample in samples:
            families[sample.region].append(sample)

    for family in pop.families:
        fractures = families[family.name]
        average_size = sum([fracture.r ** 2 for fracture in fractures]) / rep
        # Test intensity
        assert np.isclose(family.shape.intensity, len(families[family.name]) / rep, 1)
        # Test P 32
        assert np.isclose(family.shape.p_32, average_size, 0.1)


if __name__ == "__main__":
    test_intensity_p_32()
    #test_fracture_population()


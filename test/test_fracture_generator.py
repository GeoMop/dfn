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


def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return (x,y)

@pytest.mark.parametrize("volume, intensity, size_range, kappa", [pytest.param(1000, 3, [1, 10], 2.1),
                                                                  pytest.param(5000, 3, [2, 10], 7)])
def test_power_law(volume, intensity, size_range, kappa):
    """
    Test power law size
    :param volume: Cube volume
    :param intensity: Number of fractures
    :param size_range:
    :param kappa: Power param
    :return:
    """
    np.random.seed(123)
    p_law = frac.PowerLawSize(power=kappa, diam_range=size_range, intensity=intensity)
    assert np.isclose(intensity, p_law.range_intensity(p_law.sample_range))
    assert np.isclose(intensity * volume, p_law.mean_size(volume))

    # expected intensity on subrange using CDF
    mid_range = (size_range[0] + size_range[1])/2
    p1 = p_law.cdf(mid_range, p_law.sample_range) - p_law.cdf(size_range[0], p_law.sample_range)
    p2 = p_law.cdf(size_range[1], p_law.sample_range) - p_law.cdf(mid_range, p_law.sample_range)
    p_law.set_sample_range([size_range[0], mid_range])
    assert np.isclose(intensity * p1, p_law.range_intensity(p_law.sample_range))
    p_law.set_sample_range([mid_range, size_range[1]])
    assert np.isclose(intensity * p2, p_law.range_intensity(p_law.sample_range))
    p_law.set_sample_range()
    assert np.isclose(intensity, p_law.range_intensity(p_law.sample_range))

    p_law.set_range_by_intensity(intensity / 2)
    assert np.isclose(intensity / 2, p_law.range_intensity(p_law.sample_range))
    p_law.set_sample_range()

    # verify sample statistics
    n_samples = 1000
    samples = [p_law.sample(volume) for _ in range(n_samples)]

    # check ecdf vs. cdf
    all_samples = np.concatenate(samples)[1:100000]
    X, Y = ecdf(all_samples.tolist())
    Y2 = [p_law.cdf(x, p_law.sample_range) for x in X]
    #import matplotlib.pyplot as plt
    #plt.plot(X, Y2-Y, 'red')
    #plt.show()
    assert np.std(Y2 - Y) < 0.001

    # check sample size vs. intensity
    sample_lens = np.array([len(s) for s in samples])
    ref_mean_size = p_law.mean_size(volume)
    est_std = np.sqrt(ref_mean_size / n_samples)
    print("mean size: ", np.mean(sample_lens), "ref size: ", ref_mean_size, "std: ", np.std(sample_lens))
    assert np.isclose(np.mean(sample_lens), ref_mean_size, 3*est_std)

    # check sample fracture area vs. mean area
    sample_areas = np.array([sum(4 * s ** 2) for s in samples])
    mean_area = p_law.mean_area(volume)
    est_std = np.sqrt(mean_area / n_samples)
    print("mean area: ", np.mean(sample_areas), "ref area: ", mean_area, "std: ", np.std(sample_areas), est_std)
    assert np.isclose(np.mean(sample_areas), mean_area, 3*est_std)


    # check sample relative frequencies vs. p1 and p2 probabilities
    n_fr_p1 = [np.sum(s < mid_range) for s in samples]
    n_fr_p2 = [np.sum(s >= mid_range) for s in samples]
    s_p1 = np.array(n_fr_p1) / sample_lens
    s_p2 = np.array(n_fr_p2) / sample_lens
    binom_var = p1 * (1 - p1) / ref_mean_size
    est_std = np.sqrt(binom_var / n_samples)
    print("var: ", binom_var, np.var(s_p1))
    print("p1 :", np.mean(s_p1), p1, "diff: ", np.mean(s_p1) - p1, "est_std: ", est_std)
    assert np.isclose(np.mean(s_p1), p1, atol=3 * est_std)
    binom_var = p2 * (1 - p2) / ref_mean_size
    est_std = np.sqrt(binom_var / n_samples)
    print("var: ", binom_var, np.var(s_p2))
    print("p2 :", np.mean(s_p2), p2, "diff: ", np.mean(s_p2) - p2, "est_std: ", est_std)
    assert np.isclose(np.mean(s_p2), p2, atol=3 * est_std)    # ?? binom_var not sure

    # init from area
    p_law.mean_area()
    p_law2 = frac.PowerLawSize.from_mean_area(power=kappa, diam_range=size_range, p32=p_law.mean_area())
    assert np.isclose(p_law2.mean_area(), p_law.mean_area())
    assert np.isclose(p_law2.mean_size(), p_law.mean_size())

def test_fracture_population():
    """
    Test base sample structures
    :return: None
    """
    volume = 1
    pop = frac.Population(volume)
    pop.init_from_json("test_skb_data.json")
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
    TODO:
    - imporve area test variances are big, possibly need to collect size and area per repetition
    """
    rep = 100  # Number of repetitions
    volume = 10

    family_n_frac = defaultdict(int)
    family_frac_surface = defaultdict(float)
    pop = frac.Population(volume)
    pop.init_from_json("test_skb_data.json")

    for _ in range(rep):
        fractures = pop.sample()
        for fr in fractures:
            family_n_frac[fr.region] += 1
            family_frac_surface[fr.region] += fr.r ** 2

    for family in pop.families:
        print(family.name)
        n_frac = family_n_frac[family.name]
        frac_surface = family_frac_surface[family.name]
        # Test intensity
        mean_size = family.shape.mean_size(volume)
        est_std = np.sqrt(mean_size / rep)
        print("size: ", family.shape.intensity * volume, n_frac / rep, est_std)
        assert np.isclose(family.shape.intensity * volume, n_frac / rep, est_std)
        # Test P 32
        est_std = np.sqrt(family.shape.mean_area(volume) / mean_size / rep)
        ref_area = family.shape.mean_area(volume)
        sample_area = frac_surface / rep
        print("area: ", ref_area, sample_area, "diff: ",ref_area - sample_area, 10*est_std)
        assert np.isclose(ref_area, sample_area, atol=3*est_std)


if __name__ == "__main__":
    test_intensity_p_32()
    #test_fracture_population()


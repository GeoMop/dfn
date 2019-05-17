import numpy as np
import attr



class Quat:
    def __init__(self, q):
        self.q = q

    def __matmul__(self, other: 'Quat') -> 'Quat':
        """
        Composition of rotations. Quaternion multiplication.
        """
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return Quat((w, x, y, z))


    @staticmethod
    def from_euler(a:float, b:float, c:float) -> 'Quat':
        """
        X-Y-Z Euler angles to quaternion
        :param a: angle to rotate around Z
        :param b: angle to rotate around X
        :param c: angle to rotate around Z
        :return: Quaterion for composed rotation.
        """
        return Quat([np.cos(a / 2), 0, 0,np.sin(a / 2)]) @ \
                Quat([np.cos(b / 2), 0, np.sin(b / 2), 0]) @\
                Quat([np.cos(c / 2), np.sin(c / 2), 0, 0])

    def axisangle_to_q(v, theta):
        # convert rotation given by axis 'v' and angle 'theta' to quaternion representation
        v = normalize(v)
        x, y, z = v
        theta /= 2
        w = cos(theta)
        x = x * sin(theta)
        y = y * sin(theta)
        z = z * sin(theta)
        return w, x, y, z

    def q_to_axisangle(q):
        # convert from quaternion to ratation given by axis and angle
        w, v = q[0], q[1:]
        theta = acos(w) * 2.0
        return normalize(v), theta


@attr.s(auto_attribs=True)
class FisherOrientation:
    """
    Distribution for random orientation in 3d.

    Coordinate system: X - east, Y - north, Z - up
    """

    trend: float
    # mean fracture normal, azimuth of its projection to the horizontal plane
    # related term is the strike =  trend - 90; that is azimuth of the strike line
    # - the intersection of the fracture with the horizontal plane
    plunge: float
    # mean fracture normal, angle between the normal and the horizontal plane
    # ralated term is the dip = 90 - plunge; that is the angle between the fracture and the horizontal plane
    #
    # strike and dip can by understood as the first two Eulerian angles.
    dispersion: float
    # the dispersion parameter; 0 = uniform dispersion, infty - no dispersion
    # k = ln(K)
    # theta = invcos{   1/ln(K)  ln( K(1 - F)   +  F/K)) }
    # K -> -infty: theta ->
    # k

    @staticmethod
    def strike_dip(strike, dip, dispersion):
        """
        Initialize from (strike, dip, concentration==dispersion)
        """
        return FisherOrientation(strike + 90, 90 - dip, dispersion)


    def _sample_standard_fisher(self, n) -> np.array:
        """
        Normal vector of random fractures with mean direction (0,0,1).
        :param n:
        :return: array of normals (n, 3)
        """
        if self.dispersion == np.inf:
            normals = np.zeros((n, 3))
            normals[:, 2] = 1.0
        else:
            unif = np.random.uniform(size=n)
            psi = 2 * np.pi * np.random.uniform(size=n)
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            if self.dispersion == 0:
                cos_theta = 1 - 2 * unif
            else:
                exp_k = np.exp(self.dispersion)
                exp_ik = 1 / exp_k
                cos_theta = np.ln(exp_k - unif * ( exp_k - exp_ik) ) / self.dispersion
            sin_theta = np.sqrt(1 - cos_theta**2)
            normals = np.stack((sin_psi * cos_theta, cos_psi * cos_theta, sin_theta), axis=1)
        return normals
        # rotate to mean strike and dip

    def sample_normal(self, size=1):
        raw_normals = self._sample_standard_fisher(size)
        mean_norm = self._mean_normal()
        axis_angle = self.normal_to_axis_angle(mean_norm[None,:])
        return self.rotate(raw_normals, axis_angle = axis_angle[0])

    def sample_axis_angle(self, size=1):
        """
        Sample fracture orientation angles.
        :param n: Number of samples
        :return: shape (n, 4), every row: unit axis vector and angle
        """
        normals = self.sample_normal(size)
        return self.normal_to_axis_angle(normals[:])

    @staticmethod
    def normal_to_axis_angle(normals):
        z_axis = np.array([0, 0, 1], dtype=float)
        norms = normals / np.linalg.norm(normals, axis=1)[:, None]
        cos_angle = norms @ z_axis
        angles = np.arccos(cos_angle)
        # sin_angle = np.sqrt(1-cos_angle**2)

        axes = np.cross(z_axis, norms, axisb=1)
        ax_norm = np.maximum( np.linalg.norm(axes, axis=1), 1e-200)
        axes = axes / ax_norm[:, None]

        return np.concatenate([axes, angles[:, None]], axis=1)

    @staticmethod
    def rotate(vectors, axis=None, angle=0, axis_angle=None):
        """
        Rotate given vector around given 'axis' by the 'angle'.
        :param vectors: array of 3d vectors, shape (n, 3)
        :param axis_angle: pass both as array (4,)
        :return: shape (n, 3)
        """
        if axis_angle is not None:
            axis, angle = axis_angle[:3], axis_angle[3]
        if angle == 0:
            return vectors
        vectors = np.atleast_2d(vectors)
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)

        rotated = vectors * cos_angle\
                  + np.cross(axis, vectors, axisb=1) * sin_angle\
                  + axis[None, :] * (vectors @ axis)[:, None] * (1 - cos_angle)
        # Rodrigues formula for rotation of vectors around axis by an angle
        return rotated

    def _mean_normal(self):
        trend = np.radians(self.trend)
        plunge = np.radians(self.plunge)

        normal = np.array( [np.sin(trend) * np.cos(plunge),
                            np.cos(trend) * np.cos(plunge),
                            np.sin(plunge)])

        #assert np.isclose(np.linalg.norm(normal), 1, atol=1e-15)
        return normal


    # def normal_2_trend_plunge(self, normal):
    #
    #     plunge = round(degrees(-np.arcsin(normal[2])))
    #     if normal[1] > 0:
    #         trend = round(degrees(np.arctan(normal[0] / normal[1]))) + 360
    #     else:
    #         trend = round(degrees(np.arctan(normal[0] / normal[1]))) + 270
    #
    #     if trend > 360:
    #         trend = trend - 360
    #
    #     assert trend == self.trend
    #     assert plunge == self.plunge



@attr.s(auto_attribs=True)
class PowerLawSize:
    """
    Truncated Power Law model for the fracture size distribution.
    density:

    f(r) = f_0 r ** (-power - 1)

    on [size_min, size_max], zero elsewhere.
    """
    power: float
    # power of th power law
    size_min: float
    # lower bound of the power law
    size_max:float
    # upper bound of the power law

    def _cdf(self, x, min, max):
        # Distribution function
        pmin = min ** (-self.power)
        pmax = max ** (-self.power)
        return (x ** (-self.power) - pmin)/(pmax - pmin)


    def _pdf(self, x, min, max):
        # Quantile function
        pmin = min ** (-self.power)
        pmax = max ** (-self.power)
        scaled = pmin - x*(pmin - pmax)
        return scaled ** (-1 / self.power)


    def sample(self, size=1, min=0, max=np.inf):
        """
        Sample the fracture size
        :param min: Override population size_min, if min > size_min
        :param max:
        :return:
        """
        min = max(self.size_min, min)
        max = max(self.size_max, max)
        U = np.random.random(size=size)
        return self.pdf(U, min, max)

    def subinterval_fraction(self, min=0, max=np.inf):
        """
        Fraction of sizes (min, max) within the full population.
        :param min:
        :param max:
        :return:
        """
        min = max(self.size_min, min)
        max = max(self.size_max, max)
        p_min, p_max = self._cdf(np.array([min,max]), self.size_min, self.size_max)
        return p_max - p_min



# @attr.s(auto_attribs=True)
# class PoissonIntensity:
#     p32: float
#     # number of fractures
#     size_min: float
#     #
#     size_max:
#     def sample(self, box_min, box_max):



class FracturePopulation:
    """
    Data class to describe a population of random fractures with common parameters.
    """
    def __init__(self, orientation, size, intenzity):
        """

        :param orientation: Orientation stochastic model
        :param size: Size stochastic model.
        :param intenzity: Stochastic model.
        """
        self.orientation = orientation
        self.size = size
        self.intenzity = intenzity


    # size_min: float
    # # lower limit of the fracture (Power
    # self.name = kwargs.get("name", "")
    # self.trend = kwargs.get("trend")
    # self.plunge = kwargs.get("plunge")
    # self.strike = kwargs.get("strike")
    # self.dip = kwargs.get("dip")
    # self.k = kwargs.get("k")
    # self.r_0 = kwargs.get("r_0")
    # self.kappa = kwargs.get("kappa")
    # self.r_min = kwargs.get("r_min")
    # self.r_max = kwargs.get("r_max")
    # self.p_32 = kwargs.get("p_32")
    #
    # self.n_fractures = 10
    # self.p_30 = None
    # self.compute_p_30()

    def sample(self, box_min, box_max, r_min=0, r_max=np.inf):
        """
        Provide a single fracture sample from the population.
        :return:
        """
        n_fractures = self.intenzity.sample(box_min, box_max, r_min, r_max)
        fr_axis_angle = self.orientation.sample_axis_angle(n_fractures)
        fr_size = self.size.sample(n_fractures, r_min, r_max)
        # fr_centre =


class FractureGenerator:
    def __init__(self, frac_type):
        self.frac_type = frac_type

    def generate_fractures(self, min_distance, min_radius, max_radius):
        fractures = []

        for i in range(self.frac_type.n_fractures):
            x = uniform(2 * min_distance, 1 - 2 * min_distance)
            y = uniform(2 * min_distance, 1 - 2 * min_distance)
            z = uniform(2 * min_distance, 1 - 2 * min_distance)

            tpl = TPL(self.frac_type.kappa, self.frac_type.r_min, self.frac_type.r_max, self.frac_type.r_0)
            r = tpl.rnd_number()

            orient = Orientation(self.frac_type.trend, self.frac_type.plunge, self.frac_type.k)
            axis, angle = orient.compute_axis_angle()

            fd = FractureData(x, y, z, r, axis[0], axis[1], axis[2], angle, i * 100)

            fractures.append(fd)

        return fractures

    def write_fractures(self, fracture_data, file_name):
        with open(file_name, "w") as writer:
            for d in fracture_data:
                writer.write("%f %f %f %f %f %f %f %f %d\n" % (d.centre[0], d.centre[1], d.centre[2], d.r, d.rotation_axis[0],
                                                        d.rotation_axis[1], d.rotation_axis[2], d.rotation_angle, d.tag))

    def read_fractures(self, file_name):
        data = []
        with open(file_name, "r") as reader:
            for l in reader.readlines():
                x, y, z, r, axis_0, axis_1, axis_2, angle = [float(i) for i in l.split(' ')[:-1]]
                tag = int(l.split(' ')[-1])
                d = FractureData(x, y, z, r, axis_0, axis_1, axis_2, angle, tag)
                data.append(d)

        return data



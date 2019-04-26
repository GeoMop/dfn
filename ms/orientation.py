from scipy.stats import vonmises
import numpy as np
from random import *
from math import *


class Orientation:
    def __init__(self, trend, plunge, kappa):
        self.trend = trend
        self.plunge = plunge
        self.kappa = kappa

    def compute_axis_angle(self):
        x = vonmises.rvs(self.kappa, size=1)
        y = uniform(0, np.pi)

        vec_a = self.compute_mean_normal()
        vec_b, vec_c = self.compute_fracture_normal(vec_a)

        vec_n = cos(x)*vec_a + sin(x)*(sin(y)*vec_b + cos(y)*vec_c)

        axis = np.cross(vec_n, vec_a)
        angle = np.arccos(np.dot(vec_n, vec_a)/2)

        return axis, angle

    def compute_mean_normal(self):
        # dip = 90° - plunge
        # strike = trend + 90° or trend - 270°
        trend = self.trend
        plunge = self.plunge

        normal = np.zeros(3)
        normal[0] = sin(radians(trend)) * cos(radians(plunge))
        normal[1] = cos(radians(trend)) * cos(radians(plunge))
        normal[2] = -sin(radians(plunge))

        self.normal_2_trend_plunge(normal)

        assert isclose(np.linalg.norm(normal), 1, abs_tol=1e-15)
        return normal

    def compute_fracture_normal(self, vec_a):
        vec_b = np.random.randn(3)
        vec_b -= vec_b.dot(vec_a) * vec_a
        vec_b /= np.linalg.norm(vec_b)

        vec_c = np.cross(vec_a, vec_b)

        assert isclose(np.dot(vec_b, vec_a), 0, abs_tol=1e-12)
        assert isclose(np.dot(vec_c, vec_a), 0, abs_tol=1e-12)
        assert isclose(np.dot(vec_b, vec_c), 0, abs_tol=1e-12)
        assert isclose(np.linalg.norm(vec_b), 1, abs_tol=1e-12)
        assert isclose(np.linalg.norm(vec_c), 1, abs_tol=1e-12)

        return vec_b, vec_c

    def normal_2_trend_plunge(self, normal):

        plunge = round(degrees(-np.arcsin(normal[2])))
        if normal[1] > 0:
            trend = round(degrees(np.arctan(normal[0] / normal[1]))) + 360
        else:
            trend = round(degrees(np.arctan(normal[0] / normal[1]))) + 270

        if trend > 360:
            trend = trend - 360

        assert trend == self.trend
        assert plunge == self.plunge

import numpy as np


class TPL:
    def __init__(self, k, l_boundary, h_boundary, r_0):
        self.k = k
        self.l_boundary = l_boundary
        self.h_boundary = h_boundary

        self.f_0 = -(r_0 ** k)
        self.r_0 = r_0

    def rnd_number(self, size=1):
        """
        :param size:
        :return:
        """
        x = np.random.random(size=size)
        l_boundary_g = np.float_power(self.l_boundary, -self.k)
        h_boundary_g = np.float_power(self.h_boundary, -self.k)

        y = x/(-(self.r_0**self.k))
        y_max = 0
        y_min = (1 / -(self.r_0 ** self.k))

        y_range = y_max - y_min
        boundary_range = h_boundary_g - l_boundary_g
        new_x = (((y - y_min) * boundary_range)/ y_range) + l_boundary_g

        return new_x ** (-1/self.k)

    # def pdf(self, x):
    #     l_boundary_g = np.float_power(self.l_boundary, -self.k)
    #     h_boundary_g = np.float_power(self.h_boundary, -self.k)
    #
    #     return self.k *(self.r_0 ** self.k) *x ** (-self.k - 1)#/ (h_boundary_g - l_boundary_g)
    #     #return (self.k * (self.r_0 ** self.k)) * (x ** (-self.k - 1)) #/ (h_boundary_g - l_boundary_g)

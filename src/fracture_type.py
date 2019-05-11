import numpy as np
from scipy.stats import poisson


class FractureType:
    def __init__(self, **kwargs):
        print("kwargs ", kwargs)
        self.name = kwargs.get("name", "")
        self.trend = kwargs.get("trend")
        self.plunge = kwargs.get("plunge")
        self.strike = kwargs.get("strike")
        self.dip = kwargs.get("dip")
        self.k = kwargs.get("k")
        self.r_0 = kwargs.get("r_0")
        self.kappa = kwargs.get("kappa")
        self.r_min = kwargs.get("r_min")
        self.r_max = kwargs.get("r_max")
        self.p_32 = kwargs.get("p_32")

        self.n_fractures = 10
        self.p_30 = None
        self.compute_p_30()

    def compute_p_30(self):
        assert self.kappa > 2
        self.p_30 = self.p_32 / ((np.pi * self.kappa * (self.r_0 ** 2)) / (self.kappa - 2))

    def set_n_fractures(self, volume=1, use_poisson=False):
        if self.p_30 is not None:
            if use_poisson:
                self.n_fractures = poisson.rvs(self.p_30)
            else:
                self.n_fractures = round(self.p_30 * volume)
        else:
            print("First compute P_30")

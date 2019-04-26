from random import *
from fracture_data import FractureData
from tpl import TPL
from orientation import Orientation


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

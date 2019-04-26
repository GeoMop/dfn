import numpy as np


class FractureData:

    def __init__(self, x, y, z, r, axis_x, axis_y, axis_z, rotation_angle, tag):
        self.centre = np.array([x, y, z])
        self.r = r
        self.rotation_axis = np.array([axis_x, axis_y, axis_z])
        self.tag = tag
        self.rotation_angle = rotation_angle


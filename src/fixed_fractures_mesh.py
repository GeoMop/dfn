
import attr
from typing import Union
import numpy as np
import field
import gmsh


@attr.s(auto_attribs=True)
class FractureData:
    r: float
    # Size of the fracture, laying in XY plane
    centre: np.array
    # location of the barycentre of the fracture
    rotation_axis: np.array
    # axis of rotation
    rotation_angle: float
    # angle of rotation around the axis (?? counterclockwise with axis pointing up)
    region: Union[str, int]
    # name or ID of the physical group
    aspect: float = 1
    # aspect ratio of the fracture, y_length : x_length, x_length == r

    @property
    def rx(self):
        return self.r

    @property
    def ry(self):
        return self.r * self.aspect







def generate_mesh():
    r""" Create mesh and write it to a file.

    Parameters
    ----------
    fractures_data : list of FractureData
      Array of objects defining fractures.
    max_el_size : double
      Maximal size of mesh element.
    file_name : str
      File name to write mesh into.
    verbose : {0, 1}
      If set to 1, messages during mesh generation will be printed.
    """

    # geometry prameters
    box_size = 2000
    well_radius = 50
    well_length = 3000
    well_shift = 500


    # Main box
    factory = gmsh.Geometry('occ', "three_frac_symmetric")
    box = factory.box(3 * [box_size]).set_region("box")
    b_box = box.get_boundary(combined=True)

    # two vertical cut-off wells, just permeable part
    well_z_shift = -well_length/2
    left_center = [-well_shift, 0, 0]
    right_center = [well_shift, 0, 0]
    left_well = factory.cylinder(well_radius, axis=[0, 0, well_length])\
                    .translate([0,0,well_z_shift]).translate(left_center)
    right_well = factory.cylinder(well_radius, axis=[0, 0, well_length])\
                    .translate([0, 0, well_z_shift]).translate(right_center)


    b_right_well = right_well.get_boundary()
    b_left_well = left_well.get_boundary()

    # fracutres
    fractures = [
        FractureData(r, centre, axis, angle, tag) for r, centre, axis, angle, tag in
        [
            (1300, left_center,  [0, 1, 0], np.pi/6, 'left_fr'),
            (1300, right_center, [0, 1, 0], np.pi/6, 'right_fr'),
            (900, [0,0,0],      [0, 1, 0], np.pi/2, 'center_fr')
        ]]
    all_fractures = factory.make_fractures(fractures, factory.rectangle())


    box_drilled = box.cut([left_well, right_well])
    b_box_drilled = box_drilled.get_boundary()

    b_box = b_box.intersect(b_box_drilled.copy()).set_region(".outer_box")
    b_left_well = b_left_well.intersect(b_box_drilled.copy()).set_region(".left_well")
    b_right_well = b_right_well.intersect(b_box_drilled.copy()).set_region(".right_well")

    # 3d and surface fragmented by fractures
    box_and_boundary = factory.group([box_drilled, b_box, b_left_well, b_right_well])
    box_and_boundary_fr = box_and_boundary.fragment(all_fractures.copy())

    # fracture boundaries
    b_fractures_box = all_fractures.copy().intersect(b_box.copy()).prefix_regions(".box_")
    b_left_well = all_fractures.copy().intersect(b_left_well.copy()).prefix_regions(".left_well_")
    b_right_well = all_fractures.copy().intersect(b_right_well.copy()).prefix_regions(".right_well_")

    # fracures
    cut_fractures = all_fractures.intersect(box_drilled.copy())

    box_drilled_fr = box_drilled.fragment(cut_fractures.copy())
    b_box = b

    # define 3d volume and embed fractures into it
    factory.synchronize()

    min_el_size = well_radius
    fracture_el_size = box_size / 20
    fracture_size = 1000
    max_el_size = box_size / 5

    frac_bdry = cut_fractures.get_boundary()
    distance_field = field.distance_edges([tag for dm, tag in frac_bdry.dim_tags if dm == 1])
    threshold = field.threshold(distance_field, (0, fracture_el_size), (fracture_size, max_el_size))
    field.set_mesh_step_field(threshold)

    factory.write_brep()
    factory.make_mesh([box_drilled, cut_fractures])
    factory.write_mesh()
    gmsh.fltk.run()















if __name__ == "__main__":
    generate_mesh()
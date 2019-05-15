
import attr
from typing import Union
import numpy as np
import field
import gmsh

"""
TODO:
- unique names during region creation (fractures_0,1, ...)
- why separate boundary regions on wells
- keep all dimtags, remove which are not requested for meshing, test if copy of boundary dimtags are discretized well

Issues:
- need more advanced cut and intersection for set of objects:
  a_cut, b_cut, c_cut = factory.cut([a,b,c], tool)

- try gmsh.model.mesh.setSize(dimtags, size)




- class for statistic fracture
- full class for single generated fracture
- random fractures in the cube (use SKB)
- random fields
- flow123d on random fractures

- separate all fractures into coarse and fine mesh
- for every fine fracture get intersected elements in the coarse mesh
possible ways:
    - get bounding box for a fracture gmsh.model.getBoundingBox, 
      find elements in the box
    - try to add coarse mesh into a model using addDiscreteEntity
    - load fine fractures into flow123d, marked as boundary or 'tool region' (double dots)
      use tool fractures to identify elements (similar to rivers (1d intersectiong 2d)
      use field that depends on intersection surface and fracture properties
"""


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
    well_radius = 30
    well_length = 3000
    well_shift = 500


    # Main box
    factory = gmsh.Geometry('occ', "three_frac_symmetric", verbose=True)
    gopt = gmsh.GeometryOptions()
    gopt.Tolerance = 1e-2
    gopt.ToleranceBoolean = 1e-3
    gopt.MatchMeshTolerance = 1e-1

    box = factory.box(3 * [box_size]).set_region("box")
    b_box = box.get_boundary().copy().set_region(".outer_box")

    # two vertical cut-off wells, just permeable part
    well_z_shift = -well_length/2
    left_center = [-well_shift, 0, 0]
    right_center = [well_shift, 0, 0]
    left_well = factory.cylinder(well_radius, axis=[0, 0, well_length])\
                    .translate([0,0,well_z_shift]).translate(left_center)
    right_well = factory.cylinder(well_radius, axis=[0, 0, well_length])\
                    .translate([0, 0, well_z_shift]).translate(right_center)


    b_right_well = right_well.get_boundary().set_region(".left_well").copy()
    b_left_well = left_well.get_boundary().set_region(".right_well").copy()
    #b_wells = factory.group([b_left_well, b_right_well])

    # fracutres
    fractures = [
        FractureData(r, centre, axis, angle, tag) for r, centre, axis, angle, tag in
        [
            (1300, left_center,  [0, 1, 0], np.pi/6, 'left_fr'),
            (1300, right_center, [0, 1, 0], np.pi/6, 'right_fr'),
            (900, [0,0,0],      [0, 1, 0], np.pi/2, 'center_fr')
        ]]
    fractures = factory.make_fractures(fractures, factory.rectangle())
    fractures_group = factory.group(*fractures)
    print("made fractures")

    #wells = [left_well, right_well, b_left_well, b_right_well]
    #box_cut, x, box_intersect, y = factory.group(box, b_box).split_by_cut(*wells)
    #all = factory.group(box_cut, x, box_intersect, y)
    #b_well = box_intersect.split_by_dimension()[2]
    #box_cut.set_region_from_dimtag()
    #all.set_region_from_dimtag()

    #bb_box = factory.group([box, b_box])
    box_drilled = box.cut(left_well, right_well)
    b_box_drilled = box_drilled.get_boundary()
    b_left = b_box_drilled.select_by_intersect(b_left_well).set_region(".left_well")
    b_right = b_box_drilled.select_by_intersect(b_right_well).set_region(".right_well")
    b_box = b_box_drilled.select_by_intersect(b_box).set_region(".outer_box")
    box_all = factory.group(box_drilled, b_left, b_right, b_box)
    fractures_group = fractures_group.intersect(box_drilled.copy())
    box_all_fr, fractures_fr = factory.fragment(box_all, fractures_group)

    #tool_box_drilled = box_drilled.copy()
    #tool_fractures_group = fractures_group.copy()


    #box_drilled, fractures_group, b_right_well, b_left_well, b_box = factory.fragment(box_drilled, fractures_group, b_right_well, b_left_well, b_box)


    # #b_box_drilled = box_drilled.get_boundary()
    #
    #
    # # 3d and surface fragmented by fractures
    # #box_and_boundary = factory.group([box_drilled, b_box, b_left_well, b_right_well])
    # #box_and_boundary = factory.group([box_drilled])
    #factory.synchronize()

    #b_box_drilled = box_drilled.get_boundary()

    #cb_box_drilled = cb_box_drilled.copy()
    #b_box = b_box_drilled.copy().cut([b_left_well.copy(), b_right_well.copy()]).set_region(".outer_box")
    #factory.remove_duplicate_entities()
    #b_left_well = b_box_drilled.copy().intersect(b_left_well.copy()).set_region(".left_well")

    #b_right_well = b_box_drilled.copy().intersect(b_right_well.copy()).set_region(".right_well")
    #factory.remove_duplicate_entities()
    #b_box_fr_all = factory.group([b_box, b_left_well, b_right_well])
    #b_box_fr_all = factory.group([b_box])
    #
    # # fracture boundaries
    # b_fractures = factory.group(cut_fractures.get_boundary_per_region())
    # b_fractures_box = b_fractures.copy().intersect(b_box.copy()).prefix_regions(".box_")
    # b_fr_left_well = b_fractures.copy().intersect(b_left_well.copy()).prefix_regions(".left_well_")
    # b_fr_right_well = b_fractures.copy().intersect(b_right_well.copy()).prefix_regions(".right_well_")
    # b_cut_fractures = factory.group([b_fractures_box, b_fr_left_well, b_fr_right_well])

    #mesh_groups = [box_fr, b_box_fr_all, cut_fractures, b_cut_fractures]
    #mesh_groups = [box_drilled, fractures_group, b_box_fr_all]

    mesh_groups = [box_all_fr, fractures_fr]

    factory.remove_duplicate_entities()

    factory.keep_only(*mesh_groups)

    factory.write_brep()









    min_el_size = well_radius
    fracture_el_size = box_size / 20
    fracture_size = 1000
    max_el_size = box_size / 10

    # Doesn't work due to bug in occ.setMeshSize
    fractures_fr.set_mesh_step(100)

    fracture_el_size = field.constant(100, 10000)
    #frac_el_size_only = field.restrict(fracture_el_size, fractures_group, add_boundary=True)
    #field.set_mesh_step_field(frac_el_size_only)


    mesh = gmsh.MeshOptions()
    mesh.ToleranceInitialDelaunay = 0.01
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = True
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCurvePoints = 5


    factory.make_mesh(mesh_groups, eliminate=False)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)

    factory.show()















if __name__ == "__main__":
    generate_mesh()
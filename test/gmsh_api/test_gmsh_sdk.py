import numpy as np
import gmsh
import pytest

"""
Auxiliary tests of the GMSH SDK API.
Mainly tests to get insight how individual operations work and reproduce possible problems.
"""

@pytest.mark.skip
def generate_mesh():
    gmsh.initialize()
    file_name = "box_wells"
    gmsh.model.add(file_name)

    box = gmsh.model.occ.addBox(-1000, -1000, -1000, 2000, 2000, 2000)
    rec1 = gmsh.model.occ.addRectangle(-800, -800, 0, 1600, 1600)
    rec2 = gmsh.model.occ.addRectangle(-1200, -1200, 0, 2400, 2400)
    rec1_dt = (2, rec1)
    rec2_dt = (2, rec2)
    gmsh.model.occ.rotate([rec2_dt], 0,0,0,  0,1,0, np.pi/2)
    rectangle, map = gmsh.model.occ.fragment([rec1_dt, rec2_dt], [])


    box = [(3,box)]


    box_copy = gmsh.model.occ.copy(box)
    dim_tags, map = gmsh.model.occ.intersect(rectangle, box_copy)
    gmsh.model.occ.synchronize()

    dim_tags_copy = gmsh.model.occ.copy(dim_tags)
    box_copy = gmsh.model.occ.copy(box)
    box_cut, map = gmsh.model.occ.fragment(box_copy, dim_tags_copy)
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()


    b = gmsh.model.addPhysicalGroup(3, [tag for dim, tag in box_cut])
    gmsh.model.setPhysicalName(3, b, "box")

    rect = gmsh.model.addPhysicalGroup(2, [tag for dim, tag in dim_tags])
    gmsh.model.setPhysicalName(2, rect, "rectangle")

    bc_tags = gmsh.model.getBoundary(dim_tags, combined=False, oriented=False)
    bc_nodes = gmsh.model.getBoundary(dim_tags, combined=False, oriented=False, recursive=True)
    bc_box_nodes = gmsh.model.getBoundary(box_cut, combined=False, oriented=False, recursive=True)
    bc_rect = gmsh.model.addPhysicalGroup(1, [tag for dim, tag in bc_tags])
    gmsh.model.setPhysicalName(1, bc_rect, ".rectangle")
    gmsh.model.occ.setMeshSize(bc_nodes, 50)




    #gmsh.model.mesh.embed(2, [tag for dim, tag in dim_tags], 3, box_copy[0][1])
    #factory.synchronize()



    model = gmsh.model

    # generate mesh, write to file and output number of entities that produced error
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 100)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 300)

    gmsh.write(file_name + '.brep')
    gmsh.write(file_name + '.geo')
    model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()
    bad_entities = model.mesh.getLastEntityError()
    print(bad_entities)
    gmsh.write(file_name + ".msh")
    gmsh.fltk.run()
    gmsh.finalize()

    return len(bad_entities)






@pytest.mark.skip
def test_mesh_size():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    file_name = "rectangle"
    gmsh.model.add(file_name)

    p0 = gmsh.model.occ.addPoint(-800, -800, 0, meshSize=200)
    p1 = gmsh.model.occ.addPoint(800, -800, 0, meshSize=200)
    p2 = gmsh.model.occ.addPoint(800, 800, 0, meshSize=200)
    p3 = gmsh.model.occ.addPoint(-800, 800, 0, meshSize=200)
    point_tags = [p0, p1, p2, p3]
    lines = [gmsh.model.occ.addLine(point_tags[i - 1], point_tags[i]) for i in range(4)]
    cl = gmsh.model.occ.addCurveLoop(lines)

    rec1 = gmsh.model.occ.addPlaneSurface([cl])

    #rec1 = gmsh.model.occ.addRectangle(-800, -800, 0, 1600, 1600)

    gmsh.model.occ.synchronize()
    nodes = gmsh.model.getBoundary([(2, rec1)], combined=False, oriented=False, recursive=True)
    print(nodes)
    p_dimtags = [(0, tag) for tag in point_tags]
    gmsh.model.occ.setMeshSize(p_dimtags, 50)
    gmsh.model.occ.synchronize()
    # generate mesh, write to file and output number of entities that produced error
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 50)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 200)


    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.removeDuplicateNodes()
    bad_entities = gmsh.model.mesh.getLastEntityError()
    print(bad_entities)
    gmsh.fltk.run()
    gmsh.finalize()



def test_tolerances():
    """
    Test effect of varous tolerance options.
    Goal: Robust meshing of ocmplex fracture networks embedded in a 3d domains.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.option.setNumber("Geometry.Tolerance", 10)
    #gmsh.option.setNumber("Geometry.ToleranceBoolean", 10)

    box = gmsh.model.occ.addBox(0, 0, 0, 1000, 1000, 1000)
    box = (3, box)
    names = ["box"]
    rectangeles = []
    for dist in [100, 50, 20, 10]:
        rect = gmsh.model.occ.addRectangle(0, 0, dist, 1000, 1000)
        rectangeles.append((2, rect))
        names.append("rect_{}".format(dist))

    gmsh.model.occ.synchronize()
    dimtags, map = gmsh.model.occ.fragment([box] + rectangeles, [])
    gmsh.model.occ.synchronize()
    for new_for_orig, name in zip(map, names):
        print(name, new_for_orig)
        if not new_for_orig:
            continue
        dim = new_for_orig[0][0]
        group_id = gmsh.model.addPhysicalGroup(dim, [tag for dim,tag in new_for_orig])
        gmsh.model.setPhysicalName(dim, group_id, name)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", 1)
    gmsh.option.setNumber("Mesh.AnisoMax", 0.05)


    #gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 100)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 200)
    gmsh.model.mesh.generate(3)
    gmsh.write("test_tolerances.msh")
    gmsh.fltk.run()
    gmsh.finalize()





#@pytest.mark.skip
def test_boundary_fragment_inconsistency():
    """
    Problem: When fragmenting a boundary of an entity the resulting map may be wrong.
    Possible workaround: Mention the boundary before the entity or fragment only the entity and
    select the new boundary after that using original boundary.
    """
    def make_geometry():
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # cube
        box = gmsh.model.occ.addBox(-1000, -1000, -1000, 2000, 2000, 2000)
        # two "wells" extending above and bellow
        left_hole = gmsh.model.occ.addBox(-500, 0, -1200, 30, 30, 2400)
        right_hole = gmsh.model.occ.addBox(500, 0, -1200, 30, 30, 2400)
        gmsh.model.occ.synchronize()

        box = (3, box)
        b_box = gmsh.model.getBoundary([box], oriented=False)

        l_hole = (3, left_hole)
        r_hole = (3, right_hole)
        b_lhole = gmsh.model.getBoundary([l_hole], oriented=False)
        b_rhole = gmsh.model.getBoundary([r_hole], oriented=False)

        box_names = ["left", "right", "front", "back", "bottom", "top"]
        b_box_names = dict(zip(b_box, box_names))
        b_lhole_names = dict(zip(b_lhole, ["lhole_" + n for n in box_names]))
        b_rhole_names = dict(zip(b_rhole, ["rhole_" + n for n in box_names]))
        names = b_box_names
        names.update(b_lhole_names)
        names.update(b_rhole_names)

        holes = [l_hole, r_hole]
        b_holes = b_lhole + b_rhole
        return b_box, holes, b_holes, names


    def list_and_mesh(orig_shapes, orig_map, names, filename):
        for om, shape in zip(orig_map, orig_shapes):
            for dim, tag in om:
                group_id = gmsh.model.addPhysicalGroup(dim, [tag])
                gmsh.model.setPhysicalName(dim, group_id, "({}, {})".format(dim, tag))
            #named = [ (dimtag, names.get(dimtag, 'None')) for dimtag in om ]
            print("Shape: {:7s} {:13s}    {}".format(str(shape), names.get(shape, 'None'), om))

        gmsh.model.mesh.generate(3)
        gmsh.write(filename)
        gmsh.finalize()


    print("\n## fragment returns wrong map.")
    b_box, holes, b_holes, names = make_geometry()
    # Bad order, boundary is destroyed before it can be used as tool
    tools = holes + b_holes
    orig_shapes = b_box + tools
    dimtags, orig_map = gmsh.model.occ.fragment(b_box, tools)
    gmsh.model.occ.synchronize()
    list_and_mesh(orig_shapes, orig_map, names, filename="model_wrong_ids.msh")


    print("\n## fragment returns right map.")
    b_box, holes, b_holes, names = make_geometry()
    # Good order, fragment by boundary first seems to work
    tools = b_holes + holes
    orig_shapes = b_box + tools
    dimtags, orig_map = gmsh.model.occ.fragment(b_box, tools)
    gmsh.model.occ.synchronize()
    list_and_mesh(orig_shapes, orig_map, names, filename="model_right_ids.msh")

"""
# main box
Shape: (2, 1)  left             [(2, 1)]
Shape: (2, 2)  right            [(2, 2)]
Shape: (2, 3)  front            [(2, 3)]
Shape: (2, 4)  back             [(2, 4)]
Shape: (2, 5)  bottom           [(2, 39), (2, 9), (2, 25)]
Shape: (2, 6)  top              [(2, 40), (2, 15), (2, 31)]

# left hole
Shape: (3, 2)  None             [(3, 2), (3, 3), (3, 4)]
# right hole
Shape: (3, 3)  None             [(3, 5), (3, 6), (3, 7)]

Shape: (2, 7)  lhole_left       [(2, 7), (2, 13), (2, 18)]      # bottom, middle, top
Shape: (2, 8)  lhole_right      [(2, 12), (2, 17), (2, 22)]     # bottom, middle, top
Shape: (2, 9)  lhole_front      [(2, 8), (2, 14), (2, 19)]      # bottom, middle, top
Shape: (2, 10) lhole_back       [(2, 10), (2, 16), (2, 21)]     # bottom, middle, top
Shape: (2, 11) lhole_bottom     [(2, 11)]                       # BUG should be (2, 9)   is bottom cup of left hole out of the main box
Shape: (2, 12) lhole_top        [(2, 12)]                       # BUG should be (2, 15)  is already bottom part of lhole right (confirmed on the mesh)

Shape: (2, 13) rhole_left       [(2, 23), (2, 29), (2, 34)]
Shape: (2, 14) rhole_right      [(2, 28), (2, 33), (2, 38)]
Shape: (2, 15) rhole_front      [(2, 24), (2, 30), (2, 35)]
Shape: (2, 16) rhole_back       [(2, 26), (2, 32), (2, 37)]
Shape: (2, 17) rhole_bottom     [(2, 17)]                       # BUG should be (2, 25)  is already right side of the left hole
Shape: (2, 18) rhole_top        [(2, 18)]                       # BUG should be (2, 31)  is already top left side of the left hole
"""


"""
This seems to be correct.
Shape: (2, 1)  left             [(2, 1)]
Shape: (2, 2)  right            [(2, 2)]
Shape: (2, 3)  front            [(2, 3)]
Shape: (2, 4)  back             [(2, 4)]
Shape: (2, 5)  bottom           [(2, 47), (2, 21), (2, 35)]
Shape: (2, 6)  top              [(2, 48), (2, 26), (2, 40)]

Shape: (2, 7)  lhole_left       [(2, 19), (2, 24), (2, 29)]
Shape: (2, 8)  lhole_right      [(2, 23), (2, 28), (2, 32)]
Shape: (2, 9)  lhole_front      [(2, 20), (2, 25), (2, 30)]
Shape: (2, 10) lhole_back       [(2, 22), (2, 27), (2, 31)]
Shape: (2, 11) lhole_bottom     [(2, 11)]                       
Shape: (2, 12) lhole_top        [(2, 12)]

Shape: (2, 13) rhole_left       [(2, 33), (2, 38), (2, 43)]
Shape: (2, 14) rhole_right      [(2, 37), (2, 42), (2, 46)]
Shape: (2, 15) rhole_front      [(2, 34), (2, 39), (2, 44)]
Shape: (2, 16) rhole_back       [(2, 36), (2, 41), (2, 45)]
Shape: (2, 17) rhole_bottom     [(2, 17)]
Shape: (2, 18) rhole_top        [(2, 18)]

Shape: (3, 2)  None             [(3, 2), (3, 3), (3, 4)]
Shape: (3, 3)  None             [(3, 5), (3, 6), (3, 7)]
"""
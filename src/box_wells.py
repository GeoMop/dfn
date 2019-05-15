import numpy as np
from gmsh_api import gmsh


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


def test_mesh_size2():
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
    gmsh.model.occ.synchronize()
    p_dimtags = [(0, tag) for tag in point_tags]
    gmsh.model.occ.setMeshSize(p_dimtags, 50)
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 50)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 200)
    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()
    gmsh.finalize()


def test_box_wells():
    gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 1)
    # file_name = "rectangle"
    # gmsh.model.add(file_name)

    box = gmsh.model.occ.addBox(-1000, -1000, -1000, 2000, 2000, 2000)
    well_left = gmsh.model.occ.addBox(-500, 0, -1200, 30, 30, 2400)
    well_right = gmsh.model.occ.addBox(500, 0, -1200, 30, 30, 2400)
    gmsh.model.occ.synchronize()
    box = (3, box)
    wells = [(3, well_left), (3, well_right)]
    b_box = gmsh.model.getBoundary([box], oriented=False)
    b_wells = gmsh.model.getBoundary(wells, oriented=False)

    dimtags, orig_map = gmsh.model.occ.fragment([box] + b_box, wells + b_wells)
    print("dimtags:\n", dimtags)
    print("map:\n", orig_map)

    gmsh.finalize()

"""
map:
 [
 [(3, 1), (3, 2), (3, 3)],
 
 [(2, 1)],
 [(2, 2)],
 [(2, 3)],
 [(2, 4)],
 [(2, 5), (2, 16), (2, 18)],
 [(2, 3), (2, 15), (2, 17)],
 
 [(3, 4), (3, 2), (3, 5)],
 [(3, 6), (3, 3), (3, 7)],
 
 [(2, 19), (2, 10), (2, 24)],     # left sides
 [(2, 23), (2, 8), (2, 28)],      # left sides
 [(2, 20), (2, 7), (2, 25)],      # left sides
 [(2, 21), (2, 9), (2, 27)],      # left sides
 [(2, 11)],                       # left top cap      should be = 15 !! BUG, is part of right sides
 [(2, 12)],                       # left bottom cap   should be = 16 !! BUG, is part of right sides
 
 [(2, 29), (2, 14), (2, 34)],
 [(2, 33), (2, 12), (2, 38)],
 [(2, 30), (2, 11), (2, 35)],
 [(2, 31), (2, 13), (2, 37)],
 [(2, 17)],
 [(2, 18)]]
"""

if __name__ == "__main__":
    # generate_mesh()
    #test_mesh_size2()
    test_box_wells()
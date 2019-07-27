import os
import collections
import numpy as np
from typing import Tuple
import attr


class ShapeBase:
    def __init__(self, nodes):
        self.dim = len(nodes) - 1
        self.nodes = nodes
        self._measure = None
        self._edge_vectors = None
        self._edge_lens = None
        self._gamma = None


    @property
    def edge_vectors(self):
        if self._edge_vectors is None:
            self._edge_vectors = np.array([self.nodes[b] - self.nodes[a] for a,b in  self.edges])
        return self._edge_vectors

    @property
    def edge_lens(self):
        if self._edge_lens is None:
            self._edge_lens = [np.linalg.norm(edge_vec) for edge_vec in self.edge_vectors]
        return self._edge_lens



    def small_edge_ratio(self):
        min_edge = np.min(self.edge_lens)
        max_edge = max(1e-300, np.max(self.edge_lens))
        return min_edge / max_edge

    def flat_indicator(self):
        """
        Ratio of the min altitude / max edge
        :return:
        """
        dim = self.dim
        max_edge = max(1e-300, np.max(self.edge_lens))
        if dim == 3:
            alt = 6 * np.abs(self.measure) / max(1e-300, np.max(self.face_areas))
        else:
            alt = 2 * np.abs(self.measure) / max_edge

        #print(self.measure, regular_simplex_vol, self.edge_lens)
        return alt / max_edge



class Tetrahedron(ShapeBase):
    vtxs_faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
    # Tetra faces, index of face is the index of the opposite node.
    # face triangle normal is inner normal
    edges = [[0, 1], [0, 2], [0, 3], [2, 3], [1, 3], [1, 2]]

    def __init__(self, nodes):
        super().__init__(nodes)
        self._face_normals = None
        self._face_areas = None

    @property
    def measure(self):
        if self._measure is None:
            self._measure = np.linalg.det(self.nodes[1:, :] - self.nodes[0, :]) / 6
        return self._measure


    @property
    def face_normals(self):
        # nonunit face normals
        if self._face_normals is None:
            self._face_normals = np.array([-Triangle(self.nodes[face_vtxs]).normal() for face_vtxs in self.vtxs_faces])
        return self._face_normals

    @property
    def face_areas(self):
        if self._face_areas is None:
            self._face_areas = [Triangle(self.nodes[face_vtxs]).measure for face_vtxs in self.vtxs_faces]
        return self._face_areas


    def smooth_grad_error_indicator(self):
        faces = self.face_areas
        e_lens = self.edge_lens
        e_faces = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]] # faces joining the edge
        sum_pairs = max(1e-300, np.sum([faces[i] * faces[j] * elen ** 2 for (i,j),elen in zip(e_faces, e_lens)]))
        regular = (2.0 * np.sqrt(2.0 / 3.0) / 9.0)
        quality = np.abs(self.measure) * (np.sum(faces) / sum_pairs) ** (3.0/4.0) / regular
        return quality

    def gmsh_gamma(self):
        if self._gamma is None:
            faces = self.face_areas
            V = np.abs(self.measure)
            # inradius
            r = 3*V / sum(faces)
            # circum radius
            a,b,c,A,B,C = self.edge_lens
            x_area = (a*A + b*B + c*C)*(a*A + b*B - c*C)*(a*A - b*B + c*C)*(-a*A + b*B + c*C)
            assert x_area > 0, x_area
            R = np.sqrt(x_area) / 24 / max(1e-300, V)
            self._gamma = 3 * r/max(1e-300, R)
        return self._gamma




    def common_normal(self):
        """
        Find a vector that is least-square normal to all edges.
        :return:
        """
        u, s, v = np.linalg.svd(self.edge_vectors)
        approx_common_normal = v[-1, :]
        return approx_common_normal


    def skew_line_dist(self):
        a_vec = self.nodes[1] - self.nodes[0]
        b_vec = self.nodes[3] - self.nodes[2]
        normal = np.cross(a_vec, b_vec)
        normal /= np.linalg.norm(normal)
        dist = normal @ (self.nodes[0] - self.nodes[2])
        return dist, normal



class Triangle(ShapeBase):
    vtxs_faces = [[0, 1, 2]]
    edges = [[1, 2], [2, 0], [0, 1]]
    # Triangle edges, index of edge is the index of the opposite node.
    # edges of the single vertex: one comes in one comes out

    def __init__(self, nodes):
        super().__init__(nodes)

    def normal(self):
        return np.cross(self.nodes[1] - self.nodes[0], self.nodes[2] - self.nodes[0])

    @property
    def measure(self):
        if not self._measure:
            self._measure = np.linalg.norm(np.cross(self.nodes[2] - self.nodes[0], self.nodes[1] - self.nodes[0])) / 2
        return self._measure


    def smooth_grad_error_indicator(self):
        e_lens = self.edge_lens
        prod = max(1e-300, (np.prod(e_lens)) ** (2.0 / 3.0))
        quality = 4 / np.sqrt(3) * np.abs(self.measure) / prod
        return quality


    def gmsh_gamma(self):
        if self._gamma is None:
            a,b,c = self.edge_lens
            # inradius
            s = (a + b + c)/2
            #r = np.sqrt(s*(s-a)*(s-b)*(s-c)) / s
            #R = a*b*c / 4 / r / s
            #gamma =  2 * r / R
            alt_gamma = 8 * (s/a-1)*(s/b-1)*(s/c-1)
            #assert np.isclose(gamma, alt_gamma)
            self._gamma =  alt_gamma
        return self._gamma




class Line(ShapeBase):
    edges = [[0,1]]

    def __init__(self, nodes):
        super().__init__(nodes)

    @property
    def measure(self):
        return  self.edge_lens[0]



class Point:
    edges = []

    def __init__(self, nodes):
        self.dim = 0
        self.nodes = nodes

    @property
    def measure(self):
        return  0.0

    @property
    def edge_lens(self):
        return []



@attr.s(auto_attribs=True)
class Element:
    eid: int
    type: int
    tags: Tuple[int, int, int]
    node_ids: np.array
    shape: ShapeBase


class HealMesh:

    @staticmethod
    def read_mesh(mesh_file, node_tol=0.0001):
        import gmsh_io
        return HealMesh(gmsh_io.GmshIO(mesh_file), mesh_file, node_tol)


    def __init__(self, mesh_io, mesh_file="mesh.msh", node_tol=0.0001):
        self.mesh = mesh_io
        self.make_node_to_el()
        self.max_ele_id = max(self.mesh.elements.keys())
        self.max_node_id = max(self.mesh.nodes.keys())

        base, ext = os.path.splitext(mesh_file)
        self.healed_mesh_name = base + "_healed.msh"

        self.modified_elements = set()
        aabb_min = np.full(3, +np.inf)
        aabb_max = np.full(3, -np.inf)
        for n in self.mesh.nodes:
            aabb_max = np.maximum(aabb_max, n)
            aabb_min = np.minimum(aabb_min, n)
        diam = np.max(aabb_max - aabb_min)
        self._abs_node_tol = node_tol * diam
        print("diam: ", self._abs_node_tol)


    def make_node_to_el(self):
        # make node -> element map
        self._history = collections.defaultdict(list)
        self.node_els = collections.defaultdict(set)
        for eid, e in self.mesh.elements.items():
            type, tags, node_ids = e
            for n in node_ids:
                self.node_els[n].add(eid)
            self._history[eid].append(('A', node_ids))


    # modification methods
    # touched elements are collected

    def reset_modified(self):
        modif = self.modified_elements
        self.modified_elements = set()
        return modif

    def add_element(self, ele):
        e = self._make_element(None, ele)
        if self.is_duplicate_el(e):
            return

        self.max_ele_id += 1
        if e.shape.measure < 0.1:
            print("    add el:", self.max_ele_id, e.shape.measure)
            #if shape.measure < 0:
            #    node_ids[0], node_ids[1] = node_ids[1], node_ids[0]
            #    ele = (type, tags, node_ids)

        self.mesh.elements[self.max_ele_id] = ele
        for nid in e.node_ids:
            self.node_els[nid].add(self.max_ele_id)
        self._history[self.max_ele_id].append(('A', e.node_ids))
        self.modified_elements.add(self.max_ele_id)
        return self.max_ele_id

    def add_node(self, node):
        self.max_node_id += 1
        self.mesh.nodes[self.max_node_id] = node
        self.node_els[self.max_node_id] = set()
        return self.max_node_id

    def move_node(self, nid, node):
        # print("    orig node: ", self.mesh.nodes[nid])
        # print("    new  node: ", node)
        # print("    diff     : ", node - np.array(self.mesh.nodes[nid]))
        orig = [self._make_element(eid) for eid in self.active(self.node_els[nid])]
        self.mesh.nodes[nid] = node
        new_signs = [self._make_element(eid) for eid in self.active(self.node_els[nid])]
        for oe, ne in zip(orig, new_signs):
            if ne.shape.measure < 0 or oe.shape.measure < 0:
                print("    move out: ", ne.eid, oe.shape.measure, ne.shape.measure)
        self.modified_elements.update(self.node_els[nid])


    def merge_node(self, rm_node, target_node):
        """
        Merge node rm_node with node 'target_node'.
        Move both points to the average position.

        :param rm_node:
        :param target_node:
        :param new_pos:
        :return:
        """

        assert rm_node != target_node

        # remove elements containing the edge (a,b)
        merge_nodes = [rm_node, target_node]
        for i_edge_el in self.common_elements(merge_nodes):
            self.remove_element(i_edge_el)

        # average nodes, remove the second one
        node_avg = np.average([np.array(self.mesh.nodes[n]) for n in merge_nodes], axis=0)
        self.move_node(target_node, node_avg)

        # substitute target_node for the rm_node
        for i_el in self.common_elements([rm_node]):
            el_type, el_tags, el_node_ids = self.mesh.elements[i_el]
            el_node_ids = [target_node if n == rm_node else n for n in el_node_ids]
            self.remove_element(i_el)
            self.add_element((el_type, el_tags, el_node_ids))

        self.remove_node(rm_node)
        return target_node


    def remove_node(self, nid):
        assert len(self.node_els[nid]) == 0, self.node_els[nid]
        del self.mesh.nodes[nid]
        del self.node_els[nid]

    def remove_element(self, eid):
        type, tags, node_ids = self.mesh.elements[eid]
        for nid in set(node_ids):
            try:
                self.node_els[nid].remove(eid)
            except KeyError as e:
                print("Failed for ", nid)
                print(self._history[eid])
                raise e
        self._history[eid].append(('R', node_ids))
        del self.mesh.elements[eid]

    def is_duplicate_el(self, ele):
        # check that there isn't the same element already
        duplicate_els = [eid for eid in self.common_elements(ele.node_ids) if self.mesh.elements[eid][0] == ele.type]
        return duplicate_els



    def common_elements(self, node_ids, max=1000):
        """
        Generator of active elements common to given nodes.
        :param node_ids:
        :return:
        """
        node_sets = [self.node_els[n] for n in node_ids]
        elements = set.intersection(*node_sets)
        if len(elements) > max:
            print("Too many connected elements:", len(elements), " > ", max)
            for eid in elements:
                type, tags, node_ids = self.mesh.elements[eid]
                print("  eid: ", eid, node_ids)
        return self.active(elements)

    ##########

    def active(self, element_iterable):
        for eid in element_iterable:
            if eid in self.mesh.elements:
                yield eid

    def write(self, meshfile=None):
        """
        Write current mesh state into given of default mesh file.
        :param other_name:
        :return:
        """
        if meshfile is None:
            meshfile = self.healed_mesh_name
        with open(meshfile, "w") as f:
            self.mesh.write_ascii(f)





    def _make_element(self, eid, e_tuple=None):
        if e_tuple is None:
            e_tuple = self.mesh.elements[eid]
        type, tags, node_ids = e_tuple
        nodes = np.array([self.mesh.nodes[n] for n in node_ids], dtype=float)
        shape_class = {1:Point, 2:Line, 3:Triangle, 4:Tetrahedron}
        shape = shape_class[len(node_ids)](nodes)
        return Element(eid, type, tags, np.array(node_ids, dtype=int), shape)


    quality_methods = {'flow_stats': 'smooth_grad_error_indicator', 'gamma_stats': 'gmsh_gamma'}
    def quality_statistics(self, bad_el_tol=0.01):
        """
        Vector of number of elements in quality bins:
        (1, 0.5), (0.5, 0.25), ...
        :return:
        """
        methods = self.quality_methods
        bad_els = {method: [] for method in methods}
        bins = 2.0 ** (np.arange(-15, 1))
        bins = np.concatenate((bins, [np.inf]))
        histogram = {method: np.zeros_like(bins, dtype=int) for method in methods}
        for eid in self.mesh.elements:
            e = self._make_element(eid)
            if e.shape.dim > 1:
                for method, hist in histogram.items():
                    quality = getattr(e.shape, methods[method])()
                    if quality <= bad_el_tol:
                        bad_els[method].append(eid)
                    first_smaller = np.argmax(quality < bins)
                    hist[first_smaller] += 1
        return histogram, bins, bad_els


    def print_stats(self, histogram, bins, name):
        print(name)
        line = ["{:10.2e}".format(v) for v in bins]
        print(" < ".join(line))
        line = ["{:10d}".format(v) for v in histogram]
        print(" | ".join(line))


    def stats_to_yaml(self, filename, el_tol=0.01):
        methods = self.quality_methods
        hist, bins, bad_els = self.quality_statistics(bad_el_tol=el_tol)
        output = {}
        for name, method in methods.items():
            output[name] = dict(hist=hist[name].tolist(), bins=bins.tolist(), bad_elements=bad_els[name], bad_el_tol=el_tol)
        import yaml
        with open(filename, "w") as f:
            yaml.dump(output, f)


    def heal_mesh(self, gamma_tol=0.02, fraction_of_new_els=2):

        self.gamma_tol = gamma_tol
        orig_n_el = self.max_ele_id
        el_to_check = collections.deque(self.mesh.elements.keys())
        modified = []
        while el_to_check:
            el_to_check.extend(modified)
            modified = []
            # prevent infinite loop
            if self.max_ele_id > fraction_of_new_els * orig_n_el:
                break
            eid = el_to_check.pop()
            if eid in self.mesh.elements:
                ele = self._make_element(eid)
                modified = self._check_dupl_nodes(ele)
                if modified: continue
                modified = self._check_degen_nodes(ele)
                if modified: continue
                modified = self._check_duplicate_element(ele)
                if modified: continue

                if ele.shape.dim < 2: continue
                if ele.shape.gmsh_gamma() > self.gamma_tol: continue
                modified = self._check_flat_tetra(ele)
                if modified: continue
                modified = self._check_flat_triangle(ele)
                if modified: continue
                modified = self._check_small_edge(ele)
                if modified: continue



    def _check_dupl_nodes(self, ele):
        if ele.shape.dim == 0:
            return []
        for edge in ele.shape.edges:
            edge_ids = ele.node_ids[edge]
            if edge_ids[0] == edge_ids[1]:
                print("eid: {} heal degen node: {}, removed el.".format(ele.eid, edge_ids[0]))
                self.remove_element(ele.eid)
                return [e for e in self.active(self.node_els[edge_ids[0]])]
        return []


    def _check_duplicate_element(self, ele):
        dupl = self.is_duplicate_el(ele)
        if len(dupl) > 1:
            print("eid: {} heal dupl els: {}".format(ele.eid, dupl))
            for other_eid in dupl:
                if other_eid != ele.eid:
                    self.remove_element(other_eid)
            return [ele.eid]
        return []

    def _check_degen_nodes(self, ele):
        """
        Merge close nodes. Perform at most one merge per element.
        :param eid:
        :return:
        """
        if ele.shape.dim == 0:
            return []
        elen = np.min(ele.shape.edge_lens)

        if elen < self._abs_node_tol:
            i_min_edge = np.argmin(ele.shape.edge_lens)
            edge = ele.shape.edges[i_min_edge]
            edge_ids = ele.node_ids[edge]
            print("eid: {} heal close nodes ({}), merge: {}".format(ele.eid, elen, edge_ids ))
            self.merge_node(*edge_ids)
        return self.reset_modified()


    def _check_small_edge(self, ele):
        """
        Check element, possibly contract its shortest edge.
        :param mesh:
        :param eid:
        :return: List of still existing but changed elements.
        """
        quality = ele.shape.small_edge_ratio()


        merge_vtxs = ele.shape.edges[np.argmin(ele.shape.edge_lens)]
        merge_nodes = ele.node_ids[merge_vtxs]
        print("eid: {} heal short edge ({}d): {} merge nodes: {}".format(ele.eid, ele.shape.dim, quality, merge_nodes))
        self.merge_node(*merge_nodes)
        return self.reset_modified()


    def _check_flat_triangle(self, ele):
        if ele.shape.dim < 2: return []
        if ele.shape.dim == 3:
            t_shapes = [Triangle(ele.shape.nodes[face]) for face in ele.shape.vtxs_faces]
            face_gammas = [tria.gmsh_gamma() for tria in t_shapes]
            i_min_face = np.argmin(face_gammas)
            tria_shape = t_shapes[i_min_face]
            tria_node_ids = ele.node_ids[ele.shape.vtxs_faces[i_min_face]]
        else:
            tria_shape = ele.shape
            tria_node_ids = ele.node_ids
        # if np.min(tria_shape.edge_lens) / np.max(tria_shape.edge_lens) < 0.05:
        #     return []


        i_max_edge = np.argmax(tria_shape.edge_lens)

        # remove elements connected to the degenerate triangle
        for el_id in self.common_elements(tria_node_ids):
            self.remove_element(el_id)

        # project node to longest edge
        max_edge = tria_shape.edges[i_max_edge]
        max_edge_u = tria_shape.edge_vectors[i_max_edge]
        len_max_edge = tria_shape.edge_lens[i_max_edge]
        i_x_edge = max_edge[1] # common edge to the vtx 0 of the max_edge
        # edges of a vtx have different sign
        x_edge_u = -tria_shape.edge_vectors[i_x_edge]
        t = x_edge_u @ max_edge_u / len_max_edge
        tt = t / len_max_edge
        assert 0 <= tt <= 1, tt

        if tt < 0.05 or 1-tt < 0.05:
            return []
        print("eid: {} heal small triangle ({}), nodes: {} elens: {}".format(ele.eid, ele.shape.gmsh_gamma(), ele.node_ids, tria_shape.edge_lens))

        projected = tria_shape.nodes[max_edge[0]] + tt * max_edge_u
        nid_proj = tria_node_ids[i_max_edge]

        self.move_node(nid_proj, projected)

        # split elements along longest edge, ...
        edge_nids = tria_node_ids[max_edge]
        for el_id in self.common_elements(edge_nids):
            el_type, el_tags, el_node_ids = self.mesh.elements[el_id]
            if edge_nids[0] not in el_node_ids or edge_nids[1] not in el_node_ids:
                print("  Warn: ", "Missing edge", "el_id: ", el_id, el_node_ids)
                continue
            self.remove_element(el_id)
            for nid in edge_nids:
                i_pos = el_node_ids.index(nid)
                new_node_ids = el_node_ids.copy()
                new_node_ids[i_pos] = nid_proj
                self.add_element((el_type, el_tags, new_node_ids))

        return self.reset_modified()


    # def _heal_degenerate_flat(self, eid, flat_nodes, loc_points):
    #     """
    #     :param eid:
    #     :param flat_nodes:
    #     :param loc_points: (isec_node, oposite_to_isec_mode, other two points)
    #     :return:
    #     """
    #     type, tags, node_ids, shape = self._make_shape(self.mesh.elements[eid])
    #     print("  flat degen case.")
    #
    #     # move nodes
    #     for nid, node in zip(node_ids, flat_nodes):
    #         self.move_node(nid, node)
    #     self.remove_element(eid)
    #
    #     c_faces = [ # 0=degen node, 1=opposite, 2,3=remaining nodes
    #         [0, 1, 2], # half
    #         [0, 1, 3], # half
    #         [2, 3, 1], # outer, split
    #         [2, 3, 0]  # destroy
    #     ]
    #     # split attached elements
    #     face_node_ids = [[node_ids[loc_points[pt]] for pt in cf] for cf in c_faces]
    #     # keep first two faces
    #     # remove forth one
    #     for el_id in self.common_elements(face_node_ids[3]):
    #         self.remove_element(el_id)
    #
    #     # split the third face
    #     new_node_id = node_ids[loc_points[0]]
    #     face_nids = face_node_ids[2]
    #     split_edge_nid0 = face_nids[0]
    #     split_edge_nid1 = face_nids[1]
    #     for el_id in self.common_elements(face_nids):
    #
    #         if split_edge_nid0 not in el_node_ids or split_edge_nid1 not in el_node_ids:
    #             print("  Warn: ", "Missing edge", "el_id: ", el_id, el_node_ids)
    #             continue
    #         self.remove_element(el_id)
    #         loc_n0 = el_node_ids.index(split_edge_nid0)
    #         loc_n1 = el_node_ids.index(split_edge_nid1)
    #         for pos_new in [loc_n0, loc_n1]:
    #             new_node_ids = el_node_ids.copy()
    #             new_node_ids[pos_new] = new_node_id
    #             self.add_element((el_type, el_tags, new_node_ids))
    #
    #     return self.reset_modified()







    def _check_flat_tetra(self, ele):
        """
        Check that element is flat. Use angles between face normals
        :param eid:
        :param quality_tol:
        :return:
        """
        if ele.shape.dim != 3: return []
        # check that no face is to small
        face_gammas = [Triangle(ele.shape.nodes[face]).gmsh_gamma() for face in ele.shape.vtxs_faces]

        if min(face_gammas) < np.sqrt(self.gamma_tol):
            return []

        # quality = ele.shape.flat_indicator()
        # gamma = ele.shape.gmsh_gamma()
        # if quality > quality_tol:
        #     if gamma < quality_tol:
        #         print("  eid: {} flatness: {} > gamma: {}".format(ele.eid, quality, gamma))
        #     return []
        print("eid: {} heal flat el. gamma: {} nodes: {}".format(ele.eid, ele.shape.gmsh_gamma(), ele.node_ids))

        # approx normal to all edges
        common_normal = ele.shape.common_normal()
        common_normal /= np.linalg.norm(common_normal)
        # project nodes to the approx plane
        normal_component = ele.shape.nodes @ common_normal
        avg = np.average(normal_component)
        flat_nodes = ele.shape.nodes - (normal_component - avg)[:, None] * common_normal[None, :]
        assert np.isclose(np.linalg.det(flat_nodes[1:, :] - flat_nodes[0, :]), 0.0)

        # cases for projected shapes
        cos_face_norm = ele.shape.face_normals @ common_normal
        n_pos = np.sum(cos_face_norm > 0.0)
        if n_pos == 2:
            # quad flat case
            vtx_faces = np.array(ele.shape.vtxs_faces)
            i_pos_faces = np.arange(4)[cos_face_norm > 0]
            pos_faces = vtx_faces[i_pos_faces, :]
            pos_edge_vtxs = list(set.intersection(*[set(vtxs) for vtxs in pos_faces]))
            i_neg_faces = np.arange(4)[cos_face_norm <= 0]
            neg_faces = vtx_faces[i_neg_faces, :]
            neg_edge_vtxs = list(set.intersection(*[set(vtxs) for vtxs in neg_faces]))
            return self._heal_quad_flat_case(ele, flat_nodes, pos_edge_vtxs, neg_edge_vtxs)

        elif n_pos == 1:
            outer_face = np.argmax(cos_face_norm > 0)
            return self._heal_triangle_flat_case(ele, flat_nodes, outer_face)
        elif n_pos==3:
            outer_face = np.argmax(cos_face_norm <= 0)
            return self._heal_triangle_flat_case(ele, flat_nodes, outer_face)
        else:
            assert False








    def _heal_quad_flat_case(self, ele, flat_nodes, pos_edge, neg_edge):
        """
        :param eid:
        :param flat_nodes:
        :param pos_edge:
        :param neg_edge:
        :param tol: Approximate upper bound for fraction min_tet_height / max_edge
        :return:
        """
        # compute intersection of flat edges
        pos_0, pos_1 = flat_nodes[pos_edge, :]
        pos_u = pos_1 - pos_0
        neg_0, neg_1 = flat_nodes[neg_edge, :]
        neg_u = neg_1 - neg_0
        M = np.stack((pos_u, -neg_u), axis=1)
        sub_rows = [[0,1], [0, 2], [1,2]]
        sub_mat = [M[rows, :] for rows in sub_rows]
        im = np.argmax([np.abs(np.linalg.det(subM)) for subM in sub_mat])
        M = sub_mat[im]
        rhs = (pos_0 - neg_0)[sub_rows[im]]
        pos_t, neg_t = np.linalg.solve(M, -rhs)
        isec_point = pos_0 + pos_t * pos_u

        print("  isec: ", (pos_t, neg_t))
        skew_edges = [(pos_t, pos_edge), (neg_t, neg_edge)]
        for i, (t, edge) in enumerate(skew_edges):
            edge_len = np.linalg.norm(flat_nodes[edge[1]] - flat_nodes[edge[0]])
            other_edge = skew_edges[1-i][1]

            #tria_nodes = np.stack((flat_nodes[edge[0]], flat_nodes[other_edge[0]], flat_nodes[other_edge[1]]), axis=0)
            #tria_flatness = Triangle(tria_nodes).small_edge_ratio()
            assert np.abs(t-0) > 0.05, "  flat tetra, degen side, 0 == t: {}".format(t)
            #tria_nodes = np.stack((flat_nodes[edge[1]], flat_nodes[other_edge[0]], flat_nodes[other_edge[1]]), axis=0)
            #tria_flatness = Triangle(tria_nodes).small_edge_ratio()
            assert np.abs(t-1) > 0.05, "  flat tetra, degen side, 1 == t: {}".format(t)

        assert 0 < pos_t < 1 ,(pos_t, neg_t)
        assert 0 < neg_t < 1 ,(pos_t, neg_t)


        # non-degenerate
        print("  flat quad case: ")

        # move nodes
        for nid, node in zip(ele.node_ids, flat_nodes):
            self.move_node(nid, node)

        # remove flat element
        self.remove_element(ele.eid)

        # add intersection node
        new_node_id = self.add_node(isec_point)

        # split attached elements
        node_perm = pos_edge + neg_edge # canonical nodes to real nodes
        canonical_faces = [[0, 1, 2], [0, 1, 3], [2, 3, 0], [2, 3, 1]] # first two nodes
        for c_face in canonical_faces:
            face = [node_perm[vtx] for vtx in c_face]
            # split elements connected to the face
            # intersection splits the first edge of the face
            edge_n0_id = ele.node_ids[face[0]]
            edge_n1_id = ele.node_ids[face[1]]
            #print("edge nids: ", edge_n0_id, edge_n1_id)

            # active elements connected to the face
            for el_id in self.common_elements([ele.node_ids[loc_node] for loc_node in face], max=2):
                el_type, el_tags, el_node_ids = self.mesh.elements[el_id]
                if edge_n0_id not in el_node_ids or edge_n1_id not in el_node_ids:
                    print("  Warn: ", "Missing edge", "el_id: ", el_id, el_node_ids)
                    continue
                self.remove_element(el_id)
                loc_n0 = el_node_ids.index(edge_n0_id)
                loc_n1 = el_node_ids.index(edge_n1_id)
                for pos_new in [loc_n0, loc_n1]:
                    new_node_ids = el_node_ids.copy()
                    new_node_ids[pos_new] = new_node_id
                    self.add_element((el_type, el_tags, new_node_ids))

        return self.reset_modified()







    def _heal_triangle_flat_case(self, ele, flat_nodes, outer_face):
        """
        Case with single outer face splitted by other 3 faces.
        :param eid:
        :param shape:
        :param flat_nodes:
        :param outer_face:
        :return:
        """
        # try to project opposite node to edges
        i_inner_node = 3 - outer_face
        i_outer_nodes = Tetrahedron.vtxs_faces[outer_face]
        inner_node = flat_nodes[i_inner_node]
        outer_nodes = flat_nodes[np.arange(4) != i_inner_node]
        projections = []
        for i, edg in enumerate(Triangle.edges):
            e_nodes = outer_nodes[edg, :]
            e_vec = (e_nodes[1] - e_nodes[0])
            t_proj = (e_vec @ (inner_node - e_nodes[0])) / (e_vec @ e_vec)
            x_proj = e_vec * t_proj + e_nodes[0]
            rel_dist = np.linalg.norm(inner_node - x_proj) / np.linalg.norm(e_vec)
            projections.append((rel_dist, i, t_proj, x_proj))
        rel_dist, i, t, x = min(projections)
        assert rel_dist > 0.05, "  flat tria degen side, rel dist: {}".format(rel_dist)

        # nondegenerate triangle case, split elements connected to the outer face
        print("  flat tria case")

        # move nodes
        for nid, node in zip(ele.node_ids, flat_nodes):
            self.move_node(nid, node)
        self.remove_element(ele.eid)
        # split outer face
        vtxs_outer = Tetrahedron.vtxs_faces[outer_face]
        outer_nids = ele.node_ids[vtxs_outer]
        # active elements connected to the outer face
        for el_id in self.common_elements(outer_nids, max=2):
            el_type, el_tags, el_node_ids = self.mesh.elements[el_id]
            self.remove_element(el_id)
            for nid in outer_nids:
                i_nid_el = el_node_ids.index(nid)
                new_node_ids = el_node_ids.copy()
                new_node_ids[i_nid_el] = ele.node_ids[i_inner_node]
                self.add_element((el_type, el_tags, new_node_ids))
        return self.reset_modified()


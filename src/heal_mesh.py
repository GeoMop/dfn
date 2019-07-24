import os
import collections
import numpy as np




class Tetrahedron:
    vtxs_faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    edges = [[0, 1], [0, 2], [0, 3], [2, 3], [1, 3], [1, 2]]

    def __init__(self, nodes):
        self.dim = 3
        self.nodes = nodes
        self._measure = None
        self._face_areas = None
        self._edge_lens = None

    @property
    def measure(self):
        if not self._measure:
            self._measure = np.linalg.det(self.nodes[1:, :] - self.nodes[0, :]) / 6
        return self._measure

    @property
    def face_areas(self):
        if not self._face_areas:
            self._face_areas = [Triangle(self.nodes[face_vtxs]).measure for face_vtxs in self.vtxs_faces]
        return self._face_areas

    @property
    def edge_lens(self):
        if not self._edge_lens:
            self._edge_lens = [np.linalg.norm(self.nodes[i] - self.nodes[j]) for i,j in self.edges]
        return self._edge_lens

    def smooth_grad_error_indicator(self):
        faces = self.face_areas
        e_lens = self.edge_lens
        e_faces = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]] # faces joining the edge
        sum_pairs = max(1e-300, np.sum([faces[i] * faces[j] * elen ** 2 for (i,j),elen in zip(e_faces, e_lens)]))
        regular = (2.0 * np.sqrt(2.0 / 3.0) / 9.0)
        quality = np.abs(self.measure) * (np.sum(faces) / sum_pairs) ** (3.0/4.0) / regular
        return quality

    def gmsh_gamma(self):
        faces = self.face_areas
        V = np.abs(self.measure)
        # inradius
        r = 3*V / sum(faces)
        # circum radius
        a,b,c,A,B,C = self.edge_lens
        x_area = (a*A + b*B + c*C)*(a*A + b*B - c*C)*(a*A - b*B + c*C)*(-a*A + b*B + c*C)
        assert x_area > 0, x_area
        R = np.sqrt(x_area) / 24 / V
        return 3 * r/max(1e-300, R)


    def flat_indicator(self):
        max_edge = np.max(self.edge_lens)
        quality = np.abs(self.measure) / (max_edge **3 / 6 / np.sqrt(2))
        return quality

    def skew_line_dist(self):
        a_vec = self.nodes[1] - self.nodes[0]
        b_vec = self.nodes[3] - self.nodes[2]
        normal = np.cross(a_vec, b_vec)
        normal /= np.linalg.norm(normal)
        dist = normal @ (self.nodes[0] - self.nodes[2])
        return dist, normal

    def small_edge_ratio(self):
        min_edge = np.min(self.edge_lens)
        max_edge = max(1e-300, np.max(self.edge_lens))
        return min_edge / max_edge


class Triangle:
    edges = [(0, 1), (1, 2), (2, 0)]

    def __init__(self, nodes):
        self.dim = 2
        self.nodes = nodes
        self._measure = None
        self._edge_lens = None

    @property
    def measure(self):
        if not self._measure:
            self._measure = np.linalg.norm(np.cross(self.nodes[2] - self.nodes[0], self.nodes[1] - self.nodes[0])) / 2
        return self._measure

    @property
    def edge_lens(self):
        if not self._edge_lens:
            self._edge_lens = [np.linalg.norm(self.nodes[i] - self.nodes[j]) for i,j in self.edges]
        return self._edge_lens

    def smooth_grad_error_indicator(self):
        e_lens = self.edge_lens
        prod = max(1e-300, (np.prod(e_lens)) ** (2.0 / 3.0))
        quality = 4 / np.sqrt(3) * np.abs(self.measure) / prod
        return quality


    def gmsh_gamma(self):
        a,b,c = self.edge_lens
        # inradius
        s = (a + b + c)/2
        r = np.sqrt(s*(s-a)*(s-b)*(s-c)) / s
        R = a*b*c / 4 / r / s
        return 2 * r / R



    def small_edge_ratio(self):
        """
        Triangle can always be healed by the edge contraction so
        we include the flatness criteria as well.
        """
        max_edge = max(1e-300, np.max(self.edge_lens))
        height = np.abs(self.measure) / max_edge * 2
        return height / max_edge


class HealMesh:
    def __init__(self, mesh_file):
        import gmsh_io
        self.mesh = gmsh_io.GmshIO(mesh_file)

        base, ext = os.path.splitext(mesh_file)
        self.healed_mesh_name = base + "_healed.msh"


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


    def make_node_to_el(self):
        # make node -> element map
        self.node_els = collections.defaultdict(set)
        for eid, e in self.mesh.elements.items():
            type, tags, node_ids = e
            for n in node_ids:
                self.node_els[n].add(eid)


    def heal_small_edges(self, tol):
        """
        Try to remove elements with small fraction of smallest and largest edge.
        :param tol:
        :return:
        """
        self.make_node_to_el()
        self._heal_queue(tol, check_method=self._check_small_edge)


    def heal_flat_elements(self, tol):
        self._heal_queue(tol, check_method=self._check_flat)


    def _heal_queue(self, tol, check_method):
        el_to_check = collections.deque(self.mesh.elements.keys())
        while el_to_check:
            eid = el_to_check.popleft()
            if eid in self.mesh.elements:
                el_to_check.extend(check_method(eid, tol))

    def _make_shape(self, element_tuple):
        type, tags, node_ids = element_tuple
        nodes = np.array([self.mesh.nodes[n] for n in node_ids])
        if len(node_ids) == 4:
            return type, tags, node_ids, Tetrahedron(nodes)
        elif len(node_ids) == 3:
            return type, tags, node_ids, Triangle(nodes)
        else:
            return type, tags, node_ids, None


    def quality_statistics(self, methods, bad_el_tol=0.01):
        """
        Vector of number of elements in quality bins:
        (1, 0.5), (0.5, 0.25), ...
        :return:
        """
        bad_els = {method: [] for method in methods}
        bins = 2.0 ** (np.arange(-15, 1))
        bins = np.concatenate((bins, [np.inf]))
        histogram = {method: np.zeros_like(bins, dtype=int) for method in methods}
        for eid, e in self.mesh.elements.items():
            type, tags, node_ids, shape = self._make_shape(e)
            if shape:
                for method, hist in histogram.items():
                    quality = getattr(shape, method)()
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
        methods = {'flow_stats': 'smooth_grad_error_indicator', 'gamma_stats':'gmsh_gamma'}
        hist, bins, bad_els = self.quality_statistics(methods=methods.values(), bad_el_tol=el_tol)
        output = {}
        for name, method in methods.items():
            output[name] = dict(hist=hist[method].tolist(), bins=bins.tolist(), bad_elements=bad_els[method], bad_el_tol=el_tol)
        import yaml
        with open(filename, "w") as f:
            yaml.dump(output, f)




    def _modif_flat(self, nodes):
        flatness = flat_indicator_3d(nodes)
        if flatness < tol:
            tetrahedron_skew_edges = [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
            dists = [skew_line_dist(nodes[e]) for e in tetrahedron_skew_edges]
            i_min = np.argmin([np.abs(d) for d, norm in dists])
            shift = dists[i_min][0] * dists[i_min][1]
            edge = tetrahedron_skew_edges[i_min]

            nodes[edge[0]] -= shift / 2
            nodes[edge[1]] -= shift / 2
            nodes[edge[2]] += shift / 2
            nodes[edge[3]] += shift / 2

            common_normal = dists[i_min][1]
            p1 = nodes[edge[0]]
            d1 = nodes[edge[1]] - p1
            p2 = nodes[edge[2]]
            d2 = nodes[edge[3]] - p2
            n2 = np.cross(d2, common_normal)
            intersection = p1 + ((p2 - p1) @ n2) / (d1 @ n2) * d1
            return edge, nodes, intersection, flatness
        else:
            return None

    def _check_flat(self, eid, tol):
        result = check_flat_element(nodes, quality_tol)
        if result:

            min_skew_edge, nodes, isec, flatness = result
            print("eid: ", eid, "flat: ", flatness, "nodes: ", node_ids)
            # move nodes
            for nid, node in zip(node_ids, nodes):
                mesh.nodes[nid] = node

            # remove flat element
            del mesh.elements[eid]

            # add intersection node
            max_n_id = max(*mesh.nodes.keys())
            new_node_id = max_n_id + 1
            mesh.nodes[new_node_id] = isec

            # modified all elements attached to the nodes
            modified = set()
            for nid in node_ids:
                modified.update(mesh.node_els[nid])

            # split attached elements
            max_el_id = max(*mesh.elements.keys())
            for face in [[0, 1, 2], [0, 1, 3], [2, 3, 0], [2, 3, 1]]:
                # split elements connected to the face
                # intersection splits the first edge of the face
                edge_n0_id = node_ids[face[0]]
                edge_n1_id = node_ids[face[1]]
                print("edge nids: ", edge_n0_id, edge_n1_id)

                face_els = [mesh.node_els[node_ids[loc_node]] for loc_node in face]
                elements = set.intersection(*face_els)
                assert len(elements) <= 3, elements
                for el_id in elements:
                    if el_id in mesh.elements:
                        el_type, el_tags, el_node_ids = mesh.elements[el_id]
                        print("el_id: ", el_id, el_node_ids)
                        if edge_n0_id not in el_node_ids or edge_n1_id not in el_node_ids:
                            print("Warn: ", "Missing edge")
                            continue
                        loc_n0 = el_node_ids.index(edge_n0_id)
                        loc_n1 = el_node_ids.index(edge_n1_id)
                        node_ids0 = el_node_ids.copy()
                        node_ids0[loc_n0] = new_node_id
                        node_ids1 = el_node_ids.copy()
                        node_ids1[loc_n1] = new_node_id
                        el0 = (el_type, el_tags, node_ids0)
                        el1 = (el_type, el_tags, node_ids1)
                        mesh.elements[el_id] = el0
                        max_el_id += 1
                        mesh.elements[max_el_id] = el1
                        modified.add(el_id)
                        modified.add(max_el_id)
                        # update nodes -> elements lists
                        for nid in el_node_ids:
                            node_el_set = mesh.node_els[nid]
                            node_el_set.discard(eid)
                            node_el_set.add(max_el_id)
                        mesh.node_els[el_node_ids[loc_n0]].discard(max_el_id)
                        mesh.node_els[el_node_ids[loc_n1]].discard(el_id)
                        mesh.node_els[new_node_id].add(el_id)
                        mesh.node_els[new_node_id].add(max_el_id)

        # return modified elements

        # check corectness of elements

    def _check_small_edge(self, eid, quality_tol):
        """
        Check element, possibly contract its shortest edge.
        :param mesh:
        :param eid:
        :return: List of still existing but changed elements.
        """
        quality_tol = 0.001
        type, tags, node_ids, shape = self._make_shape(self.mesh.elements[eid])
        merge_nodes = None
        if shape is not None:
            quality = shape.small_edge_ratio()
            if quality < quality_tol:
                loc_node_a, loc_node_b = shape.edges[np.argmin(shape.edge_lens)]
                merge_nodes = (node_ids[loc_node_a], node_ids[loc_node_b])


        if merge_nodes is None:
            return []

        # merge nodes
        print("eid: {} q{}d: {} nodes: {}".format(eid, shape.dim, quality, merge_nodes))
        node_a, node_b = merge_nodes
        els_a = self.node_els[node_a]
        els_b = self.node_els[node_b]
        # remove elements containing the edge (a,b)
        for i_edge_el in els_a & els_b:
            if i_edge_el in self.mesh.elements:
                del self.mesh.elements[i_edge_el]
                print("del el: ", i_edge_el)
        # substitute node a for the node b
        for i_b_el in els_b:
            if i_b_el in self.mesh.elements:
                type, tags, node_ids = self.mesh.elements[i_b_el]
                node_ids = [node_a if n == node_b else n for n in node_ids]
                self.mesh.elements[i_b_el] = (type, tags, node_ids)
        # average nodes, remove the second one
        node_avg = np.average([np.array(self.mesh.nodes[n]) for n in merge_nodes], axis=0)
        #print("node avg: ", node_avg)
        self.mesh.nodes[node_a] = node_avg
        del self.mesh.nodes[node_b]
        print("del node: ", node_b)

        # merge node element lists
        self.node_els[node_a] = els_a | els_b
        del self.node_els[node_b]

        return self.node_els[node_a]



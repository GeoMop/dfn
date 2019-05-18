"""
1. DONE, gmsh reference simplex
2. DONE, generate square fracture, side a = 1,
   uniform random orientation f(theta) = sin(theta)
   centre in cube (-a sqrt(2)) (1,1,1) : (1+a sqrt(2))(1,1,1)
3. DONE, if positive intersetion area -> accept
4. DONE, generate mesh
   random corsssection (0.01, 0.1)*a
5. compute three fluxes in single flow123d calculation (three rotations of the simplex + fracture)
6. write down:
   fracture angles, conductivity angles
   conductivity eigenvalues
   fracture size, crossection
   intersection area
   intersection center of mass

7. generate minimum 10 000 samples
8.
"""

from typing import List
import attr
from gmsh_api import gmsh
import fracture_generator as fg
import numpy as np
import os
import subprocess
import yaml
import sys

base_dir = "../homogenization"




@attr.s(auto_attribs=True)
class SimplexSample:
    dir: str
    fracture_normal: List[float] = None
    conductivity_base: float = None
    fr_conductivity: float = None
    fr_cross_section: float = None
    fr_size: float = None
    mass: float = None
    center_of_mass: List[float] = None
    result_fluxes: List[float] = None


class Realization:
    fr_side = 1
    el_size = 0.1
    summary = "samples.txt"

    def __init__(self, dir,  population, summary_file):
        self.sample = SimplexSample(dir = dir, fr_size=self.fr_side)

        self.setup_dir(dir)
        self.summary_file = self.base_file(summary_file)

        self.make_mesh(population)
        self.setup_flow123d()


    def setup_dir(self, dir):
        self.dir = os.path.join(base_dir, "samples", dir)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def base_file(self, name):
        return os.path.join(base_dir, name)

    def sample_file(self, name):
        return os.path.join(self.dir, name)

    def make_primitive_shapes(self):
        fr_side = self.fr_side
        self._ss = self.geo.make_simplex(dim=3).set_region("simplex")
        self._fr = self.geo.rectangle([fr_side, fr_side]).set_region("fracture")

        s0 = self.make_rotations(self.geo.make_simplex(dim=2))
        s1 = self.make_rotations(self.geo.make_simplex(dim=2).rotate([1, 0, 0], np.pi / 2))
        s2 = self.make_rotations(self.geo.make_simplex(dim=2).rotate([0, 1, 0], -np.pi / 2))
        self.face_tools = [(s0, '{}_zface'), (s1, '{}_yface'), (s2, '{}_xface')]

    def random_fr(self, population):
        simplex = self._ss
        fracture = self._fr
        fr_normal = population.sample_normal()
        self.sample.fracture_normal = fr_normal[0].tolist()
        fr_a = population.normal_to_axis_angle(fr_normal)
        axis, angle = fr_a[0, :3], fr_a[0, 3]

        # random position
        pt = np.random.rand(3)
        box_min = -self.fr_side * np.sqrt(2) * np.ones(3)
        box_max = (1 + self.fr_side * np.sqrt(2)) * np.ones(3)
        center = (box_max - box_min) * pt + box_min

        # axial rotation
        normal_angle = 2 * np.pi * np.random.uniform()
        fr = fracture.copy().rotate([0, 0, 1], normal_angle).rotate(axis, angle).translate(center)

        fr = fr.intersect(simplex)
        center, mass = fr.center_of_mass()
        if mass < 1e-3:
            return False
        else:
            self.sample.mass = mass
            self.sample.center_of_mass = center.tolist()
            simpl, fr = self.geo.fragment(simplex.copy(), fr)
            sf_group = self.geo.group(simpl, fr)
            self.instance = sf_group
            return True

    def make_rotations(self, shape):
        xyz_instance = shape.copy().translate([2, 0, 0]).modify_regions("{}_xyz")
        yzx_instance = shape.copy().rotate([1, 1, 1], -2 * np.pi / 3) \
            .translate([4, 0, 0]).modify_regions("{}_yzx")
        zxy_instance = shape.copy().rotate([1, 1, 1], -4 * np.pi / 3) \
            .translate([6, 0, 0]).modify_regions("{}_zxy")
        return [xyz_instance, yzx_instance, zxy_instance]


    def make_mesh(self, population):
        self.geo = gmsh.Geometry('occ', "three_frac_symmetric", verbose=True)
        geopt = gmsh.GeometryOptions()
        geopt.ToleranceBoolean = 1e-3
        geopt.MatchMeshTolerance = 1e-3
        geopt.Tolerance = 1e-3

        #self.geo.keep_only()
        self.make_primitive_shapes()

        while not self.random_fr(population):
            pass

        instances = self.make_rotations(self.instance)
        self.geo.remove_duplicate_entities()

        # Select boundaries, must be done after all, since copy doesn't preserve boundary structure
        b_group = []
        for face_tool in self.face_tools:
            tool_instances, format = face_tool
            for inst, inst_tool in zip(instances, tool_instances):
                b_inst = inst.get_boundary_per_region()
                b_inst_group = self.geo.group(*b_inst)
                b = b_inst_group.select_by_intersect(inst_tool).modify_regions(format)
                b_group.append(b)

        self.geo.remove_duplicate_entities()
        mesh_objects = b_group + instances
        mesh = gmsh.MeshOptions()
        mesh.ToleranceInitialDelaunay = 1e-3
        mesh.CharacteristicLengthMin = self.el_size
        mesh.CharacteristicLengthMax = self.el_size

        self.geo.make_mesh(mesh_objects, eliminate=True)
        mesh_file = self.sample_file("simplex_mesh.msh2")
        self.geo.write_brep(self.sample_file("simplex.brep"))
        self.geo.write_mesh(mesh_file)
        self.mesh_file = mesh_file[:-1]
        os.rename(mesh_file, self.mesh_file)
        del self.geo

    @staticmethod
    def substitute_placeholders(file_in, file_out, params):
        """
        Substitute for placeholders of format '<name>' from the dict 'params'.
        :param file_in: Template file.
        :param file_out: Values substituted.
        :param params: { 'name': value, ...}
        """
        used_params = []
        with open(file_in, 'r') as src:
            text = src.read()
        for name, value in params.items():
            placeholder = '<%s>' % name
            n_repl = text.count(placeholder)
            if n_repl > 0:
                used_params.append(name)
                text = text.replace(placeholder, str(value))
        with open(file_out, 'w') as dst:
            dst.write(text)
        return used_params



    def setup_flow123d(self):
        self.sample.fr_cross_section = np.random.uniform(0.01, 0.1)
        self.sample.conductivity_base = 1
        self.sample.fr_conductivity = 100

        params = {
            'fr_cross_section': self.sample.fr_cross_section,
            'bulk_conductivity': self.sample.conductivity_base,
            'fr_conductivity': self.sample.fr_conductivity,
            'mesh_file': os.path.basename(self.mesh_file)
        }
        self.substitute_placeholders(
            self.base_file("flow_in.yaml"),
            self.sample_file("flow_in.yaml"),
            params)

    def run(self):
        args = [
            "../../flow.sh",
            "--yaml_balance",
            "flow_in.yaml"
        ]
        with open(self.sample_file("stdout"), "w") as stdout:
            with open(self.sample_file("stderr"), "w") as stderr:
                completed = subprocess.run(args, cwd=self.dir,
                                   stdout=stdout, stderr=stderr)
        if completed.returncode == 0:
            self.sample.result_fluxes = self.extract_results().tolist()

        # append result
        with open(self.summary_file, "a") as f:
            line = str(attr.asdict(self.sample)) + "\n"
            f.write(line)



    def extract_results(self):
        with open(os.path.join(self.dir, "output", "water_balance.yaml")) as f:
            balance = yaml.load(f)
        data = balance['data']
        homo_cond_tn = np.zeros((3,3))
        ori_face_map = {'xyz':0, 'yzx':1, 'zxy':2 }
        face_map = {'xface': 0, 'yface': 1, 'zface': 2}
        for item in data:
            region = item['region']
            if region[0] != '.':
                continue
            tokens = region.split('_')
            if tokens[0] in ['.fracture', '.simplex']:
                orientation, face = tokens[1:]
                i_ori = ori_face_map[orientation]
                i_face = face_map[face]
                tn_col = i_ori
                tn_row = (i_face + i_ori) % 3
                bc_flux = item['data'][0]
                homo_cond_tn[tn_row][tn_col] += bc_flux
        return homo_cond_tn













def create_samples(id_range):


    fracture_population = fg.FisherOrientation(0, 0, 0)

    summary_file = "summary_{}_{}.txt".format(*id_range)

    for id in range(id_range[0], id_range[1]):
        dir = "{:06d}".format(id)
        x = Realization(dir, fracture_population, summary_file)
        #print(attr.asdict(x.sample))
        x.run()
    #geo.show()


def process():
    # sym_homo_cond_tn = (homo_cond_tn + homo_cond_tn.T) / 2
    # eigv = np.linalg.eigvals(homo_cond_tn)
    # sym_eigv = np.linalg.eigvals(sym_homo_cond_tn)
    # print("tensor:\n", homo_cond_tn, "\neigv: ", eigv)
    # print("\n\nsym tensor:\n", sym_homo_cond_tn, "\neigv: ", sym_eigv)
    # return homo_cond_tn
    pass

def main():
     command = sys.argv[1]
     if command == 'sample':
         id_range = [int(token) for token in sys.argv[2:4]]
         create_samples(id_range)
     elif command == 'process':
         process()



main()
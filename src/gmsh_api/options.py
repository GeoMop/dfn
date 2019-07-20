import enum
import gmsh

"""
Structure:
gmsh
gmsh.option
gmsh.model
gmsh.model.mesh
gmsh.model.mesh.field
gmsh.model.geo
gmsh.model.geo,mesh
gmsh.model.occ
gmsh.view
gmsh.plugin
gmsh.graphics
gmsh.fltk
gmsh.onelab
gmsh.logger

gmsh_api, issues:
- terrible interface to fields
- get_boundary not part of geometry model (occ/geo), need lot of synchronizations
- all existing dimtags are meshed not only those with assigned physical groups
- physical groups are not assigned to the objects, but groups are formed from objects, 
  possible error having single object in more physical groups
- Mesh.Format option - doc do not support gmsh 2.0 format
  gmsh.write function seems to ignore the format and use extensions which are not documented
- gmsh.model.occ.setMeshSize - seems have no effect, in particular in combination with getBoundary
- no constant field
- gmsh.model.occ.removeAllDuplicates ... doesn't work
- seems that occ.copy() doesn't preserve boundaries, so boundary dim tags are copied twice
"""





class Algorithm2d(enum.IntEnum):
    MeshAdapt = 1
    Automatic = 2
    Delaunay = 5
    FrontalDelaunay = 6
    BAMG = 7
    FrontalDelaunayQuads = 8
    ParalelogramsPacking = 9


class Algorithm3d(enum.IntEnum):
    Delaunay = 1
    Frontal = 4
    MMG3D = 7
    RTree = 9
    HXT = 10

class QualityType(enum.IntEnum):
    SignedInversConditionNumber = 0     # SICN~signed inverse condition number (TODO: explain)
    SignedInverseGradientError = 1      # SIGE~signed inverse gradient error (TODO: explain)
    Gamma = 2                           # (default) gamma ~ vol / sum_face / max_edge; ( "element height" / diameter)
    Distorsion = 3                      # Disto~minJ / maxJ (TODO: explain)

class OptionsBase:
    """
    Auxiliary class to set GMSH options as a class attributes:

    my_options.option_name = value

    Valid option names are defined in the constructor of the derived class as
    their own attributes. After that call of 'finish_init' will:
    1. collect existing attributes
    2. set appropriate GMSH options to default values
    3. set __setattr__ method so that furhter assignements to attribute are translated to
       setting the GMSH option.

    """
    def __init__(self, prefix):
        object.__setattr__(self, 'prefix', prefix)
        # Prefix of the GMSH option, e.g. 'Mesh.'
        object.__setattr__(self, 'names_map', {})
        # Dictionary of valid options: option_name -> type
        object.__setattr__(self, '_sa', self.init_setattr)



    def finish_init(self):
        object.__setattr__(self, '_sa', self.instance_setattr)


    def _add(self, gmsh_name, default):
        """
        Define new option with name 'gmsh_name'.
        :param default: either initial value or just type (enum, bool, float, int, str)

        If default value is provided it is passed to GMSH immediately.
        """
        if isinstance(default, type):
            option_type = default
            self.names_map[gmsh_name] = (gmsh_name, option_type)
        else:
            option_type = type(default)
            self.names_map[gmsh_name] = (gmsh_name, option_type)
            self.instance_setattr(gmsh_name, default)

    def __setattr__(self, key, value):
        self._sa(key, value)


    def init_setattr(self, key, value):
        """
        Syntactic sugar for _add.
        """
        self._add(key, value)



    def instance_setattr(self, key, value):
        assert key in self.names_map, "Unknown option {}.".format(key)
        gmsh_name, option_type = self.names_map[key]
        full_name = self.prefix + gmsh_name
        if isinstance(value, (int, float, bool)):
            assert option_type in {int, float, bool} or issubclass(option_type, enum.Enum), str(option_type)
            gmsh.option.setNumber(full_name, value)
        elif isinstance(value, str):
            assert option_type is str
            gmsh.option.setString(full_name, value)
        else:
            raise ValueError("Unsupported value type {} for GMSH option type.")


class Mesh(OptionsBase):
    def __init__(self):
        super().__init__('Mesh.')
        self.Algorithm = Algorithm2d.Automatic
        # 2D mesh algorithm
        self.Algorithm3D = Algorithm3d.Delaunay
        # 3D mesh algorithm
        self.ToleranceInitialDelaunay = 1e-12
        # Tolerance for initial 3D Delaunay mesher
        self.ToleranceEdgeLength = 0
        # Skip a model edge in mesh generation if its length is less than user’s defined tolerance
        self.AngleToleranceFacetOverlap = 0.1
        #
        self.AnisoMax = 1e-10
        # Maximum allowed element anisotropy.

        self.CharacteristicLengthFromPoints = True
        # Compute mesh element sizes from values given at geometry points
        self.CharacteristicLengthFromCurvature = True
        # Automatically compute mesh element sizes from curvature (experimental)
        self.CharacteristicLengthExtendFromBoundary = int
        # Extend computation of mesh element sizes from the boundaries into the interior
        # (for 3D Delaunay, use 1: longest or 2: shortest surface edge length)
        self.CharacteristicLengthMin = float
        # Minimum mesh element size
        self.CharacteristicLengthMax = float
        # Maximum mesh element size
        self.CharacteristicLengthFactor = float
        # Factor applied to all mesh element sizes

        self.QualityType = QualityType.Gamma
        # Type of quality measure (TODO: ?? is it used in the mesh optimizer)

        self.MinimumCirclePoints = 6
        # Specify point density for curves when CharacteristicLengthFromCurvature is ON.
        # Minimum number of points used to mesh a full circle.
        self.MinimumCurvePoints = 6
        # Minimum number of points used to mesh a (non-straight) curve

        self.RandomFactor = 1e-9
        # Random factor used in the 2D meshing algorithm (should be increased if RandomFactor * size(triangle)/size(model) approaches machine accuracy)
        self.RandomFactor3D = 1e-12
        # Random factor used in the 3D meshing algorithm
        self.finish_init()


class Geometry(OptionsBase):
    def __init__(self):
        super().__init__('Geometry.')

        self.Tolerance = 1e-08
        # Geometrical tolerance, influence: MeshRemoveDuplicateNodes, mesh merging, ...
        self.ToleranceBoolean = 0.0
        # Geometrical tolerance for boolean operations


        # According to the GMSH source it seems that following options are used for matching GMSH mesh and geometry
        # during the "Merge" operation in GUI.
        # self.MatchGeomAndMesh = False
        # # Matches geometries and meshes.
        # self.MatchMeshScaleFactor = 1
        # # Rescaling factor for the mesh to correspond to size of the geometry
        # self.MatchMeshTolerance = 1e-06
        # # Tolerance for matching mesh and geometry

        self.AutoCoherence = 1
        # 1 - automaticaly remove duplicate entities, 2 - automaticaly remove also degenerate entities


        self.OCCFixDegenerated = False
        # Fix degenerated edges/faces in STEP, IGES and BRep models
        self.OCCFixSmallEdges = False
        # Fix small edges in STEP, IGES and BRep models
        self.OCCFixSmallFaces = False
        # Fix small faces in STEP, IGES and BRep models
        self.OCCBooleanPreserveNumbering = True
        # Try to preserve numbering of entities through OCC boolean operations
        self.OCCSewFaces = False
        # Sew faces in STEP, IGES and BRep models.

        self.finish_init()

from random_frac import *

nf = 100        # number of fractures
mind = 0.05     # minimal distance from boundary
minr = 0.0004     # minimal fracture radius
maxr = 0.5     # maximal fracture radius


# generate random fractures
frac_data = generate_fractures(nf, mind, minr, maxr)

# write fractures to file for future reuse
write_fractures(frac_data, 'fractures.dat')

# read  fractures from file
frac_data = read_fractures('fractures.dat')

# generate mesh and save to file
#generate_mesh(frac_data, max_el_size=1, file_name="test_api.msh2", verbose=1)

generate_mesh(frac_data, max_el_size=1, file_name="test_api.msh2", verbose=1, shape="rectangle")
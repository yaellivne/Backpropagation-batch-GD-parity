# Yael Livne 313547929, Nir David 313231805
import numpy as np

# section 1


def define_db():
    bin_arr = [0, 1]
    x_mat = np.array([np.array([x, y, z]) for x in bin_arr for y in bin_arr for z in bin_arr])
    t = [np.logical_xor(x_mat[i][j], x_mat[i][j+1]) for i in range(0, 8) for j in range(0,2)]

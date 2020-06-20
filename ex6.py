# Yael Livne 313547929, Nir David 313231805
import numpy as np

# section 1


def define_db():
    bin_arr = [0, 1]
    x_mat = np.array([np.array([x, y, z]) for x in bin_arr for y in bin_arr for z in bin_arr])
    t = []
    for row in x_mat:
        a = np.logical_xor(row[0], row[1])
        t.append(int(np.logical_xor(a, row[2])))
    return x_mat,t

if __name__ == '__main__':
    x_mat,t = define_db()
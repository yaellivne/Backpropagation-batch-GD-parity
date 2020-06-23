# Yael Livne 313547929, Nir David 313231805
import numpy as np
import matplotlib.pyplot as plt


def calc_parity(x_vec):
    """
    :param x_vec: (n,2^n) vector of binary combinations where n=3
    :return: bitwise xor between elements of x_vec
    """
    a = np.logical_xor(x_vec[0], x_vec[1])
    return int(np.logical_xor(a, x_vec[2]))


def define_db():
    """
    defines the needed training set
    :return:
    x_arr - (3,8) vector of all possible binary combinations + 1 for the wi,0 weight multiplication. total: (4,8)
    t_vec - parity sign for each row in 2^n dims = (1,8)
    """
    bin_arr = [0, 1]
    x_arr = np.array([np.array([1, x, k, z]) for x in bin_arr for k in bin_arr for z in bin_arr])
    t_vec = []
    for row in x_arr:
        t_vec.append(calc_parity(row))
    return x_arr, t_vec


def logistic_sigmoid(x):
    """
    :param x: vector of any dimensions
    :return: logistic sigmoid applied on x
    """
    return 1 / (1 + np.exp(-x))


def init_weights(shape):
    """
    :return:
    weight vector size (shape)- iid normally distributed with 0 mean and sigma 1
    """
    return np.random.normal(0, 1, shape)


def mse(y_output, t_vec):
    """
    :param y_output: output of the network
    :param t_vec: training set to compare
    :return: MSE between y and t
    """
    return (1 / 8) * np.sum(np.square(y_output - t_vec))


def calc_zi(alpha):
    """
    :param alpha: mapping function between the cells: for example w20 + w21*x1 + w22*x2 + w23*x3
    :return: zi = logistic sigmoid applied on (alpha)
    """
    return np.vstack((np.ones(8), logistic_sigmoid(alpha)))


def batch_gd(w_prev, eta_param, delta, zi):
    """
    :param w_prev: weight vector in iteration t
    :param eta_param: hyper parameter of the program
    :param delta: delta i is the backward pass result (per inner/output cell)
    :param zi: inner cells / output cell
    :return: next weight vector - in iteration t+1
    """
    return w_prev - eta_param * delta.dot(zi.T)


def forward_pass(x_train_vec, w_zi, w_output):
    """
    :param x_train_vec: training set of all binary combinations
    :param w_zi: weights vector for inner cells z_i
    :param w_output: weights vector for output cell y
    :return: z: inner cells, y_output: last cell in network
    """
    alpha_i = w_zi.dot(x_train_vec.T)  # (3,8)
    z = calc_zi(alpha_i)  # (4,8) first row is 1
    y_output = logistic_sigmoid(w_output.dot(z))  # (1,8)
    return z, y_output


def backward_pass(y_output, t_parity, z_i, w_output):
    """
    :param y_output: last cell in network
    :param t_parity: parity sign for each row in 2^n dims = (1,8)
    :param z_i: inner cells in network
    :param w_output: weights vector for output cell y
    :return: delta functions for inner cells and output cells
    """
    # logistic sigmoid derivative: h' = s(1-s)
    # delta output: h'*(y-t)
    delta_output_vec = y_output * (1 - y_output) * (y_output - t_parity)  # (1,8)
    # delta inner: h'*(sum(wk*delta_n_k)
    y_inner = z_i[1:, :]
    delta_inner_vec = (y_inner * (1 - y_inner)) * w_output[1:].reshape(
        (-1, 1)) * delta_output_vec  # (3,8) * (3,1) * (1,8) = (3,8)
    return delta_output_vec, delta_inner_vec


def main_run(problem_dims):
    """
    :param problem_dims: number of hidden layers
    :return: mse_vec: MSE vector calculated per iteration
    """
    mse_vec = []
    x_train, t = define_db()  # x_mat - (8, 4) first is 1 ; t - (1, 8)
    num_run, iteration = 0, 0
    while num_run <= 100:
        num_run += 1
        w_zi = init_weights(problem_dims)
        w_output = init_weights(problem_dims[0]+1)  # (1,4)
        while iteration <= 2000:
            iteration += 1
            z_i, y = forward_pass(x_train, w_zi, w_output)
            mse_vec.append(mse(y, t))
            delta_output, delta_inner = backward_pass(y, t, z_i, w_output)

            w_output = batch_gd(w_output, eta, delta_output, z_i)
            w_zi = batch_gd(w_zi, eta, delta_inner, x_train.T)
    return mse_vec


def plot_results(mse_vec, n):
    """
    :param mse_vec: vector of the MSE calculated per iteration
    :param n: number of hidden layers
    :return: just a plot function
    """
    plt.plot(mse_vec, marker='.')
    plt.ylabel('MSE')
    plt.xlabel('Iteration number')
    plt.title(f'MSE over 100 runs vs. iteration number - \n{n} hidden layers')
    plt.show()


def plot_both(err_vec3, err_vec6):
    """
    :param err_vec3: vector of the MSE calculated per iteration for 3 hidden layers
    :param err_vec6: vector of the MSE calculated per iteration for 6 hidden layers
    :return:
    """
    plt.plot(err_vec3, marker='.', label='3 hidden layers')
    plt.plot(err_vec6, marker='.', label='6 hidden layers')
    plt.ylabel('MSE')
    plt.xlabel('Iteration number')
    plt.legend()
    plt.title('MSE over 100 runs vs. iteration number')
    plt.show()


if __name__ == '__main__':
    'Starting section A:'
    eta = 2
    err_vec = main_run((3, 4))
    # plot_results(err_vec, 3)
    'End of section A'
    'Start of section B'
    err_vec_b = main_run((6, 4))
    # plot_results(err_vec_b, 6)
    'End of section B'
    plot_both(err_vec, err_vec_b)

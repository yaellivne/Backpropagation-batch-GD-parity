# Yael Livne 313547929, Nir David 313231805
import numpy as np
import matplotlib.pyplot as plt


def calc_parity(x_vec):
    a = np.logical_xor(x_vec[0], x_vec[1])
    return int(np.logical_xor(a, x_vec[2]))


def define_db():
    bin_arr = [0, 1]
    x_arr = np.array([np.array([1, x, k, z]) for x in bin_arr for k in bin_arr for z in bin_arr])
    t_vec = []
    for row in x_arr:
        t_vec.append(calc_parity(row))
    return x_arr, t_vec


def logistic_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_weights(shape):
    """
    :return:
    weight vector size 4- iid normally distributed with 0 mean and sigma 1
    """
    return np.random.normal(0, 1, shape)


def mse(y_output, t_vec):
    return (1 / 8) * np.sum(np.square(y_output - t_vec))


def calc_zi(alpha):
    return np.vstack((np.ones(8), logistic_sigmoid(alpha)))


def batch_gd(w_prev, eta_param, delta, zi):
    return w_prev - eta_param * delta.dot(zi.T)


def forward_pass(x_train_vec):
    alpha_i = w_zi.dot(x_train_vec.T)  # (3,8)
    z = calc_zi(alpha_i)  # (4,8) first row is 1
    y_output = logistic_sigmoid(w_output.dot(z))  # (1,8)
    return z, y_output


def backward_pass(y_output, t_pairity):
    # logistic sigmoid derivative: h' = s(1-s)
    # delta output: h'*(y-t)
    delta_output_vec = y_output * (1 - y_output) * (y_output - t_pairity)  # (1,8)
    # delta inner: h'*(sum(wk*delta_n_k)
    y_inner = z_i[1:, :]
    delta_inner_vec = (y_inner * (1 - y_inner)) * w_output[1:].reshape(
        (-1, 1)) * delta_output_vec  # (3,8) * (3,1) * (1,8) = (3,8)
    return delta_output_vec, delta_inner_vec


if __name__ == '__main__':
    'Starting section A:'
    eta = 2
    err_vec = []
    x_train, t = define_db()  # x_mat - (8, 4) first is 1 ; t - (1, 8)
    num_run, iteration = 0, 0
    while num_run <= 100:
        num_run += 1
        w_zi = init_weights((3, 4))
        w_output = init_weights(4)  # (1,4)
        while iteration <= 2000:
            iteration += 1
            z_i, y = forward_pass(x_train)
            err_vec.append(mse(y, t))
            delta_output, delta_inner = backward_pass(y, t)

            w_output = batch_gd(w_output, eta, delta_output, z_i)
            w_zi = batch_gd(w_zi, eta, delta_inner, x_train.T)

    mean_err = np.array(err_vec)
    plt.plot(mean_err, marker='.')
    plt.ylabel('MSE')
    plt.xlabel('Iteration number')
    plt.title('MSE over 100 runs vs. iteration number - \n3 hidden layers')
    plt.show()
    'End of section A'
    'Start of section B'

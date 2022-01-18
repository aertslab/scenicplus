from random import uniform
import numpy as np
import numba

"""
REFs:    M. Hopfensitz et al., "Multiscale Binarization of Gene Expression Data for Reconstructing Boolean Networks," 
        in IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 9, no. 2, pp. 487-498, March-April 2012, 
        doi: 10.1109/TCBB.2011.62.

        https://github.com/cran/Binarize/blob/master/src/binarizeBASCA.c    Author: Stefan Mundus
        https://github.com/cran/Binarize/blob/master/src/common.c           Author: Stefan Mundus


Algorithm to binarize a vector.
Steps:
    1. Compute a series of step functions (each function minimizes the eucledian distance between the new step function and the original data)
    2. Find strongest discontinuity in each step function
    3. Estimate location and variation of the strongest discontinuities
"""

"""
HELPER FUNCTIONS:
"""


@numba.jit(nopython=True)
def cost_ab(vect, a, b):
    """
    Calculates quadratic distance between original data points and mean of data points in range a to b inclusive
    """
    # Add 1 to the range as we want to calculate the quadratic distance from a to b, with b included.
    return np.sum((vect[a:b + 1] - np.mean(vect[a:b + 1])) ** 2)


@numba.jit(nopython=True)
def costs(costs_matrix, costs_is_cached, vect, a, b):
    # Check if we have the cost_ab(vect, a, b) value cached.
    if costs_is_cached[a, b] == np.bool_(True):
        return costs_matrix[a, b]

    # Else calculate cost_ab(vect, a, b) and cache it for next time.
    current_cost = cost_ab(vect, a, b)
    costs_matrix[a, b] = current_cost
    costs_is_cached[a, b] = np.bool_(True)

    return current_cost


@numba.jit(nopython=True)
def init_cost_matrix(vect):
    N = vect.shape[0]
    C = np.zeros((N - 1, N), dtype=np.float64)
    C[0] = [cost_ab(vect, i, N - 1) for i in range(0, N - 1 + 1)]
    return C


@numba.jit(nopython=True)
def calc_jump_height(vect, P, i, j):
    """
    Calculate jump height/size between data point Pij and Pij + 1, with P the matrix containing location of discontinuities.
    """
    N = vect.shape[0] - 1
    if i == 0 and j > 0:
        return np.mean(vect[P[j, i] + 1:  P[j, i + 1] + 1]) - np.mean(vect[0: P[j, i] + 1])
    elif i == j > 0:
        return np.mean(vect[P[j, i] + 1: N + 1]) - np.mean(vect[P[j, i - 1] + 1: P[j, i] + 1])
    elif i == j == 0:
        return np.mean(vect[P[j, i] + 1: N + 1]) - np.mean(vect[0: P[j, i] + 1])
    else:
        return np.mean(vect[P[j, i] + 1: P[j, i + 1] + 1]) - np.mean(vect[P[j, i - 1] + 1: P[j, i] + 1])


@numba.jit(nopython=True)
def calc_error(vect, P, i, j):
    """
    Calculate approximation error of a threshold at the discontinuity with respect to the original data
    This is the sum of the quadratic distances of all data points to the threshold z defined by the i-th discontinuity
    """
    N = vect.shape[0]
    z = (vect[P[j, i]] + vect[P[j, i] + 1]) / 2
    return np.sum(((vect - z) ** 2)[0: N])


@numba.jit(nopython=True)
def moving_block_bootstrap(v):
    N = v.shape[0]
    bootstrapped_values = np.zeros(N, dtype=np.float64)
    bl = round(N ** 0.25) + 1
    sample_count = np.ceil(N / bl)
    m = N - bl

    index = 0

    for i in range(0, sample_count):
        rand = round(uniform(-0.5, m + 0.5))
        rand = rand if rand <= m else m

        for j in range(0, bl):
            if index >= N:
                break
            bootstrapped_values[index] = v[rand + j]
            index += 1
    return bootstrapped_values


@numba.jit(nopython=True)
def norm_dev_median(v, vect):
    N = vect.shape[0]
    median_val = np.floor(np.median(v))
    dev = np.abs(v - median_val)
    mean_val = np.mean(dev)
    return mean_val / (N - 1)


"""
MAIN FUNCTIONS
"""


@numba.jit(nopython=True)
def calc_cost_and_ind_matrix(vect):
    """
    Calculates the matrix C and ind
        C stores the cost of a step function having j intermediate (rows) discontinuities between data points i and N (columns)
        ind contains indicices of optimal break points of all step functions
    """
    N = vect.shape[0]
    C = init_cost_matrix(vect)
    ind = np.zeros((N - 2, N), dtype=np.int64)
    costs_matrix = np.zeros((N, N), np.float64)
    costs_is_cached = np.zeros((N, N), np.bool_)
    for j in range(1, N - 2 + 1):
        for i in range(0, N - j):
            cost_min = np.inf
            d_min = -1
            for d in range(i, N - j):
                cost = costs(costs_matrix, costs_is_cached, vect, i, d) + C[j - 1, d + 1]
                if cost < cost_min:
                    cost_min = cost
                    d_min = d
            C[j, i] = cost_min
            ind[j - 1, i] = d_min
    return C, ind


@numba.jit(nopython=True)
def calc_P_matrix(ind):
    """
    Converts ind matrix from calc_cost_and_ind_matrix to a matrix with rows representing the number of discontinuities
    (1-based) and as values the location of the i-th ().
    """

    N = ind.shape[0]
    P = np.zeros(ind.shape, dtype=np.int64)
    for j in range(0, N):
        z = j
        P[j, 0] = ind[z, 0]
        if j > 0:
            z = z - 1
            for i in range(1, j + 1):
                P[j, i] = ind[z, int(P[j, i - 1]) + 1]
                z = z - 1
    return P


@numba.jit(nopython=True)
def calc_scores(vect, P):
    """
    Calculate the score of al step function discontinuities in P
    This score is the jump height divided by the approximation error
    """

    N = P.shape[0]
    Q = np.zeros(P.shape, dtype=np.float64)  # stores scores for each discontinuity
    # stores the score of the discontinuity with the maximum score for each step function
    Q_max = np.zeros(N, dtype=np.float64)
    # stores the index of the discontinuity with the maximum score for each step function
    ind_Q_max = np.zeros(N, dtype=np.int64)

    for j in range(0, N):
        q_max = -1
        ind_q_max = -1
        for i in range(0, j + 1):
            # calculate jump height
            h = calc_jump_height(vect, P, i, j)
            e = calc_error(vect, P, i, j)
            q = h / e
            if q > q_max:
                q_max = q
                ind_q_max = P[j, i]
            Q[j, i] = q
        Q_max[j] = q_max
        ind_Q_max[j] = ind_q_max
    return Q, Q_max, ind_Q_max


@numba.jit(nopython=True)
def calc_threshold(vect, v):
    v_med = int(np.floor(np.median(v)))
    return (vect[v_med + 1] + vect[v_med]) / 2


@numba.jit(nopython=True)
def calc_P(v, vect, tau, n_samples):
    nom = norm_dev_median(v, vect)
    t_zero = tau - nom

    p = 1

    for i in range(0, n_samples):
        samples = moving_block_bootstrap(v)
        mdm = norm_dev_median(samples, vect)
        t_star = nom - mdm
        p += (t_star >= t_zero) * 1

    p /= (n_samples + 1)

    return p


@numba.jit(nopython=True)
def binarize(vect, tau=0.01, n_samples=999, calc_p=True, max_elements=100):
    # original step function is just the sorted vector
    vect_sorted = np.sort(vect)

    # if vector is too long, only use top features (scalability)
    if vect_sorted.shape[0] > max_elements:
        vect_sorted = vect_sorted[0:max_elements]

    # step 1: Compute a series of step functions (each function minimizes the eucledian distance between the new step function and the original data)
    _, ind = calc_cost_and_ind_matrix(vect_sorted)
    P = calc_P_matrix(ind)

    # step 2: Find strongest discontinuity in each step function
    Q, _, v = calc_scores(vect_sorted, P)

    # step 3: Estimate location and variation of the strongest discontinuities
    threshold = calc_threshold(vect_sorted, v)
    # p_val = calc_P(v, vect_sorted, tau, n_samples) if calc_p else None

    binarized_measurements = [(val > threshold) * 1 for val in vect]

    return threshold, binarized_measurements

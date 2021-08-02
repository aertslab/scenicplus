from statistics import median, mean
from math import floor, ceil
from random import uniform

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

def MatrixZeros(rows, cols):
    #little function to prevent importing numpy
    return [ [ 0 for i in range(cols) ] for j in range(rows) ]

def ArrayZeros(l):
    return [0 for i in range(l)]

def mean_ab(vect, a, b):
    S = sum(vect[a:b + 1]) # + 1 because we want to calculate the sum from a to b, with b inclusive
    n = b - a + 1
    return S/n

def cost_ab(vect, a, b):
    """
    Calculates quadratic distance between original data points and mean of data points in range a to b inclusive
    """
    Yab = mean_ab(vect, a, b)
    return sum([ (vect[i] - Yab)**2 for i in range(a, b + 1)]) # + 1 because we want to calculate the sum from a to b, with b inclusive

def initCostMatrix(vect):
    N = len(vect)
    C = MatrixZeros(N - 1, N)
    C[0] = [cost_ab(vect, i, N-1) for i in range(0, N - 1 + 1)]
    return C

def calcJumpHeight(vect, P, i, j):
    """
    Calculate jump height/size between data point Pij and Pij + 1, with P the matrix containing location of discontinuities.
    """
    N = len(vect) - 1
    if i == 0 and j > 0:
        return mean_ab(vect, P[j][i] + 1, P[j][i + 1]) - mean_ab(vect, 0, P[j][i])
    elif i == j > 0:
        return mean_ab(vect, P[j][i] + 1, N)           - mean_ab(vect, P[j][i - 1] + 1, P[j][i])
    elif i == j == 0:
        return mean_ab(vect, P[j][i] + 1, N)           - mean_ab(vect, 0, P[j][i])
    else:
        return mean_ab(vect, P[j][i] + 1, P[j][i + 1]) - mean_ab(vect, P[j][i - 1] + 1, P[j][i])

def calcError(vect, P, i, j):
    """
    Calculate approximation error of a threshold at the discontinuity with respect to the original data
    This is the sum of the quadratic distances of all data points to the threshold z defined by the i-th discontinuity
    """
    N = len(vect)
    z = (vect[ P[j][i] ] + vect[ P[j][i] + 1 ]) / 2
    return sum( [ (vect[i] - z)**2 for i in range(0, N)] )

def movingBlockBootstrap(v):
    N = len(v)
    bootstrappedValues = ArrayZeros(N)
    bl = round(N**0.25) + 1
    sample_count = ceil(N / bl)
    m = N - bl

    index = 0

    for i in range(0, sample_count):
        rand = round(uniform(-0.5, m + 0.5))
        rand = rand if rand <= m else m

        for j in range(0, bl):
            if index >= N:
                break
            bootstrappedValues[index] = v[rand + j]
            index += 1
    return bootstrappedValues

def normDevMedian(v, vect):
    N = len(vect)
    median_val = floor(median(v))
    dev = [abs(x - median_val) for x in v]
    mean_val = mean(dev)
    return mean_val/(N-1)

"""
MAIN FUNCTIONS
"""

def calcCostAndIndMatrix(vect):
    """
    Calculates the matrix C and ind
        C stores the cost of a step function having j intermediate (rows) discontinuities between data points i and N (columns)
        ind contains indicices of optimal break points of all step functions
    """
    N = len(vect)
    C = initCostMatrix(vect)
    ind = MatrixZeros(N - 2, N)
    for j in range( 1, N - 2 + 1 ):
        for i in range( 0, N - j ):
            cost_min = float("inf")
            d_min = -1
            for d in range(i, N - j):
                cost = cost_ab(vect, i, d) + C[j - 1][d + 1]
                if cost < cost_min:
                    cost_min = cost
                    d_min = d
            C[j][i] = cost_min
            ind[j-1][i] = d_min
    return C, ind


def calcPMatrix(ind):
    """
    Converts ind matrix from calcCostAndIndMatrix to a matrix with rows representing the number of discontinuities (1-based) and as values the location of the i-th ()
    """

    P = MatrixZeros(len(ind), len(ind[0]))
    for j in range(0, len(ind)):
        z = j
        P[j][0] = ind[z][0]
        if j > 0:
            z = z - 1
            for i in range(1, j + 1):
                P[j][i] = ind[z][int(P[j][i - 1]) + 1]
                z = z - 1
    return P

def calcScores(vect, P):
    """
    Calculate the score of al step function discontinuities in P
    This score is the jump height divided by the approximation error
    """

    Q = MatrixZeros(len(P), len(P[0]))  #stores scores for each discontinuity
    Q_max = ArrayZeros(len(P))          #stores the score of the discontinuity with the maximum score for each step function
    ind_Q_max = ArrayZeros(len(P))      #stores the index of the discontinuity with the maximum score for each step function

    for j in range(0, len(P)):
        q_max = -1
        ind_q_max = -1
        for i in range(0, j + 1):
            #calculate jump height
            h = calcJumpHeight(vect, P, i, j)
            e = calcError(vect, P, i, j)
            q = h/e
            if q > q_max:
                q_max = q
                ind_q_max = P[j][i]
            Q[j][i] = q
        Q_max[j] = q_max
        ind_Q_max[j] = ind_q_max
    return Q, Q_max, ind_Q_max

def calcThreshold(vect, v):
    v_med = floor(median(v))
    return (vect[v_med + 1] + vect[v_med]) / 2

def calcP(v, vect, tau, n_samples):
    nom = normDevMedian(v, vect)
    t_zero = tau - nom

    p = 1

    for i in range(0, n_samples):
        samples = movingBlockBootstrap(v)
        mdm = normDevMedian(samples, vect)
        t_star = nom - mdm
        p += (t_star >= t_zero) * 1

    p /= (n_samples + 1)

    return p

class Result:
    #retian nomenclatur from R version
    def __init__(self, originalMeasurements, intermedSteps, intermedScores, intermedStrongSteps, BinarizedMeasurements, threshold, pVal):
        self.originalMeasurements = originalMeasurements
        self.intermediateSteps = intermedSteps
        self.intermediateStrongestSteps = intermedStrongSteps
        self.intermediateScores = intermedScores
        self.binarizedMeasurements = BinarizedMeasurements
        self.threshold = threshold
        self.pVal = pVal
    
def binarize(vect, tau = 0.01, n_samples = 999):
    #original step function is just the sorted vector
    vect_sorted = sorted(vect)

    #step 1: Compute a series of step functions (each function minimizes the eucledian distance between the new step function and the original data)
    _, ind = calcCostAndIndMatrix(vect_sorted)
    P = calcPMatrix(ind)

    #step 2: Find strongest discontinuity in each step function
    Q, _, v = calcScores(vect_sorted, P)

    #step 3: Estimate location and variation of the strongest discontinuities
    threshold = calcThreshold(vect_sorted, v)
    p_val = calcP(v, vect_sorted, tau, n_samples)

    BinarizedMeasurements = [(val > threshold) * 1 for val in vect]


    return Result(
        originalMeasurements = vect,
        intermedSteps = P,
        intermedScores = Q,
        intermedStrongSteps = v,
        BinarizedMeasurements = BinarizedMeasurements,
        threshold = threshold,
        pVal = p_val
    )
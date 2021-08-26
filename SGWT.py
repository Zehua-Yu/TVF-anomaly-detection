# Coding: UTF-8
# Author: Zehua Yu
# Date: 2021/8/22 13:22
# IDE: PyCharm

import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import fminbound
import scipy.sparse.linalg as ssl
import matplotlib.pylab as plt
import pandas as pd
import matplotlib.pyplot as plt


def laplacian(A, laplacian_type='raw'):
    N = A.shape[0]
    degrees = A.sum(1)
    # To deal with loops, must extract diagonal part of A
    diagw = np.diag(A)

    # w will consist of non-diagonal entries only
    ni2, nj2 = A.nonzero()
    w2 = A[ni2, nj2]
    ndind = (ni2 != nj2).nonzero()  # Non-diagonal indices
    ni = ni2[ndind]
    nj = nj2[ndind]
    w = w2[ndind]

    di = np.arange(N)  # diagonal indices

    if laplacian_type == 'raw':
        # non-normalized laplaciand L = D - A
        L = np.diag(degrees - diagw)
        L[ni, nj] = -w
        L = lil_matrix(L)

    return L


def rough_l_max(L):
    l_max = np.linalg.eigvalsh(L.todense()).max()
    l_max_ub = 1.01 * l_max
    return l_max_ub


def set_scales(l_min, l_max, N_scales):
    t1 = 1
    t2 = 2
    s_min = t1 / l_max
    s_max = t2 / l_min
    # Scales should be decreasing ... higher j should give larger s
    s = np.exp(np.linspace(np.log(s_max), np.log(s_min), N_scales));
    return s


def kernel(x, g_type='abspline', a=2, b=2, t1=1, t2=2):
    if g_type == 'abspline':
        g = kernel_abspline3(x, a, b, t1, t2)
    elif g_type == 'mh':
        g = x * np.exp(-x)
    return g


def kernel_derivative(x, a, b, t1, t2):
    return x


def kernel_abspline3(x, alpha, beta, t1, t2):
    # Convert to array if x is scalar, so we can use fminbound
    if np.isscalar(x):
        x = np.array(x, ndmin=1)

    r = np.zeros(x.size)

    # Compute spline coefficients
    # M a = v
    M = np.array([[1, t1, t1 ** 2, t1 ** 3],
                  [1, t2, t2 ** 2, t2 ** 3],
                  [0, 1, 2 * t1, 3 * t1 ** 2],
                  [0, 1, 2 * t2, 3 * t2 ** 2]])
    v = np.array([[1],
                  [1],
                  [t1 ** (-alpha) * alpha * t1 ** (alpha - 1)],
                  [-beta * t2 ** (-beta - 1) * t2 ** beta]])
    a = np.linalg.lstsq(M, v)[0]

    r1 = np.logical_and(x >= 0, x < t1).nonzero()
    r2 = np.logical_and(x >= t1, x < t2).nonzero()
    r3 = (x >= t2).nonzero()
    r[r1] = x[r1] ** alpha * t1 ** (-alpha)
    r[r3] = x[r3] ** (-beta) * t2 ** (beta)
    x2 = x[r2]
    r[r2] = a[0] + a[1] * x2 + a[2] * x2 ** 2 + a[3] * x2 ** 3

    return r


def filter_design(l_max, N_scales, design_type='default', lp_factor=20,
                  a=2, b=2, t1=1, t2=2):
    g = []
    gp = []
    l_min = l_max / lp_factor
    t = set_scales(l_min, l_max, N_scales)
    if design_type == 'default':
        # Find maximum of gs. Could get this analytically, but this also works
        f = lambda x: -kernel(x, a=a, b=b, t1=t1, t2=t2)
        x_star = fminbound(f, 1, 2)
        gamma_l = -f(x_star)
        l_min_fac = 0.6 * l_min
        g.append(lambda x: gamma_l * np.exp(-(x / l_min_fac) ** 4))
        gp.append(lambda x: -4 * gamma_l * (x / l_min_fac) ** 3 *
                            np.exp(-(x / l_min_fac) ** 4) / l_min_fac)
        for scale in t:
            g.append(lambda x, s=scale: kernel(s * x, a=a, b=b, t1=t1, t2=t2))
            gp.append(lambda x, s=scale: kernel_derivative(scale * x) * s)
    elif design_type == 'mh':
        l_min_fac = 0.4 * l_min
        g.append(lambda x: 1.2 * np.exp(-1) * np.exp(-(x / l_min_fac) ** 4))
        for scale in t:
            g.append(lambda x, s=scale: kernel(s * x, g_type='mh'))
        # TODO: Raise exception

    return (g, gp, t)


def cheby_coeff(g, m, N=None, arange=(-1, 1)):
    if N is None:
        N = m + 1

    a1 = (arange[1] - arange[0]) / 2.0
    a2 = (arange[1] + arange[0]) / 2.0
    n = np.pi * (np.r_[1:N + 1] - 0.5) / N
    s = g(a1 * np.cos(n) + a2)
    c = np.zeros(m + 1)
    for j in range(m + 1):
        c[j] = np.sum(s * np.cos(j * n)) * 2 / N

    return c


def delta(N, j):
    r = np.zeros((N,1), dtype=int)
    r[j] = 1
    return r


def cheby_op(f, L, c, arange):
    if not isinstance(c, list) and not isinstance(c, tuple):
        r = cheby_op(f, L, [c], arange)
        return r[0]

    N_scales = len(c)
    M = np.array([coeff.size for coeff in c])
    max_M = M.max()

    a1 = (arange[1] - arange[0]) / 2
    a2 = (arange[1] + arange[0]) / 2

    Twf_old = f
    Twf_cur = (L * f - a2 * f) / a1
    r = [0.5 * c[j][0] * Twf_old + c[j][1] * Twf_cur for j in range(N_scales)]

    for k in range(1, max_M):
        Twf_new = (2 / a1) * (L * Twf_cur - a2 * Twf_cur) - Twf_old
        for j in range(N_scales):
            if 1 + k <= M[j] - 1:
                r[j] = r[j] + c[j][k + 1] * Twf_new

        Twf_old = Twf_cur
        Twf_cur = Twf_new

    return r


def framebounds(g, lmin, lmax):
    N = 1e4  # number of points for line search
    x = np.linspace(lmin, lmax, N)
    Nscales = len(g)

    sg2 = np.zeros(x.size)
    for ks in range(Nscales):
        sg2 += (g[ks](x)) ** 2

    A = np.min(sg2)
    B = np.max(sg2)

    return (A, B, sg2, x)


def view_design(g, t, arange):
    x = np.linspace(arange[0], arange[1], 1e3)
    h = plt.figure()

    J = len(g)
    G = np.zeros(x.size)

    for n in range(J):
        if n == 0:
            lab = 'h'
        else:
            lab = 't=%.2f' % t[n - 1]
        plt.plot(x, g[n](x), label=lab)
        G += g[n](x) ** 2

    plt.plot(x, G, 'k', label='G')

    (A, B, _, _) = framebounds(g, arange[0], arange[1])
    plt.axhline(A, c='m', ls=':', label='A')
    plt.axhline(B, c='g', ls=':', label='B')
    plt.xlim(arange[0], arange[1])

    plt.title('Scaling function kernel h(x), Wavelet kernels g(t_j x) \n'
              'sum of Squares G, and Frame Bounds')
    plt.yticks(np.r_[0:4])
    plt.ylim(0, 3)
    plt.legend()

    return h

def ftsd(f, g, t, L):
    '''
    TODO
    Compute forward transform by explicitly computing eigenvectors and
    eigenvalues of graph laplacian
     Uses persistent variables to store eigenvectors, so decomposition
    will be computed only on first call

    Inputs:
    f - input data
    g - sgw kernel
    t - desired wavelet scale
    L - graph laplacian

    Outputs:
    r - output wavelet coefficients
    '''
    tem1 = []
    tem2 = []
    for i in range(np.shape(L)[0]):
        for j in range(np.shape(L)[0]):
            tem1.append(L[i, j])
        tem2.append(tem1)
        tem1 = []
    L = tem2

    V, D = np.linalg.eig(np.array(L))
    lambda1 = np.diag(D)
    fhat = V.T*f
    # a1 = np.dot(fhat, g(np.array(t[0])*np.array(lambda1)))
    # r = V*(np.dot(fhat, g(t[0]*lambda1)))

    return r

if __name__ == '__main__':
    A = pd.read_csv('tf_20200106_71s6_7_1armadist.csv', header=None)
    N = len(A.iloc[0, :])
    jcenter = 1
    d = delta(N, jcenter)
    L = lil_matrix(A)
    lmax = rough_l_max(L)
    # lmax = 4.039353399475705
    Nscales = 3
    (g, gp, t) = filter_design(lmax, Nscales)
    m = 50
    arange = (0, lmax)
    c = [cheby_coeff(g[i], m, m + 1, arange) for i in range(len(g))]
    wpall = cheby_op(d, L, c, arange)
    # wp_e = ftsd(d, kernel_abspline3(x, a=2, b=2, t1=1, t2=2), t, L)
    print(np.size(wpall), wpall)
    for i in range(Nscales + 1):
        plt.subplot(Nscales + 1, 1, i + 1)
        plt.plot(wpall[i])
    plt.show()

    print('Done')
import scipy.signal
import scipy.optimize
import numpy as np
import sklearn.decomposition as skl_decomposition

PARAMETERS = np.array([np.logspace(np.log10(0.2), np.log10(2.2), num=10) * 4.0 * np.pi,
                       np.logspace(4.0 + np.log10(5.0), 5.0 + np.log10(6.0), num=10)])
N_COMPONENTS = 10

# TODO USE ENUMERATE INSTEAD OF FOR I IN RANGE(LEN(#))


def scattering_correction(A_app, Z_ref, wavenumbers, parameters=PARAMETERS):
    Z = Z_ref
    alpha_0, gamma = parameters
    Q_ext = np.zeros((len(alpha_0)*len(gamma), len(wavenumbers)))

    # TODO put everything in functions and well organized
    # TODO reduce the number of for loops using np.sum()
    ns_im = np.divide(Z, wavenumbers)
    ns_re = np.real(scipy.signal.hilbert(ns_im))  # not sure about the -
    n_index = 0
    for i in range(len(alpha_0)):
        for j in range(len(gamma)):
            for k in range(len(A_app)):
                rho = alpha_0[i] * (1.0 + gamma[j] * ns_re[k]) * wavenumbers[k]
                beta = np.arctan(ns_im[k] / (1 / gamma[j] + ns_re[k]))
                Q_ext[n_index][k] = 2.0 - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * (np.cos(beta) / rho) * \
                    np.sin(rho - beta) - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * \
                    (np.cos(beta) / rho) ** 2.0 * np.cos(rho - 2.0 * beta)
                # TODO reescriure aixo pq entri en una sola linia
            n_index += 1

    # orthogonalize Q_ext wrt Z_ref
    for i in range(n_index):
        Q_ext[i][:] -= np.dot(Q_ext[i][:], Z_ref) / np.linalg.norm(Z_ref) ** 2 * Z_ref

    pca = skl_decomposition.IncrementalPCA(n_components=N_COMPONENTS)
    pca.fit(Q_ext)
    p_i = pca.components_

    def fit_fun(x, bb, cc, *args):
        return apparent_spectrum_fit_function(x, Z_ref, p_i, bb, cc, *args)

    fit_parameters = scipy.optimize.curve_fit(fit_fun, range(len(wavenumbers)), A_app, p0=np.ones(2 + N_COMPONENTS))

    popt = fit_parameters[0]
    b, c, g_i = popt[0], popt[1], popt[2:]
    Z_corr = np.zeros(np.shape(Z_ref))
    for i in range(len(wavenumbers)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        Z_corr[i] = (A_app[i] - c - sum1)/b

    return Z_corr


def apparent_spectrum_fit_function(i, Z_ref, p_i, b, c, *g_i):
    sum1 = 0
    if np.shape(g_i) == (1, N_COMPONENTS):
        g_i = g_i[0]
    for j in range(len(g_i)):
        sum1 += g_i[j] * p_i[j][i]
    Z_corr = b * Z_ref[i] + c + sum1
    return Z_corr


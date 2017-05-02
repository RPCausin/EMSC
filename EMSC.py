import numpy as np
import scipy.optimize
import sklearn.decomposition as skl_decomposition


def konevskikh_parameters(a, n0, f):
    alpha0 = 4.0 * np.pi * a * (n0 - 1.0)
    gamma = np.divide(f, n0 - 1.0)
    return alpha0, gamma


def GramSchmidt(V):
    V = np.array(V)
    U = np.zeros(np.shape(V))

    for k in range(len(V)):
        sum1 = 0
        for j in range(k):
            sum1 += np.dot(V[k], U[j]) / np.dot(U[j], U[j]) * U[j]
        U[k] = V[k] - sum1
    return U


# Per comprovar ortogonalitat:
def check_orthogonality(U):
    for i in range(len(U)):
        for j in range(i, len(U)):
            if i != j:
                print(np.dot(U[i], U[j]))


def find_nearest_number_index(array, value):
    array = np.array(array)
    if np.shape(np.array(value)) is ():
        index = (np.abs(array - value)).argmin()
    else:
        value = np.array(value)
        index = np.zeros(np.shape(value))
        k = 0
        for val in value:
            index[k] = (np.abs(array - val)).argmin()
            k += 1
        index = index.astype(int)
    return index


def Q_ext_kohler(wn, alpha):
    rho = alpha * wn
    return 2.0 - (4.0 / rho) * np.sin(rho) + (2.0 / rho) ** 2.0 * (1.0 - np.cos(rho))


def apparent_spectrum_fit_function(wn, Z_ref, p, b, c, g):
    A = b * Z_ref + c + np.dot(g, p)
    # print(b, c, g1, g2, g3, g4, g5, g6)
    return A


def Kohler(wavenumbers, App, m0, n_components=8):
    wn = np.copy(wavenumbers)
    A_app = np.copy(App)
    m_0 = np.copy(m0)
    ii = np.argsort(wn)
    wn = wn[ii]
    A_app = A_app[ii]
    m_0 = m_0[ii]
    # n_components = 6
    alpha = np.linspace(3.14, 49.95, 150) * 1.0e-4
    p0 = np.ones(2 + n_components)
    Q_ext = np.zeros((np.size(alpha), np.size(wn)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(wn, alpha=alpha[i])
    pca = skl_decomposition.IncrementalPCA(n_components=n_components)
    pca.fit(Q_ext)
    p_i = pca.components_

    # print(np.sum(pca.explained_variance_ratio_)*100)

    def min_fun(x):
        bb, cc, g = x[0], x[1], x[2:]
        return np.linalg.norm(A_app - apparent_spectrum_fit_function(wn, m_0, p_i, bb, cc, g)) ** 2.0

    # p0 = np.array([0.6, 0.3, 1, 1, 1, 1, 1, 1])

    # bounds = [(0, 1.0), (-1.0, 1.0)]
    # for i in range(n_components):
    #    bounds.append((-1e5, 1e5))

    # res = scipy.optimize.basinhopping(min_fun, p0, niter=1000)
    # res = scipy.optimize.differential_evolution(min_fun, bounds, maxiter=1000)
    res = scipy.optimize.minimize(min_fun, p0, bounds=None, method='Powell')
    # print(res)
    # assert(res.success) # Raise AssertionError if res.success == False


    b, c, g_i = res.x[0], res.x[1], res.x[2:]

    Z_corr = np.zeros(np.shape(m_0))
    for i in range(len(wavenumbers)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        Z_corr[i] = (A_app[i] - c - sum1) / b

    return Z_corr[::-1]


def Konevskikh(wavenumbers, App, m0, n_components=8, iterations=1):
    from scipy.signal import hilbert
    #    from scipy.fftpack import fft, ifft


    wn = np.copy(wavenumbers)
    A_app = np.copy(App)
    m_0 = np.copy(m0)
    ii = np.argsort(wn)
    wn = wn[ii]
    A_app = A_app[ii]
    m_0 = m_0[ii]

    # Normalization
    #    A_app /= np.sqrt(np.sum(A_app ** 2))
    #    m_0 /= np.sqrt(np.sum(m_0 ** 2))

    # n_components = 6
    alpha_0, gamma = np.array([np.logspace(np.log10(0.1), np.log10(2.2), num=10) * 4.0e-4 * np.pi,
                               np.logspace(np.log10(0.05e4), np.log10(0.05e5), num=10) * 1.0e-2])
    # alpha_0, gamma = np.array([np.linspace(0.2, 2.2, num=10) * 4.0e-4 * np.pi, np.linspace(5.0e4, 6.0e5,
    # num=10) * 1.0e-2])
    p0 = np.ones(2 + n_components)
    Q_ext = np.zeros((len(alpha_0) * len(gamma), len(wn)))

    m_n = np.copy(m_0)
    for n_iteration in range(iterations):
        ns_im = np.divide(m_n, wn)
        ns_re = -1.0 * np.imag(hilbert(ns_im))
        #   ns_re = np.real(ifft(fft(np.divide(-1.0, np.pi * wn)) * fft(ns_im)))

        n_index = 0
        for i in range(len(alpha_0)):
            for j in range(len(gamma)):
                for k in range(len(A_app)):
                    rho = alpha_0[i] * (1.0 + gamma[j] * ns_re[k]) * wn[k]
                    beta = np.arctan(ns_im[k] / (1.0 / gamma[j] + ns_re[k]))
                    Q_ext[n_index][k] = 2.0 - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * (np.cos(beta) / rho) * \
                        np.sin(rho - beta) - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * \
                        (np.cos(beta) / rho) ** 2.0 * np.cos(rho - 2.0 * beta) + \
                        4.0 * (np.cos(beta) / rho) ** 2.0 * np.cos(2.0 * beta)
                    # TODO reescriure aixo pq entri en una sola linia

                n_index += 1

        # orthogonalize Q_ext wrt Z_ref
        for i in range(n_index):
            Q_ext[i][:] -= np.dot(Q_ext[i][:], m_0) / np.linalg.norm(m_0) ** 2.0 * m_0
        # Q_ext = GramSchmidt(np.copy(Q_ext))

        pca = skl_decomposition.IncrementalPCA(n_components=n_components)
        pca.fit(Q_ext)
        p_i = pca.components_

        # print(np.sum(pca.explained_variance_ratio_)*100)

        #    import matplotlib.pyplot as plt
        #    plt.figure(2)
        #    for i in range(len(p_i)):
        #        plt.plot(wn, p_i[i])
        #    plt.plot(wn, ns_im, label='ns$_{im}$')
        #    plt.plot(wn, ns_re, label='ns$_{re}$')
        #    plt.gca().invert_xaxis()
        #    plt.xlabel("Wavenumber (cm$^{-1}$)")
        #    plt.ylabel("PCs (a.u.)")
        #    plt.show()
        #    plt.figure(2).tight_layout()

        def min_fun(x):
            bb, cc, g = x[0], x[1], x[2:]
            return np.linalg.norm(A_app - apparent_spectrum_fit_function(wn, m_0, p_i, bb, cc, g)) ** 2.0

        # p0 = np.array([0.6, 0.3, 1, 1, 1, 1, 1, 1])

        # bounds = [(0, 1.0), (-1.0, 1.0)]
        # for i in range(n_components):
        #    bounds.append((-1e5, 1e5))

        # res = scipy.optimize.basinhopping(min_fun, p0, niter=1000)
        # res = scipy.optimize.differential_evolution(min_fun, bounds, maxiter=1000)
        res = scipy.optimize.minimize(min_fun, p0, method='Powell')
        # print(res)
        # assert(res.success) # Raise AssertionError if res.success == False

        b, c, g_i = res.x[0], res.x[1], res.x[2:]

        Z_corr = (A_app - c - np.dot(g_i, p_i)) / b

        m_n = np.copy(Z_corr)

    return Z_corr[::-1]

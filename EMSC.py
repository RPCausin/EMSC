from scipy.fftpack import fft, ifft
import scipy.optimize
import numpy as np
import sklearn.decomposition as skl_decomposition

# PARAMETERS = np.array([np.logspace(np.log10(0.2e-4), np.log10(2.2e-4), num=10) * 4.0 * np.pi,
#                       np.logspace(4.0 + np.log10(5.0e-2), 5.0 + np.log10(6.0e-2), num=10)])

PARAMETERS = np.array([np.logspace(np.log10(0.1), np.log10(6.0), num=10) * 4.0e-4 * np.pi,
    np.logspace(np.log10(1e3), np.log10(2e5), num=10)])

ALPHA = np.linspace(3.14, 49.95, 150)*1.0e-4 # cm

REGIONS = ([1780, 2683], [3680, 4000])

# TODO USE ENUMERATE INSTEAD OF FOR I IN RANGE(LEN(#))


def find_nearest_number_index(array, value):
    return (np.abs(array - value)).argmin()


def check_value_in_region(value, r1, r2):
    r = np.sort([r1, r2])
    if (value > r[0]) and (value < r[1]):
        return True
    else:
        return False


def apply_regions(m_0_in, wn, reg=REGIONS):
    m_0_out = np.copy(m_0_in)
    regions = np.copy(reg)
    for region in regions:
        for idx, value in np.ndenumerate(region):
            if type(value) is np.str_:
                region[idx[0]] = np.int(value)
            elif (issubclass(np.int64, type(value))) or (issubclass(np.float64, type(value))):
                region[idx[0]] = find_nearest_number_index(wn, value)
        if region[0] > region[1]:
            for idx, value in np.ndenumerate(wn):
                if check_value_in_region(value, region[0], region[1]):
                    m_0_out[idx] = 0.0
    
    
#    wn_out = np.array([])
#    m_0_out = np.array([])
#    regions = reg
#    for region in regions:
#        for idx, value in np.ndenumerate(region):
#            if type(value) is np.str_:
#                region[idx[0]] = np.int(value)
#            elif (issubclass(np.int64, type(value))) or (issubclass(np.float64, type(value))):
#                region[idx[0]] = find_nearest_number_index(wn, value)
#        if region[0] > region[1]:
#            wn_out = np.append(wn_out, wn[region[0]:region[1]:-1])
#            m_0_out = np.append(m_0_out, m_0[region[0]:region[1]:-1])
#        else:
#            wn_out = np.append(wn_out, wn[region[0]:region[1]])
#            m_0_out = np.append(m_0_out, m_0[region[0]:region[1]])
    return m_0_out # , wn_out


def Q_ext_kohler(wn, alpha=None, *args):
    if alpha is None:
        r, n_r = args
        alpha = 4 * np.pi * r * (n_r - 1)
    rho = alpha * wn
    return 2 - (4 / rho) * np.sin(rho) + (2 / rho) ** 2 * (1 - np.cos(rho))


def fit_fun_reference(i, p_i, c, N_COMPONENTS, *g_i):
    sum1 = 0
    if np.shape(g_i) == (1, N_COMPONENTS):
        g_i = g_i[0]
    for j in range(len(g_i)):
        sum1 += g_i[j] * p_i[j][i]
    return c + sum1


def apparent_spectrum_fit_function(i, Z_ref, p_i, b, c, N_COMPONENTS, *g_i):
    sum1 = 0
    print(p_i)
    if np.shape(g_i) == (1, N_COMPONENTS):
        g_i = g_i[0]
    for j in range(len(g_i)):
        sum1 += g_i[j] * p_i[j][i]
    return b * Z_ref[i] + c + sum1


def no_reference_correction(m_0, wavenumbers, regions=REGIONS, alpha=ALPHA, N_COMPONENTS=6, fit_parameters=None):
    wn = wavenumbers
    if fit_parameters is None:
        fit_parameters = (0.5 * np.ones(1 + N_COMPONENTS))
    m = apply_regions(m_0, wavenumbers)
    Q_ext = np.zeros((np.size(alpha), np.size(wn)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(wn, alpha=alpha[i])
    pca = skl_decomposition.IncrementalPCA(n_components=N_COMPONENTS)
    pca.fit(Q_ext)
    p_i = pca.components_

    def fit_fun(x, aa, *args):
        return fit_fun_reference(x, p_i, aa, N_COMPONENTS, *args)

    f_params = scipy.optimize.curve_fit(fit_fun, range(len(wn)), m, p0=fit_parameters)
    popt = f_params[0]
    a, g_i = popt[0], popt[1:]
    m_corr = np.zeros(np.shape(m_0))
    Q_fit = np.zeros(np.shape(m_0))
    for i in range(len(wavenumbers)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        Q_fit[i] = a + sum1
        m_corr[i] = m_0[i] - Q_fit[i]

    return m_corr, Q_fit


def scattering_correction_kohler(*args):
    raise NotImplementedError


def scattering_correction_bassan(*args):
    raise NotImplementedError


def scattering_correction_konesvkikh(A_app, Z_ref, wavenumbers, parameters=PARAMETERS, fit_parameters=None, N_COMPONENTS=10):
    alpha_0, gamma = parameters
    Q_ext = np.zeros((len(alpha_0)*len(gamma), len(wavenumbers)))
    
    if fit_parameters is None:
        fit_parameters = (1.0 * np.ones(2 + N_COMPONENTS))

    # TODO put everything in functions and well organized
    # TODO reduce the number of for loops using np.sum()

    ns_im = np.divide(Z_ref, wavenumbers)
    ns_re = np.real(ifft(fft(np.divide(-1.0, np.pi * wavenumbers)) * fft(ns_im)))
    # Usually im are 1e-22 but this should be checked
    n_index = 0
    for i in range(len(alpha_0)):
        for j in range(len(gamma)):
            for k in range(len(A_app)):
                rho = alpha_0[i] * (1.0 + gamma[j] * ns_re[k]) * wavenumbers[k]
#                beta = np.arctan(ns_im[k] / (1 / gamma[j] + ns_re[k]))
#                Q_ext[n_index][k] = 2.0 - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * (np.cos(beta) / rho) * \
#                    np.sin(rho - beta) - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * \
#                    (np.cos(beta) / rho) ** 2.0 * np.cos(rho - 2.0 * beta) + \
#                    4 * (np.cos(beta) / rho) ** 2 * np.cos(2 * beta)
                # TODO reescriure aixo pq entri en una sola linia
                
                tanB = ns_im[k] / (1 / gamma[j] + ns_re[k])
                cosB = np.sqrt(1 / (1 + tanB ** 2))
                sinB = np.sqrt(1 - cosB ** 2)
                sinrmB = np.sin(rho) * cosB - sinB * np.cos(rho)
                cos2B = cosB ** 2 - sinB ** 2
                sin2B = 2.0 * sinB * cosB
                cosrm2B = np.cos(rho) * cos2B + np.sin(rho) * sin2B
                Q_ext[n_index][k] = 2.0 - 4.0 * np.exp(-1.0 * rho * tanB) * (cosB / rho) * sinrmB - \
                    4.0 * np.exp(-1.0 * rho * tanB) * (cosB / rho) ** 2 * cosrm2B + 4.0 * (cosB / rho) ** 2 * cos2B
                
            n_index += 1

    # orthogonalize Q_ext wrt Z_ref
    for i in range(n_index):
        Q_ext[i][:] -= np.dot(Q_ext[i][:], Z_ref) / np.linalg.norm(Z_ref) ** 2 * Z_ref

    pca = skl_decomposition.IncrementalPCA(n_components=N_COMPONENTS)
    pca.fit(Q_ext)
    p_i = pca.components_
    print(np.sum(pca.explained_variance_ratio_)*100)

    def fit_fun(x, bb, cc, *args):
        return apparent_spectrum_fit_function(x, Z_ref, p_i, bb, cc, N_COMPONENTS, *args)

    fit_parameters = scipy.optimize.curve_fit(fit_fun, range(len(wavenumbers)),
                                              A_app, p0=fit_parameters, method='dogbox')

    popt = fit_parameters[0]
    print(popt)
    b, c, g_i = popt[0], popt[1], popt[2:]
    Z_corr = np.zeros(np.shape(Z_ref))
    for i in range(len(wavenumbers)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        Z_corr[i] = (A_app[i] - c - sum1)/b

    return Z_corr

import numpy as np
import scipy.optimize
import sklearn.decomposition as skl_decomposition
from scipy.signal import hilbert


def konevskikh_parameters(a, n0, f):
    """
    Compute parameters for Konevskikh algorithm
    :param a: cell radius
    :param n0: refractive index
    :param f: scaling factor
    :return: parameters alpha0 and gamma
    """
    alpha0 = 4.0 * np.pi * a * (n0 - 1.0)
    gamma = np.divide(f, n0 - 1.0)
    return alpha0, gamma


def GramSchmidt(V):
    """
    Perform Gram-Schmidt normalization for the matrix V
    :param V: matrix
    :return: nGram-Schmidt normalized matrix
    """
    V = np.array(V)
    U = np.zeros(np.shape(V))

    for k in range(len(V)):
        sum1 = 0
        for j in range(k):
            sum1 += np.dot(V[k], U[j]) / np.dot(U[j], U[j]) * U[j]
        U[k] = V[k] - sum1
    return U


def check_orthogonality(U):
    """
    Check orthogonality of a matrix
    :param U: matrix
    """
    for i in range(len(U)):
        for j in range(i, len(U)):
            if i != j:
                print(np.dot(U[i], U[j]))


def find_nearest_number_index(array, value):
    """
    Find the nearest number in an array and return its index
    :param array:
    :param value: value to be found inside the array
    :return: position of the number closest to value in array
    """
    array = np.array(array)  # Convert to numpy array
    if np.shape(np.array(value)) is ():  # If only one value wants to be found:
        index = (np.abs(array - value)).argmin()  # Get the index of item closest to the value
    else:  # If value is a list:
        value = np.array(value)
        index = np.zeros(np.shape(value))
        k = 0
        # Find the indexes for all values in value
        for val in value:
            index[k] = (np.abs(array - val)).argmin()
            k += 1
        index = index.astype(int)  # Convert the indexes to integers
    return index


def Q_ext_kohler(wn, alpha):
    """
    Compute the scattering extinction values for a given alpha and a range of wavenumbers
    :param wn: array of wavenumbers
    :param alpha: scalar alpha
    :return: array of scattering extinctions calculated for alpha in the given wavenumbers
    """
    rho = alpha * wn
    Q = 2.0 - (4.0 / rho) * np.sin(rho) + (2.0 / rho) ** 2.0 * (1.0 - np.cos(rho))
    return Q


def apparent_spectrum_fit_function(wn, Z_ref, p, b, c, g):
    """
    Function used to fit the apparent spectrum
    :param wn: wavenumbers
    :param Z_ref: reference spectrum
    :param p: principal components of the extinction matrix
    :param b: Reference's linear factor
    :param c: Offset
    :param g: Extinction matrix's PCA scores (to be fitted)
    :return: fitting of the apparent specrum
    """
    A = b * Z_ref + c + np.dot(g, p)  # Extended multiplicative scattering correction formula
    return A


def reference_spectrum_fit_function(wn, p, c, g):
    """
    Function used to fit a reference spectrum (without using another spectrum as reference).
    :param wn: wavenumbers
    :param p: principal components of the extinction matrix
    :param c: offset
    :param g: PCA scores (to be fitted)
    :return: fitting of the reference spectrum
    """
    A = c + np.dot(g, p)
    return A


def apparent_spectrum_fit_function_Bassan(wn, Z_ref, p, c, m, h, g):
    """
    Function used to fit the apparent spectrum in Bassan's algorithm
    :param wn: wavenumbers
    :param Z_ref: reference spectrum
    :param p: principal componetns of the extinction matrix
    :param c: offset
    :param m: linear baseline
    :param h: reference's linear factor
    :param g: PCA scores to be fitted
    :return: fitting of the apparent spectrum
    """
    A = c + m * wn + h * Z_ref + np.dot(g, p)
    return A


def correct_reference(m, wn, a, d, w_regions):
    """
    Correct reference spectrum as in Kohler's method
    :param m: reference spectrum
    :param wn: wavenumbers
    :param a: Average refractive index range
    :param d: Cell diameter range
    :param w_regions: Weighted regions
    :return: corrected reference spectrum
    """
    n_components = 6  # Set the number of principal components

    # Copy the input variables
    m = np.copy(m)
    wn = np.copy(wn)

    # Compute the alpha range:
    alpha = 4.0 * np.pi * 0.5 * np.linspace(np.min(d) * (np.min(a) - 1.0), np.max(d) * (np.max(a) - 1.0), 150)

    p0 = np.ones(1 + n_components)  # Initial guess for the fitting

    # Compute extinction matrix
    Q_ext = np.zeros((np.size(alpha), np.size(wn)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(wn, alpha=alpha[i])

    # Perform PCA to Q_ext
    pca = skl_decomposition.IncrementalPCA(n_components=n_components)
    pca.fit(Q_ext)
    p_i = pca.components_  # Get the principal components of the extinction matrix

    # Get the weighted regions of the wavenumbers, the reference spectrum and the principal components
    w_indexes = []
    for pair in w_regions:
        min_pair = min(pair)
        max_pair = max(pair)
        ii1 = find_nearest_number_index(wn, min_pair)
        ii2 = find_nearest_number_index(wn, max_pair)
        w_indexes.extend(np.arange(ii1, ii2))
    wn_w = np.copy(wn[w_indexes])
    m_w = np.copy(m[w_indexes])
    p_i_w = np.copy(p_i[:, w_indexes])

    def min_fun(x):
        """
        Function to be minimized for the fitting
        :param x: offset and PCA scores
        :return: difference between the spectrum and its fitting
        """
        cc, g = x[0], x[1:]
        # Return the squared norm of the difference between the reference spectrum and its fitting:
        return np.linalg.norm(m_w - reference_spectrum_fit_function(wn_w, p_i_w, cc, g)) ** 2.0

    # Perform the minimization using Powell method
    res = scipy.optimize.minimize(min_fun, p0, bounds=None, method='Powell')

    c, g_i = res.x[0], res.x[1:]  # Obtain the fitted parameters

    # Apply the correction:
    m_corr = np.zeros(np.shape(m))
    for i in range(len(wn)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        m_corr[i] = (m[i] - c - sum1)

    return m_corr  # Return the corrected spectrum


def Kohler(wavenumbers, App, m0, n_components=8):
    """
    Correct scattered spectra using Kohler's algorithm
    :param wavenumbers: array of wavenumbers
    :param App: apparent spectrum
    :param m0: reference spectrum
    :param n_components: number of principal components to be calculated 
    :return: corrected data
    """
    # Make copies of all input data:
    wn = np.copy(wavenumbers)
    A_app = np.copy(App)
    m_0 = np.copy(m0)
    ii = np.argsort(wn)  # Sort the wavenumbers from smallest to largest
    # Sort all the input variables accordingly
    wn = wn[ii]
    A_app = A_app[ii]
    m_0 = m_0[ii]

    # Initialize the alpha parameter:
    alpha = np.linspace(3.14, 49.95, 150) * 1.0e-4  # alpha = 2 * pi * d * (n - 1) * wavenumber
    p0 = np.ones(2 + n_components)  # Initialize the initial guess for the fitting

    # # Initialize the extinction matrix:
    Q_ext = np.zeros((np.size(alpha), np.size(wn)))
    for i in range(np.size(alpha)):
        Q_ext[i][:] = Q_ext_kohler(wn, alpha=alpha[i])

    # Perform PCA of Q_ext:
    pca = skl_decomposition.IncrementalPCA(n_components=n_components)
    pca.fit(Q_ext)
    p_i = pca.components_  # Extract the principal components

    # print(np.sum(pca.explained_variance_ratio_)*100)  # Print th explained variance ratio in percentage

    def min_fun(x):
        """
        Function to be minimized by the fitting
        :param x: array containing the reference linear factor, the offset, and the PCA scores 
        :return: function to be minimized
        """
        bb, cc, g = x[0], x[1], x[2:]
        # Return the squared norm of the difference between the apparent spectrum and the fit
        return np.linalg.norm(A_app - apparent_spectrum_fit_function(wn, m_0, p_i, bb, cc, g)) ** 2.0

    # Minimize the function using Powell method
    res = scipy.optimize.minimize(min_fun, p0, bounds=None, method='Powell')
    # print(res)  # Print the minimization result
    # assert(res.success) # Raise AssertionError if res.success == False

    b, c, g_i = res.x[0], res.x[1], res.x[2:]  # Obtain the fitted parameters

    # Apply the correction to the apparent spectrum
    Z_corr = np.zeros(np.shape(m_0))
    for i in range(len(wavenumbers)):
        sum1 = 0
        for j in range(len(g_i)):
            sum1 += g_i[j] * p_i[j][i]
        Z_corr[i] = (A_app[i] - c - sum1) / b

    return Z_corr[::-1]  # Return the correction in reverse order for compatibility


def Bassan(wavenumbers, App, m0, n_components=8, iterations=1, w_regions=None):
    """
    Correct scattered spectra using Bassan's algorithm.
    :param wavenumbers: array of wavenumbers
    :param App: apparent spectrum
    :param m0: reference spectrum
    :param n_components: number of principal components to be calculated for the extinction matrix
    :param iterations: number of iterations of the algorithm
    :param w_regions: the regions to be taken into account for the fitting
    :return: corrected apparent spectrum
    """
    # Copy the input data
    wn = np.copy(wavenumbers)
    A_app = np.copy(App)
    m_0 = np.copy(m0)
    ii = np.argsort(wn)  # Sort the wavenumbers
    # Apply the sorting to the input variables
    wn = wn[ii]
    A_app = A_app[ii]
    m_0 = m_0[ii]

    # Define the weighted regions:
    if w_regions is not None:
        m_0 = correct_reference(np.copy(m_0), wn, a, d, w_regions)  # Correct the reference spectrum as in Kohler method
        w_indexes = []
        # Get the indexes of the regions to be taken into account
        for pair in w_regions:
            min_pair = min(pair)
            max_pair = max(pair)
            ii1 = find_nearest_number_index(wn, min_pair)
            ii2 = find_nearest_number_index(wn, max_pair)
            w_indexes.extend(np.arange(ii1, ii2))
        # Take the weighted regions of wavenumbers, apparent and reference spectrum
        wn_w = np.copy(wn[w_indexes])
        A_app_w = np.copy(A_app[w_indexes])
        m_0_w = np.copy(m_0[w_indexes])

    n_loadings = 10  # Number of values to be computed for each parameter (a, b, d)
    a = np.linspace(1.1, 1.5, n_loadings)  # Average refractive index
    d = np.linspace(2.0, 8.0, n_loadings) * 1.0e-4  # Cell diameter
    Q = np.zeros((n_loadings ** 3, len(wn)))  # Initialize the extinction matrix
    m_n = np.copy(m_0)  # Initialize the reference spectrum, that will be updated after each iteration
    for iteration in range(iterations):
        # Compute the scaled real part of the refractive index by Kramers-Kronig transform:
        nkk = -1.0 * np.imag(hilbert(m_n))
        # Build the extinction matrix
        n_row = 0
        for i in range(n_loadings):
            b = np.linspace(0.0, a[i] - 1.0, 10)  # Range of amplification factors of nkk
            for j in range(n_loadings):
                for k in range(n_loadings):
                    n = a[i] + b[j] * nkk  # Compute the real refractive index
                    alpha = 2.0 * np.pi * d[k] * (n - 1.0)
                    rho = alpha * wn
                    #  Compute the extinction coefficients for each combination of a, b and d:
                    Q[n_row] = 2.0 - np.divide(4.0, rho) * np.sin(rho) + \
                               np.divide(4.0, rho ** 2.0) * (1.0 - np.cos(rho))
                    n_row += 1

        # Orthogonalization of th extinction matrix with respect to the reference spectrum:
        for i in range(n_loadings ** 3):
            Q[i] -= np.dot(Q[i], m_0) / np.linalg.norm(m_0) ** 2.0 * m_0

        # Perform PCA of the extinction matrix
        pca = skl_decomposition.IncrementalPCA(n_components=n_components)
        pca.fit(Q)
        p_i = pca.components_  # Get the principal components

        if w_regions is None:  # If all regions have to be taken into account:
            def min_fun(x):
                """
                Function to be minimized for the fitting
                :param x: fitting parameters (offset, baseline, reference's linear factor, PCA scores)
                :return: squared norm of the difference between the apparent spectrum and its fitting
                """
                cc, mm, hh, g = x[0], x[1], x[2], x[3:]
                return np.linalg.norm(A_app - apparent_spectrum_fit_function_Bassan(wn, m_0, p_i, cc, mm, hh, g)) ** 2.0
        else:  # If only the specified regions have to be taken into account:
            # Take the indexes of the specified regions
            w_indexes = []
            for pair in w_regions:
                min_pair = min(pair)
                max_pair = max(pair)
                ii1 = find_nearest_number_index(wn, min_pair)
                ii2 = find_nearest_number_index(wn, max_pair)
                w_indexes.extend(np.arange(ii1, ii2))
            p_i_w = np.copy(p_i[:, w_indexes])  # Get the principal components of the extinction matrix at the

            # specified regions

            def min_fun(x):
                """
                Function to be minimized for the fitting
                :param x: fitting parameters (offset, baseline, reference's linear factor, PCA scores)
                :return: squared norm of the difference between the apparent spectrum and its fitting
                """
                cc, mm, hh, g = x[0], x[1], x[2], x[3:]
                return np.linalg.norm(A_app_w -
                                      apparent_spectrum_fit_function_Bassan(wn_w, m_0_w, p_i_w, cc, mm, hh, g)) ** 2.0

        p0 = np.append([1.0, 0.0005, 0.9], np.ones(n_components))  # Initial guess for the fitting
        res = scipy.optimize.minimize(min_fun, p0, method='Powell')  # Perform the fitting

        # print(res)  # Print the result of the minimization
        # assert(res.success) # Raise AssertionError if res.success == False

        c, m, h, g_i = res.x[0], res.x[1], res.x[2], res.x[3:]  # Take the fitted parameters

        Z_corr = (A_app - c - m * wn - np.dot(g_i, p_i)) / h  # Apply the correction

        m_n = np.copy(Z_corr)  # Take the corrected spectrum as the reference for the next iteration

    return np.copy(Z_corr[::-1])  # Return the corrected spectrum in inverted order for compatibility


def Konevskikh(wavenumbers, App, m0, n_components=8, iterations=1):
    """
    Correct scattered spectra using Konevskikh algorithm
    :param wavenumbers: array of wavenumbers
    :param App: apparent spectrum
    :param m0: reference spectrum
    :param n_components: number of components
    :param iterations: number of iterations
    :return: corrected spectrum
    """
    # Copy the input variables
    wn = np.copy(wavenumbers)
    A_app = np.copy(App)
    m_0 = np.copy(m0)
    ii = np.argsort(wn)  # Sort the wavenumbers
    wn = wn[ii]
    A_app = A_app[ii]
    m_0 = m_0[ii]

    # Initialize parameters range:
    alpha_0, gamma = np.array([np.logspace(np.log10(0.1), np.log10(2.2), num=10) * 4.0e-4 * np.pi,
                               np.logspace(np.log10(0.05e4), np.log10(0.05e5), num=10) * 1.0e-2])
    p0 = np.ones(2 + n_components)
    Q_ext = np.zeros((len(alpha_0) * len(gamma), len(wn)))  # Initialize extinction matrix

    m_n = np.copy(m_0)  # Copy the reference spectrum
    for n_iteration in range(iterations):
        ns_im = np.divide(m_n, wn)  # Compute the imaginary part of the refractive index
        # Compute the real part of the refractive index by Kramers-Kronig transform
        ns_re = -1.0 * np.imag(hilbert(ns_im))

        # Compute the extinction matrix
        n_index = 0
        for i in range(len(alpha_0)):
            for j in range(len(gamma)):
                for k in range(len(A_app)):
                    rho = alpha_0[i] * (1.0 + gamma[j] * ns_re[k]) * wn[k]
                    beta = np.arctan(ns_im[k] / (1.0 / gamma[j] + ns_re[k]))
                    Q_ext[n_index][k] = 2.0 - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * (np.cos(beta) / rho) * \
                        np.sin(rho - beta) - 4.0 * np.exp(-1.0 * rho * np.tan(beta)) * (np.cos(beta) / rho) ** 2.0 * \
                        np.cos(rho - 2.0 * beta) + 4.0 * (np.cos(beta) / rho) ** 2.0 * np.cos(2.0 * beta)
                    # TODO: rewrite this in a simpler way

                n_index += 1

        # Orthogonalize the extinction matrix with respect to the reference:
        for i in range(n_index):
            Q_ext[i][:] -= np.dot(Q_ext[i][:], m_0) / np.linalg.norm(m_0) ** 2.0 * m_0
        # Q_ext = GramSchmidt(np.copy(Q_ext))  # Apply Gram-Schmidt othogonalization to Q_ext (don't uncomment this)

        # Compute PCA of the extinction matrix
        pca = skl_decomposition.IncrementalPCA(n_components=n_components)
        pca.fit(Q_ext)
        p_i = pca.components_  # Get the principal components

        def min_fun(x):
            bb, cc, g = x[0], x[1], x[2:]
            return np.linalg.norm(A_app - apparent_spectrum_fit_function(wn, m_0, p_i, bb, cc, g)) ** 2.0

        res = scipy.optimize.minimize(min_fun, p0, method='Powell')
        # print(res)  # Print the minimization results
        # assert(res.success) # Raise AssertionError if res.success == False

        b, c, g_i = res.x[0], res.x[1], res.x[2:]  # Get the fitted parameters

        Z_corr = (A_app - c - np.dot(g_i, p_i)) / b  # Apply the correction

        m_n = np.copy(Z_corr)  # Update te reference with the correction

    return Z_corr[::-1]  # Return the corrected spectrum

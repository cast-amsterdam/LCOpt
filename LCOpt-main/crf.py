import math
import numpy as np
import scipy.stats as ss
import heapq

from tqdm import tqdm


def resolution(tR1, tR2, W1, W2):
    """Return the resolution of 2 peaks, given tR and W for both peaks."""
    resolution = ((2*abs(tR2-tR1))/(W1+W2))
    return(resolution)


def sigmoid(x, a, b):
    """Return a sigmoidal transformation of x."""
    sigmoid = 1/(1 + np.exp(-a*x + b))
    return(sigmoid)


def kaiser_ratio(peak_heights, valley_height):
    """
    Given two peak heights and their valley height, computes the Kaiser p-to-v ratio
    :param peak_heights: list of two peak heights
    :param valley_height: height of intermediate valley height
    :return: Kaiser p-to-v ratio
    """
    return 1 - (valley_height / (0.5 * sum(peak_heights)))

def discrimination_factor(peak_heights, valley_height):
    """
    Given two peak heights and their valley height, computes the discrimination factor
    :param peak_heights: list of two peak heights
    :param valley_height: height of intermediate valley height
    :return: discrimination factor
    """
    return 1 - (valley_height / min(peak_heights))

def sort_peaks(retention_times, peak_widths):
    """
    Sort peaks based on retention time
    and return sorted retention time list and peak width list.
    """
    number_of_peaks = len(retention_times)

    # Create a list of tuples, one for each peak (rt, W)
    peak_tuple_list = []
    for i in range(number_of_peaks):
        peak_tuple = (retention_times[i], peak_widths[i])
        peak_tuple_list.append(peak_tuple)
    # Sort according to first element
    peak_tuples_sorted = sorted(peak_tuple_list, key=lambda x: x[0])

    retention_times = []
    peak_widths = []
    for i in range(number_of_peaks):
        retention_times.append(peak_tuples_sorted[i][0])
        peak_widths.append(peak_tuples_sorted[i][1])

    return(retention_times, peak_widths)


def crf(retention_times, peak_widths, phi_list=None):
    """
    Return CRF score for a chromatogram characterized by a list of retention
    times and a corresponding list of peak widths.
    """
    n_peaks = len(retention_times)
    if n_peaks < 2:
        return 0

    # Parameters sigmoidal transformations
    b0 = 3.93
    b1 = 3.66
    b2 = -0.0406
    b3 = -4.646

    resolutions = []
    sigmoid_resolutions = []

    # Sort retention times and peak widths
    retention_times, peak_widths = sort_peaks(retention_times, peak_widths)

    prod_S = 1

    # Loop over all neighboring peak pairs and get S. Multiply together.
    for i in range(n_peaks - 1):
        tR1 = retention_times[i]
        tR2 = retention_times[i+1]
        W1 = peak_widths[i]
        W2 = peak_widths[i+1]

        R = resolution(tR1, tR2, W1, W2)
        S = sigmoid(R, b0, b1)
        prod_S = prod_S * S

        sigmoid_resolutions.append(S)
        resolutions.append(R)

    # Create f and g
    f = prod_S ** (1/(n_peaks-1))

    # Get T
    tR_last = retention_times[-1]
    W_last = peak_widths[-1]
    T = tR_last + 0.5*(W_last)

    g = sigmoid(T, b2, b3)

    score = f * g

    # Optional penalty for gradient segments with negative slope
    # Comment out to remove penalty:
    #if(sorted(phi_list) != phi_list):
        #return(0.8 * score)

    return(score)

def capped_sum_of_resolutions(retention_times, peak_widths, phi_list=None, max_time=60, min_res=0, max_res=1.5):
    """
    Resolution equation as defined in eq. 5 and 6 of
     https://chemrxiv.org/engage/chemrxiv/article-details/62e2a383e7fc8f9e388caabc
    Uses symmetric resolution equation.
    :param retention_times: ndarray containing retention_times
    :param peak_widths: ndarray containing peak widths
    :param phi_list: list of phi points
    :param max_time: int maximum allowed time
    :param min_res: float minimum required resolution
    :param max_res: float maximum required resolution
    :return: float score
    """
    n_peaks = len(retention_times)

    if n_peaks < 2:
        return 1

    resolutions = np.zeros(len(retention_times))

    mask = retention_times < max_time
    for i in range(n_peaks - 1):
        # check if tR1 is later then max_time, if yes we can stop
        tR1 = retention_times[i]
        tR2 = retention_times[i+1]
        W1 = peak_widths[i]
        W2 = peak_widths[i+1]

        resolutions[i] = resolution(tR1, tR2, W1, W2)
        if resolutions[i] < min_res:
            resolutions[i]=0
        if min_res > resolutions[i] > max_res:
            resolutions[i] = resolutions[i]/max_res
        if resolutions[i] >= max_res:
            resolutions[i]=1

    # zero out scores of peaks that have eluted after max_time
    resolutions = resolutions*mask

    score = resolutions.sum()
    # Optional penalty for gradient segments with negative slope
    # Comment out to remove penalty:
        #if(sorted(phi_list) != phi_list):
            #return(0.8 * score)

    return(score)

def product_of_resolutions(retention_times, peak_widths, phi_list=None, max_time=60, min_res=0, max_res=1.5):
    """
    Product of resolutions
    Uses symmetric resolution equation.
    :param retention_times: ndarray containing retention_times
    :param peak_widths: ndarray containing peak widths
    :param phi_list: list of phi points
    :param max_time: int maximum allowed time
    :param min_res: float minimum required resolution
    :param max_res: float maximum required resolution
    :return: float score
    """
    n_peaks = len(retention_times)

    if n_peaks < 2:
        return 1

    resolutions = np.ones(len(retention_times))

    mask = retention_times < max_time
    for i in range(n_peaks - 1):
        # check if tR1 is later than max_time, if yes we can stop
        tR1 = retention_times[i]
        tR2 = retention_times[i+1]
        W1 = peak_widths[i]
        W2 = peak_widths[i+1]

        resolutions[i] = resolution(tR1, tR2, W1, W2)
        if resolutions[i] < min_res:
            resolutions[i]=0
        if min_res > resolutions[i] > max_res:
            resolutions[i] = resolutions[i]/max_res
        if resolutions[i] >= max_res:
            resolutions[i]=1

    # zero out scores of peaks that have eluted after max_time
    resolutions = resolutions*mask

    score = resolutions.prod()
    # Optional penalty for gradient segments with negative slope
    # Comment out to remove penalty:
    #if phi_list is not None:
        #if(sorted(phi_list) != phi_list):
            #return(0.8 * score)

    return(score)

def tyteca_eq_11(retention_times, peak_widths, max_time=60, min_time=2, max_res=1.5, prefacs=[1,1,1]):
    """
    Implements the CRF defined in Tyteca 2014, 10.1016/J.CHROMA.2014.08.014, Category II-A, equation 11.
    :param retention_times: ndarray containing retention_times
    :param peak_widths: ndarray containing peak widths
    :param max_time: int maximum allowed time
    :param min_time: int minimum allowed time
    :param max_res: float maximum required resolution
    :param prefacs: prefactors that dictate importance of each term.
    :return: float CRF score
    """
    n_peaks = len(retention_times)

    if n_peaks < 2:
        return max_res

    nobs_term = n_peaks**prefacs[0]

    resolutions = np.zeros(n_peaks)
    for i in range(n_peaks - 1):
        # check if tR1 is later than max_time, if yes we can stop
        tR1 = retention_times[i]
        tR2 = retention_times[i+1]
        W1 = peak_widths[i]
        W2 = peak_widths[i+1]

        resolutions[i] = resolution(tR1, tR2, W1, W2)
        if resolutions[i] > max_res:
            resolutions[i] = max_res

    res_term = resolutions.sum()
    max_time_term = prefacs[1] + np.abs(max_time - np.max(retention_times))
    min_time_term = prefacs[2] * (min_time-np.min(retention_times))

    return nobs_term + res_term - max_time_term + min_time_term

def tyteca_eq_24(retention_times, peak_widths, max_res=1.5):
    """
    Implements the CRF defined in Tyteca 2014, 10.1016/J.CHROMA.2014.08.014, Category I-B, equation 24.
    :param retention_times: ndarray containing retention_times
    :param peak_widths: ndarray containing peak widths
    :param max_res: float maximum required resolution
    :return: float CRF score
    """
    n_peaks = len(retention_times)

    if n_peaks < 2:
        return max_res

    resolutions = np.zeros(n_peaks)
    for i in range(n_peaks - 1):
        # check if tR1 is later than max_time, if yes we can stop
        tR1 = retention_times[i]
        tR2 = retention_times[i+1]
        W1 = peak_widths[i]
        W2 = peak_widths[i+1]

        resolutions[i] = resolution(tR1, tR2, W1, W2)
        if resolutions[i] > max_res:
            resolutions[i] = max_res
        res_term = np.sum(resolutions)

        return n_peaks + (res_term / (max_res*(n_peaks-1)))


def sum_of_kaiser(retention_times, peak_heights, valley_heights, max_time=60):
    """
    Takes the sum  of Kaiser ratio between neighboring peaks.
    :param retention_times: list of retention times
    :param peak_heights: list of peak heights
    :param valley_heights: list of valley heights
    :param max_time: float, ignores compounds eluting after this time.
    :return:
    """
    score = 0
    for i in range(len(peak_heights)-1):
        if retention_times[i] < max_time:
            score += kaiser_ratio([peak_heights[i], peak_heights[i+1]], valley_heights[i])
    return score

def prod_of_kaiser(retention_times, peak_heights, valley_heights, max_time=60):
    """
    Should resemble eq. 5 of Tyteca with max_time.
    :param retention_times: list of  retention times
    :param peak_heights: list of peak heights
    :param valley_heights: list of valley heights
    :param max_time: float, ignores compounds eluting after this time.
    :return: product of Kaiser peak-to-valley ratios
    """
    score = 1
    for i in range(len(peak_heights)-1):
        if retention_times[i] < max_time:
            score *= kaiser_ratio([peak_heights[i], peak_heights[i+1]], valley_heights[i])
    return score

def tyteca_eq_28(retention_times, peak_heights, valley_heights, a=0.5, b=0.5, max_time=60):
    """
    Implements the CRF defined in Tyteca 2014, 10.1016/J.CHROMA.2014.08.014, Category I-B, equation 28.

    :param retention_times:
    :param peak_heights:
    :param valley_heights:
    :param a:
    :param b:
    :return:
    """
    nobs = len(retention_times)
    if nobs < 2:
        return 0
    sum_kaiser = sum_of_kaiser(retention_times, peak_heights, valley_heights, max_time)
    prod_kaiser = prod_of_kaiser(retention_times, peak_heights, valley_heights, max_time)
    root = ((nobs-1) ** prod_kaiser)**(1/(nobs-1))

    return nobs + a * root + b * ((nobs-1)**sum_kaiser)/(nobs-1)

def tyteca_eq_35(retention_times, peak_heights, valley_heights, deadtime):
    """
    Implements the CRF defined in Tyteca 2014, 10.1016/J.CHROMA.2014.08.014, Category II-B, equation 35.

    :param retention_times:
    :param peak_heights:
    :param valley_heights:
    :param deadtime:
    :return:
    """
    nobs = len(retention_times)
    if nobs < 1:
        return 0
    kaiser = sum_of_kaiser(retention_times, peak_heights, valley_heights, max_time=10000)
    last = max(retention_times)
    time = (last - deadtime) / last
    return nobs + kaiser - time

def tyteca_eq_40(retention_times, peak_heights, valley_heights):
    """
    Implements the CRF defined in Tyteca 2014, 10.1016/J.CHROMA.2014.08.014, Category II-B, equation 40.

    :param retention_times:
    :param peak_heights:
    :param valley_heights:
    :return: crf score
    """
    nobs = len(retention_times)
    if nobs < 1:
        return 0
    kaiser = prod_of_kaiser(retention_times, peak_heights, valley_heights, max_time=10000)
    last = max(retention_times)
    return nobs + 1/last * kaiser

################################################################################
################################################################################
################################# PEAK PURITY ##################################
################################################################################
################################################################################


#https://stats.stackexchange.com/questions/311592/how-to-find-the-point-where-two-normal-distributions-intersect
def gaussian_intersection(m1, m2, std1, std2):
    '''
    Parameters:
        m1, m2: Means of Gaussians
        std1, std2: Standard deviations of Gaussians

    Returns:
        Points of intersection of 2 Gaussian curves
    '''
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])


def get_non_overlapping_area_on_interval(interval_points, mus, sigmas,  max_time):
    point1 = interval_points[0]
    point2 = interval_points[1]

    # Pick a random point on the interval (maybe in the middle)
    if(point1 == -math.inf):
        mid_point = point2 - 0.001
    elif(point2 == math.inf):
        mid_point = point1 + 0.001
    else:
        mid_point = point1 + (point2 - point1)/2

    curve_list = []
    # Evaluate all functions at that point
    for i in range(len(mus)):
        mu = mus[i]
        sigma = sigmas[i]
        y = ss.norm.pdf(mid_point, mu, sigma)
        curve_list.append(y)

    # Find the area underneath the highest and subtract area underneath
    # second highest

    # Integrate highest over interval
    max_value = max(curve_list)
    max_index = curve_list.index(max_value)
    mu_h = mus[max_index]
    sigma_h = sigmas[max_index]

    # If the retention time of the highest peak is larger than 20, we dont want to count it
    # this is to encourage faster retention times.
    delta_t = max_time

    if(mu_h > delta_t):
        return(0)
    else:
        integral1 = ss.norm.cdf(point2, mu_h, sigma_h) - ss.norm.cdf(point1, mu_h, sigma_h)
        #print(integral1)

        # Integrate second highest over interval
        two_largest = heapq.nlargest(2, curve_list)
        second_max_value = min(two_largest)
        second_max_index = curve_list.index(second_max_value)
        mu_h2 = mus[second_max_index]
        sigma_h2 = sigmas[second_max_index]

        integral2 = ss.norm.cdf(point2, mu_h2, sigma_h2) - ss.norm.cdf(point1, mu_h2, sigma_h2)
        #print(integral2)

        # Return non-overlapping area between interval
        non_overlapping_area = integral1 - integral2
        # if non_overlapping_area > 0.1:
        #     print(mu_h, mu_h2, non_overlapping_area)
        #return((1/mu_h)*non_overlapping_area)
        return(non_overlapping_area)

def time_info(tR_list, W_list):
    return min(tR_list), max(tR_list)

def peak_purity(tR_list, W_list,  max_time=60):
    # Cast to array
    tR_list = np.array(tR_list)
    W_list = np.array(W_list)

    mus = tR_list
    sigmas = W_list/4

    number_of_curves = len(mus)

    # For every curve, find the intersections with all other curves
    # Yields a list of intersections from left to right
    intersections_list = []
    i_range = range(number_of_curves - 1)
    #for i in tqdm(i_range):
    for i in i_range:
        mu1 = mus[i]
        sigma1 = sigmas[i]

        if(mu1 < max_time):

            range1 = range(i + 1, number_of_curves)

            for j in range1:
                mu2 = mus[j]
                sigma2 = sigmas[j]
                intersections = gaussian_intersection(mu1, mu2, sigma1, sigma2)
                intersections_list.extend(intersections)

    intersections_list.sort()
    # Add -inf and inf
    intersections_list.insert(0, -math.inf)
    intersections_list.append(math.inf)

    total_non_overlapping_area = 0 # not sure if starting with 1 is justified

    # For each interval this yields, determine which curve is the highest
    # and which curve is the second highest in this interval
    for i in range(len(intersections_list) -1):
        interval = [intersections_list[i], intersections_list[i + 1]]
        area = get_non_overlapping_area_on_interval(interval, mus, sigmas, max_time)
        total_non_overlapping_area = total_non_overlapping_area + area
        #print('peak impurity ', total_non_overlapping_area)

    return(total_non_overlapping_area)

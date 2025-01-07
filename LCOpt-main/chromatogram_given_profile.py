import crf
import sys
import ast
import globals
import retention_model
import read_data as rd
import plot_chromatogram


t_0 = globals.t_0
t_D = globals.t_D
N = globals.N


def plot_chromatogram_given_gradient_profile(phi_list, t_init, t_list, sample_name):
    """
    Plot chromatogram for a specified gradient profile.
    Please specify the sample in read_data.py

    :phi_list: List of phi values; one for each turning point in the gradient profile.
    :t_init: t_init value (min) (length of initial isocratic segment)
    :t_list: List of t values; one for each turning point in the gradient profile.
    :sample_name: Name of the sample to use.
    """

    k0_list, S_list = rd.read_data(sample_name)
    #k0_list = [math.exp(lnk0) for lnk0 in lnk0_list]

    tR_list = []
    W_list = []

    for i in range(len(k0_list)):
        k_0 = k0_list[i]
        S = S_list[i]

        tR, W = retention_model.retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N)
        tR_list.append(tR)
        W_list.append(W)


    score = crf.crf(tR_list, W_list, phi_list)
    plot_chromatogram.plot_chromatogram(tR_list, W_list, phi_list, t_list, t_D, t_0, t_init, score, return_fig=False)


def main():
    if len(sys.argv) > 5:
        print('You have specified too many arguments.')
        sys.exit()

    if len(sys.argv) < 5:
        print('Please specify the following parameters in order:')
        print("- List of phi values; 1 for each turning point. Ex.: '[0.10, 0.10, 0.11, 0.275, 0.295, 0.30]'")
        print("- t_init value (min) (length of initial isocratic segment) Ex.: '5'")
        print("- List of t values; 1 for each turning point. Ex.: '[0, 11.7, 23.5, 36, 48, 60]'")
        print("- Name of the sample to use. Ex.: 'sample_real'")
        sys.exit()

    phi_list = ast.literal_eval(sys.argv[1])
    t_init = int(sys.argv[2])
    t_list = ast.literal_eval(sys.argv[3])
    sample_name = sys.argv[4]

    plot_chromatogram_given_gradient_profile(phi_list, t_init, t_list, sample_name)

if __name__ == '__main__':
    main()

# write me an example of how to run this script
# python chromatogram_given_profile.py '[0.10, 0.10, 0.11, 0.275, 0.295, 0.30]' 5 '[0, 11.7, 23.5, 36, 48, 60]' sample1

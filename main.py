"""
Peak deconvolution in Python.
Reference: http://kitchingroup.cheme.cmu.edu/blog/2013/01/29/Curve-fitting-to-get-overlapping-peak-areas/
Usage:
	python main.py input_file_path output_file_name N init_guess_amp_0 init_guess_center_0 init_guess_sigma_0 ... init_guess_amp_N-1 init_guess_center_N-1 init_guess_sigma_N-1
Options:
	input_file_path=<str>	-- 	Path to the data file (tab separated)
	N=<int>					-- 	Number of n_peaks
	init_guess=<float>		--  Initial guesses for center,sigma,amplitude
	output_file_name=<str>	--  Path of the file where the output will be generated

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
import pprint



def peak_deconv(argument_dictionary):


	datafile = argument_dictionary["input_file_path"]

	# read the input
	with open(datafile) as f:
	    lines = f.readlines()

	# now get the data
	t, intensity = [], []
	for line in lines:
	    data = line.split()
	    t.append(float(data[0]))
	    intensity.append(float(data[1]))

	t = np.array(t).astype(float)
	intensity = np.array(intensity).astype(float)


	plt.plot(t, intensity)
	plt.xlim([np.min(t), np.max(t)])
	plt.xlabel('Time (s)')
	plt.ylabel('Intensity (arb. units)')



	# TODO: Work on a better baseline
	intensity -= np.mean(intensity[(t > 30000) & (t < 40000)])
	plt.figure()
	plt.plot(t, intensity)
	plt.xlim([np.min(t), np.max(t)])
	plt.xlabel('Time (s)')
	plt.ylabel('Intensity (arb. units)')

	parguess = argument_dictionary["initial_guess"]
	plt.figure()
	plt.plot(t, intensity)
	plt.plot(t,n_peaks(t, *parguess),'g-')
	plt.xlim([np.min(t), np.max(t)])
	plt.xlabel('Time (s)')
	plt.ylabel('Intensity (arb. units)')


	# actual fitting
	popt, pcov = curve_fit(n_peaks, t, intensity, parguess)

	plt.xlim([np.min(t), np.max(t)])
	plt.plot(t, intensity)
	plt.plot(t,n_peaks(t, *parguess),'g-')
	plt.plot(t, n_peaks(t, *popt), 'r-')
	plt.legend(['data', 'initial guess','final fit'])

	plt.savefig(argument_dictionary["output_file_name"])


def asym_peak(t, amp, sigma, centr):
	#     amp = pars[0]  # amplitude
	#     sigma = pars[1]  # variance
	#     centr = pars[2]  # expected value

	f=amp*(1/(sigma*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((t-centr)/sigma)**2)))
	return f


def n_peaks(t,*pars):
	p_total = 0
	peaknum = len(pars) // 3
	for n in range(peaknum):
		tmp = asym_peak(t, amp=pars[3*n], sigma=pars[3*n+1], centr=pars[3*n+2])

		p_total = p_total + tmp
	return p_total


def parse_input(args):

	argument_dictionary = dict()

	argument_dictionary["input_file_path"] = args[0]
	argument_dictionary["output_file_name"] = args[1]
	argument_dictionary["N"] = int(args[2])
	argument_dictionary["initial_guess"] = []
	for num in args[3:]:
		argument_dictionary["initial_guess"].append(float(num))

	argument_dictionary["initial_guess"] = tuple(argument_dictionary["initial_guess"])

	pprint.pprint(argument_dictionary)
	return argument_dictionary




if __name__ == '__main__':
	import sys
	if len(sys.argv) != 1 + 3 + int(sys.argv[3]) * 3:  # wrong number of arguments
		print("Wrong number of arguments!, expected = ", 3 + int(sys.argv[3]) * 3)
		print("Usage: python main.py input_file_path N init_guess_amp init_guess_center init_guess_sigma output_file_name")
		sys.exit(1)

	argument_dictionary = parse_input(sys.argv[1:])
	peak_deconv(argument_dictionary)

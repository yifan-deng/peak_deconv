# Peak deconvolution


### Usage

```py
python main.py input_file_path output_file_name N init_guess_amp_0 init_guess_center_0 init_guess_sigma_0 ... init_guess_amp_N-1 init_guess_center_N-1 init_guess_sigma_N-1
```

where
- input_file_path=<str>	-- 	Path to the data file (described below)
- N=<int>					-- 	Number of n_peaks
- init_guess=<float>		--  Initial guesses for center,sigma,amplitude (described below)
- output_file_name=<str>	--  Path of the file where the output will be generated

### File Format
This script takes a file which has the data that is separated by space. The first column is for x-axis and 2nd column for y axis.

### Initial Guess
The script also takes the number of peaks and the inital guesses. The initial guesses include amptitude, center, and sigma in this order. The output is a file that plot the data, initial guess, and final fit together.

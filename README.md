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

A sample exact command is  
```py
python main.py data/UV_Vis_TwoPeak_Raw.txt data/output.png 2 200 2500 22000 80 1000 24000"
```

### File Format
This script takes a file which has the data that is separated by space. The first column is for x-axis and 2nd column for y axis.
An example of the data file looks like:
| # | Wavelength (nm) | Intensity (a.u.) |
|:-:|:---------------:|:----------------:|
| 1 |   15012.08473   |     4.02E-04     |
| 2 |   15187.38938   |     4.69E-04     |
| 3 |   15362.69402   |     5.53E-04     |

An example of the data file is located at data/UV_Vis_TwoPeak.txt

### Initial Guess
The script also takes the number of peaks and the inital guesses. The initial guesses include amptitude, center, and sigma in this order. The output is a file that plot the data, initial guess, and final fit together.

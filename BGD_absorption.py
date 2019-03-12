import BGD
import numpy as np
import pickle
import sys

def FindRMS(spectrum):
	x = len(spectrum)
	a = int(x / 10)
	rms_list = []
	for _set in range(9):
		rms = np.std(spectrum[(_set * a):(_set * a) + (2 * a)])
		rms_list.append(rms)
	median_rms = np.nanmedian(rms_list)
	return median_rms
def TrimSpectrum(vel_axis, spectrum, min_v, max_v):
	min_vel_index = np.argmin(abs(vel_axis - min_v))
	max_vel_index = np.argmin(abs(vel_axis - max_v))
	trimmed_vel_axis = vel_axis[min_vel_index:max_vel_index]
	trimmed_spectrum = spectrum[min_vel_index:max_vel_index]
	return (trimmed_vel_axis, trimmed_spectrum)
def BinSpectrum(vel_axis = None, spectrum = None, bin_channels = None):
	binned_vel = [(vel_axis[x] + vel_axis[x + bin_channels - 1])/2. for x in range(0, len(vel_axis) - bin_channels, bin_channels)]
	binned_spectrum = [sum(spectrum[x:x + bin_channels])/bin_channels for x in range(0, len(vel_axis) - bin_channels, bin_channels)]
	return (binned_vel, binned_spectrum)


source_list = ['g003', 'g006', 'g007', 'g334', 'g336', 'g340a', 'g340b', 'g344', 'g346', 'g347', 'g348', 'g350', 'g351a', 'g351b', 'g353', 'g356'] 
freq_list = ['1612', '1665', '1667', '1720']
source_name_dict = {	'g003': 'g003.74+0.64.', 
						'g006': 'g006.32+1.97.', 
						'g007': 'g007.47+0.06.', 
						'g334': 'g334.72-0.65.', 
						'g336': 'g336.49-1.48.', 
						'g340a': 'g340.79-1.02a.', 
						'g340b': 'g340.79-1.02b.', 
						'g344': 'g344.43+0.05.', 
						'g346': 'g346.52+0.08.', 
						'g347': 'g347.75-1.14.', 
						'g348': 'g348.44+2.08.', 
						'g349': 'g349.73+1.67.', 
						'g350': 'g350.50+0.96.', 
						'g351a': 'g351.61+0.17a.', 
						'g351b': 'g351.61+0.17b.', 
						'g353': 'g353.411-0.3.', 
						'g356': 'g356.91+0.08.'}

min_vel = -175.
max_vel = 200.

if len(sys.argv) != 2:
	print('Error! You haven\'t provided enough arguments!')
else:
	source = sys.argv[1]

	source_name = source_name_dict[source]
	# Initiate dictionaries
	vel_axes = {'1612': None, '1665': None, '1667': None, '1720': None}
	tau_spectra = {'1612': None, '1665': None, '1667': None, '1720': None}
	tau_rms_values = {'1612': None, '1665': None, '1667': None, '1720': None}
	Tc_values = {'1612': None, '1665': None, '1667': None, '1720': None}
		
	

	with open('pickles/' + source + '_1612.pickle', 'rb') as f:
		source_dict_1612 = pickle.load(f, encoding='bytes')
	with open('pickles/' + source + '_1665.pickle', 'rb') as f:
		source_dict_1665 = pickle.load(f, encoding='bytes')
	with open('pickles/' + source + '_1667.pickle', 'rb') as f:
		source_dict_1667 = pickle.load(f, encoding='bytes')
	with open('pickles/' + source + '_1720.pickle', 'rb') as f:
		source_dict_1720 = pickle.load(f, encoding='bytes')

	for _bin in reversed([2, 3, 4, 5, 6, 7, 8]):
		'''
		Contents of dict:
			'tau', 'Tc_rms', 'tau_rms', 'Tb_cont_sub', 'vel_axis', 'Smooth_sub_Tb', 'Tb', 'Tc'
		'''
		vel_axis_1612 = np.array(source_dict_1612[b'vel_axis'])
		vel_axis_1665 = np.array(source_dict_1665[b'vel_axis'])
		vel_axis_1667 = np.array(source_dict_1667[b'vel_axis'])
		vel_axis_1720 = np.array(source_dict_1720[b'vel_axis'])
		spectrum_1612 = np.array(source_dict_1612[b'Tb_cont_sub'])
		spectrum_1665 = np.array(source_dict_1665[b'Tb_cont_sub'])
		spectrum_1667 = np.array(source_dict_1667[b'Tb_cont_sub'])
		spectrum_1720 = np.array(source_dict_1720[b'Tb_cont_sub'])
		# Trim
		(trimmed_vel_axis_1612, trimmed_spectrum_1612) = TrimSpectrum(vel_axis_1612, spectrum_1612, min_vel, max_vel)
		(trimmed_vel_axis_1665, trimmed_spectrum_1665) = TrimSpectrum(vel_axis_1665, spectrum_1665, min_vel, max_vel)
		(trimmed_vel_axis_1667, trimmed_spectrum_1667) = TrimSpectrum(vel_axis_1667, spectrum_1667, min_vel, max_vel)
		(trimmed_vel_axis_1720, trimmed_spectrum_1720) = TrimSpectrum(vel_axis_1720, spectrum_1720, min_vel, max_vel)

		# Bin
		(binned_vel_axis_1612, binned_spectrum_1612) = BinSpectrum(trimmed_vel_axis_1612, trimmed_spectrum_1612, _bin)
		(binned_vel_axis_1665, binned_spectrum_1665) = BinSpectrum(trimmed_vel_axis_1665, trimmed_spectrum_1665, _bin)
		(binned_vel_axis_1667, binned_spectrum_1667) = BinSpectrum(trimmed_vel_axis_1667, trimmed_spectrum_1667, _bin)
		(binned_vel_axis_1720, binned_spectrum_1720) = BinSpectrum(trimmed_vel_axis_1720, trimmed_spectrum_1720, _bin)
		# Calculate tau
		tau_1612 = -np.log(1 + binned_spectrum_1612/source_dict_1612[b'Tc'])
		tau_1665 = -np.log(1 + binned_spectrum_1665/source_dict_1665[b'Tc'])
		tau_1667 = -np.log(1 + binned_spectrum_1667/source_dict_1667[b'Tc'])
		tau_1720 = -np.log(1 + binned_spectrum_1667/source_dict_1720[b'Tc'])
		tau_rms_1612 = FindRMS(tau_1612)
		tau_rms_1665 = FindRMS(tau_1665)
		tau_rms_1667 = FindRMS(tau_1667)
		tau_rms_1720 = FindRMS(tau_1720)
		
		# Load into dictionaries
		Tc_values['1612'] = source_dict_1612[b'Tc']
		Tc_values['1665'] = source_dict_1665[b'Tc']
		Tc_values['1667'] = source_dict_1667[b'Tc']
		Tc_values['1720'] = source_dict_1720[b'Tc']
		tau_spectra['1612'] = tau_1612
		tau_spectra['1665'] = tau_1665
		tau_spectra['1667'] = tau_1667
		tau_spectra['1720'] = tau_1720
		tau_rms_values['1612'] = tau_rms_1612
		tau_rms_values['1665'] = tau_rms_1665
		tau_rms_values['1667'] = tau_rms_1667
		tau_rms_values['1720'] = tau_rms_1720
		vel_axes['1612'] = binned_vel_axis_1612
		vel_axes['1665'] = binned_vel_axis_1665
		vel_axes['1667'] = binned_vel_axis_1667
		vel_axes['1720'] = binned_vel_axis_1720


		for tau_tol in [2, 3, 4, 5, 6, 7]:
			for vel_convergence in [1.5, 2., 2.5, 3., 3.5]:
				for rms_mult in [1, 0.5, 0.75, 1.25, 1.5, 1.75, 2]:
					print('\n\n\n\n\ntau_tol = \t' + str(tau_tol) + '\tvel_convergence = \t' + str(vel_convergence) + '\trms_mult = \t' + str(rms_mult) + '\tbin = \t' + str(_bin) + '\n\n')

					tau_rms_values['1612'] = rms_mult * tau_rms_values['1612']
					tau_rms_values['1665'] = rms_mult * tau_rms_values['1665']
					tau_rms_values['1667'] = rms_mult * tau_rms_values['1667']
					tau_rms_values['1720'] = rms_mult * tau_rms_values['1720']

					BGD.Main(
							source_name = source_name, 
							vel_axes = [vel_axes['1612'], vel_axes['1665'], vel_axes['1667'], vel_axes['1720']], 
							tau_spectra = [tau_spectra['1612'], tau_spectra['1665'], tau_spectra['1667'], tau_spectra['1720']], 
							tau_rms = [tau_rms_values['1612'], tau_rms_values['1665'], tau_rms_values['1667'], tau_rms_values['1720']], 
							Tbg = [Tc_values['1612'], Tc_values['1665'], Tc_values['1667'], Tc_values['1720']], 
							quiet = True, 
							_bin = _bin, 
							vel_convergence = vel_convergence,
							tau_tol = tau_tol, 
							rms_mult = rms_mult)

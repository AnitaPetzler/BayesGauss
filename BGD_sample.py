import BGD
import pickle

'''
Sample code showing implementation of the Bayesian Gaussian Decomposition algorithm BGD 
(https://github.com/AnitaPetzler/BayesGauss/)
'''

def FindVelIndex(min_vel, max_vel, vel_axis):
	'''
	Finds the min and max indices corresponding to 
	the min and max velocities given.
	'''
	dv = vel_axis[1] - vel_axis[0]
	v_at_0 = vel_axis[0]

	min_v_index = int(min((min_vel - v_at_0) / dv, (max_vel - v_at_0) / dv))
	max_v_index = int(max((min_vel - v_at_0) / dv, (max_vel - v_at_0) / dv))

	if min_v_index < 0:
		min_v_index = 0

	return (min_v_index, max_v_index)

vel_range = {'4C+25.14': (-85, 75), 'ch002': (-52, 10)}

for source_name in ['4C+25.14', 'ch002']:
	for em in ['absorption', 'emission']:
		
		save_as_name = source_name + '_' + em

		# Loading main line data
		with open('pickles/' + source_name + '_1665_' + em + '.pickle', 'r') as f:
			spectrum_1665 = pickle.load(f)
		with open('pickles/' + source_name + '_1665_vel.pickle', 'r') as f:
			vel_axis_1665 = pickle.load(f)
		with open('pickles/rms_' + source_name + '_1665_' + em + '.pickle', 'r') as f:
			rms_1665 = pickle.load(f)
		with open('pickles/' + source_name + '_1667_' + em + '.pickle', 'r') as f:
			spectrum_1667 = pickle.load(f)
		with open('pickles/' + source_name + '_1667_vel.pickle', 'r') as f:
			vel_axis_1667 = pickle.load(f)
		with open('pickles/rms_' + source_name + '_1667_' + em + '.pickle', 'r') as f:
			rms_1667 = pickle.load(f)

		# Loading satellite lines. Current version of BGD requires both satellite lines, so if only 
		# one is present just duplicate it for the other one.
		try:
			with open('pickles/' + source_name + '_1612_' + em + '.pickle', 'r') as f:
				spectrum_1612 = pickle.load(f)
			with open('pickles/' + source_name + '_1612_vel.pickle', 'r') as f:
				vel_axis_1612 = pickle.load(f)
			with open('pickles/rms_' + source_name + '_1612_' + em + '.pickle', 'r') as f:
				rms_1612 = pickle.load(f)
		except IOError:
			with open('pickles/' + source_name + '_1720_' + em + '.pickle', 'r') as f:
				spectrum_1612 = pickle.load(f)
			with open('pickles/' + source_name + '_1720_vel.pickle', 'r') as f:
				vel_axis_1612 = pickle.load(f)
			with open('pickles/rms_' + source_name + '_1720_' + em + '.pickle', 'r') as f:
				rms_1612 = pickle.load(f)
			print '1612 replaced by 1720 for ' + save_as_name

		try:
			with open('pickles/' + source_name + '_1720_' + em + '.pickle', 'r') as f:
				spectrum_1720 = pickle.load(f)
			with open('pickles/' + source_name + '_1720_vel.pickle', 'r') as f:
				vel_axis_1720 = pickle.load(f)
			with open('pickles/rms_' + source_name + '_1720_' + em + '.pickle', 'r') as f:
				rms_1720 = pickle.load(f)
		except IOError:
			with open('pickles/' + source_name + '_1612_' + em + '.pickle', 'r') as f:
				spectrum_1720 = pickle.load(f)
			with open('pickles/' + source_name + '_1612_vel.pickle', 'r') as f:
				vel_axis_1720 = pickle.load(f)
			with open('pickles/rms_' + source_name + '_1612_' + em + '.pickle', 'r') as f:
				rms_1720 = pickle.load(f)
			print '1720 replaced by 1612 for ' + save_as_name

		# Trim spectra so all 4 cover the same velocity range. 
		(min_index_1612, max_index_1612) = FindVelIndex(vel_range[source_name][0], vel_range[source_name][1], vel_axis_1612)
		(min_index_1665, max_index_1665) = FindVelIndex(vel_range[source_name][0], vel_range[source_name][1], vel_axis_1665)
		(min_index_1667, max_index_1667) = FindVelIndex(vel_range[source_name][0], vel_range[source_name][1], vel_axis_1667)
		(min_index_1720, max_index_1720) = FindVelIndex(vel_range[source_name][0], vel_range[source_name][1], vel_axis_1720)

		vel_axes = [vel_axis_1612[min_index_1612:max_index_1612], vel_axis_1665[min_index_1665:max_index_1665], vel_axis_1667[min_index_1667:max_index_1667], vel_axis_1720[min_index_1720:max_index_1720]]
		spectra = [spectrum_1612[min_index_1612:max_index_1612], spectrum_1665[min_index_1665:max_index_1665], spectrum_1667[min_index_1667:max_index_1667], spectrum_1720[min_index_1720:max_index_1720]]
		rms = [rms_1612, rms_1665, rms_1667, rms_1720]

		expected_min_fwhm = 1.
		# Run BGD
		final_parameters = BGD.Main(source_name, vel_axes, spectra, rms, expected_min_fwhm,save_as_name)
		# Print results to terminal
		BGD.ResultsReport(final_parameters, save_as_name)



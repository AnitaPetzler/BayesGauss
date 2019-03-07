from emcee.utils import MPIPool
from mpfit import mpfit
from statistics import mean
import copy
import corner
import datetime
import emcee
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import statistics
import sys

'''
To Do List:

- Move max acceptable fwhm to Main
- Re-implement convergence test
- test if chains have more than one mode - split somehow?


################################################################
#                                                              #
# Special version of BGD for Arecibo on-off observations of OH #
#           Fit exp(-tau) x 4 and Texp x 4 using mpfit         #
#            Sample posterior probability using mcmc           #
#                                                              #
################################################################

Note to user:

Increase the maximum number of open files:
In terminal (Mac):

$ ulimit -a

	...
	open files                      (-n) 256
	...

$ ulimit -n 10000
$ ulimit -a

	...
	open files                      (-n) 10000
	...

'''
def FindGaussians(data, dv, tau_tol = None):

	# print('Beginning Initial Analysis ' + str(datetime.datetime.now()))
	(parameters, vel_llh_list, llh_list) = InitialAnalysis(data = data, dv = dv, tau_tol = tau_tol)
	# print('\tIdentifying significant regions ' + str(datetime.datetime.now()))
	(baseline_llh, baseline_rms) = FindBaseline(vel_llh_list, llh_list)

	vel_sig = [vel_llh_list[x] for x in range(len(vel_llh_list)) if llh_list[x] > baseline_llh + 2 * baseline_rms]
	reduced_vel_sig = ReduceList(vel_sig, 0.5, 1.)

	sig_vel_ranges = []
	for group in reduced_vel_sig:
		group_range = [group[0], group[-1]]
		sig_vel_ranges.append(group_range)
	# print('\t\tSignificant velocity ranges identified: ' + str(sig_vel_ranges) + ' ' + str(datetime.datetime.now()))
	# print('\t\tParameters: ' + str(parameters))
	return sig_vel_ranges
def InitialAnalysis(data, dv, tau_tol = None):
	'''
	identifies ranges that might have features.
	'''
	def mpfitInitial(p, fjac, data):
		if data['Texp_spectrum']['1665'] == []:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p, data)
			residuals = np.concatenate((
						(tau_model_1612 - data['tau_spectrum']['1612']) / data['tau_rms']['1612'], 
						(tau_model_1665 - data['tau_spectrum']['1665']) / data['tau_rms']['1665'], 
						(tau_model_1667 - data['tau_spectrum']['1667']) / data['tau_rms']['1667'], 
						(tau_model_1720 - data['tau_spectrum']['1720']) / data['tau_rms']['1720']))

		else:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = ModelInitial(p, data)
			residuals = np.concatenate((
						(tau_model_1612 - data['tau_spectrum']['1612']) / data['tau_rms']['1612'], 
						(tau_model_1665 - data['tau_spectrum']['1665']) / data['tau_rms']['1665'], 
						(tau_model_1667 - data['tau_spectrum']['1667']) / data['tau_rms']['1667'], 
						(tau_model_1720 - data['tau_spectrum']['1720']) / data['tau_rms']['1720'], 
						(Texp_model_1612 - data['Texp_spectrum']['1612']) / data['Texp_rms']['1612'], 
						(Texp_model_1665 - data['Texp_spectrum']['1665']) / data['Texp_rms']['1665'], 
						(Texp_model_1667 - data['Texp_spectrum']['1667']) / data['Texp_rms']['1667'], 
						(Texp_model_1720 - data['Texp_spectrum']['1720']) / data['Texp_rms']['1720'])) 

		return [0, residuals]
	def ModelInitial(p, data):
		[v, fwhm, tau_1612, tau_1665, tau_1667, tau_1720, Tex_1612, Tex_1665, Tex_1667] = p
		Tex_1720 = 1.72053 / ((1.665402 / Tex_1665) + (1.667359 / Tex_1667) - (1.612231 / Tex_1612))
		new_p = [v, fwhm, tau_1612, tau_1665, tau_1667, tau_1720, Tex_1612, Tex_1665, Tex_1667, Tex_1720]
		return MakeModel(new_p, data)
	def dBIC(p, data):
		if data['Texp_spectrum']['1665'] == []:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p, data)
			n = len(data['tau_spectrum']['1612']) + len(data['tau_spectrum']['1665']) + len(data['tau_spectrum']['1667']) + len(data['tau_spectrum']['1720'])
			sse_model = sum((tau_model_1612 - data['tau_spectrum']['1612'])**2.) + sum((tau_model_1665 - data['tau_spectrum']['1665'])**2.) + sum((tau_model_1667 - data['tau_spectrum']['1667'])**2.) + sum((tau_model_1720 - data['tau_spectrum']['1720'])**2.)
			sse_null = sum((data['tau_spectrum']['1612'])**2.) + sum(data['tau_spectrum']['1665']**2.) + sum(data['tau_spectrum']['1667']**2.) + sum(data['tau_spectrum']['1720']**2.)
			return (n * np.log(sse_model / n) + 6. * np.log(n)) - (n * np.log(sse_null / n) + np.log(n))
		else:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = ModelInitial(p, data)
			n = len(data['tau_spectrum']['1612']) + len(data['tau_spectrum']['1665']) + len(data['tau_spectrum']['1667']) + len(data['tau_spectrum']['1720']) + len(data['Texp_spectrum']['1612']) + len(data['Texp_spectrum']['1665']) + len(data['Texp_spectrum']['1667']) + len(data['Texp_spectrum']['1720'])
			sse_model = sum((tau_model_1612 - data['tau_spectrum']['1612'])**2.) + sum((tau_model_1665 - data['tau_spectrum']['1665'])**2.) + sum((tau_model_1667 - data['tau_spectrum']['1667'])**2.) + sum((tau_model_1720 - data['tau_spectrum']['1720'])**2.) + sum((Texp_model_1612 - data['Texp_spectrum']['1612'])**2.) + sum((Texp_model_1665 - data['Texp_spectrum']['1665'])**2.) + sum((Texp_model_1667 - data['Texp_spectrum']['1667'])**2.) + sum((Texp_model_1720 - data['Texp_spectrum']['1720'])**2.)
			sse_null = sum((data['tau_spectrum']['1612'])**2.) + sum(data['tau_spectrum']['1665']**2.) + sum(data['tau_spectrum']['1667']**2.) + sum(data['tau_spectrum']['1720']**2.) + sum(data['Texp_spectrum']['1612']**2.) + sum(data['Texp_spectrum']['1665']**2.) + sum(data['Texp_spectrum']['1667']**2.) + sum(data['Texp_spectrum']['1720']**2.)
			return (n * np.log(sse_model / n) + 10. * np.log(n)) - (n * np.log(sse_null / n) + np.log(n))

	min_vel = max([min(data['vel_axis']['1612']), min(data['vel_axis']['1665']), min(data['vel_axis']['1667']), min(data['vel_axis']['1720'])])
	max_vel = min([max(data['vel_axis']['1612']), max(data['vel_axis']['1665']), max(data['vel_axis']['1667']), max(data['vel_axis']['1720'])])
	
	if data['Texp_spectrum']['1665'] == []:
		parinfo = [	{'parname':'velocity','fixed':1}, 
			{'parname':'fwhm','step':1.E-5, 'limited': [1, 1], 'limits': [0.01, 5.]}, 
			{'parname':'tau_1612_height','step':1.E-5}, 
			{'parname':'tau_1665_height','step':1.E-5}, 
			{'parname':'tau_1667_height','step':1.E-5}, 
			{'parname':'tau_1720_height','step':1.E-5}]
	else:
		parinfo = [	{'parname':'velocity','fixed':1}, 
			{'parname':'fwhm','step':1.E-5, 'limited': [1, 1], 'limits': [0.01, 5.]}, 
			{'parname':'tau_1612_height','step':1.E-5}, 
			{'parname':'tau_1665_height','step':1.E-5}, 
			{'parname':'tau_1667_height','step':1.E-5}, 
			{'parname':'tau_1720_height','step':1.E-5}, 
			{'parname':'Tex_1612_height','step':1.E-2, 'limited': [1, 1], 'limits': [-1000., 1000.]}, 
			{'parname':'Tex_1665_height','step':1.E-2, 'limited': [1, 1], 'limits': [-1000., 1000.]}, 
			{'parname':'Tex_1667_height','step':1.E-2, 'limited': [1, 1], 'limits': [-1000., 1000.]}]

	global parameters_global
	global llh_list_global
	global vel_llh_list_global

	parameters_global = []
	llh_list_global = []
	vel_llh_list_global = []

	while min_vel < max_vel - 30.:
		trimmed_data = TrimData(data, min_vel, min_vel + 30.)
		fa = {'data': trimmed_data}
		for vel in np.arange(min_vel + 10., min_vel + 20., 0.5):
			# v, fwhm, tau x4, Tex x3
			if data['Texp_spectrum']['1665'] == []:
				guess = [vel, 1., 0.001, 0.001, 0.001, 0.001]
			else:
				guess = [vel, 1., 0.001, 0.001, 0.001, 0.001, 10., 10., 10.]

			mp = mpfit(mpfitInitial, guess, parinfo = parinfo, functkw = fa, maxiter = 1000, quiet = True)
			parameters = mp.params
			
			if data['Texp_spectrum']['1665'] != []:
				parameters = (np.append(mp.params, [1.72053/((1.665402/mp.params[7]) + (1.667359/mp.params[8]) - (1.612231/mp.params[6]))])).tolist()

			parameters_global.append(parameters)

			llh_variable = lnprob(parameters, data, min_vel, max_vel, dv = dv, tau_tol = tau_tol)
			llh_list_global.append(llh_variable)
			vel_llh_list_global.append(vel)
		min_vel += 10

	vel_llh_list_global_new = [vel_llh_list_global[x] for x in range(len(vel_llh_list_global)) if np.isinf(llh_list_global[x]) == False]
	llh_list_global_new = [llh_list_global[x] for x in range(len(llh_list_global)) if np.isinf(llh_list_global[x]) == False]
	return (parameters_global, vel_llh_list_global_new, llh_list_global_new)
def TrimData(data, min_vel, max_vel):
	data_temp = copy.deepcopy(data)

	vel_1612 = np.array(data_temp['vel_axis']['1612'])
	vel_1665 = np.array(data_temp['vel_axis']['1665'])
	vel_1667 = np.array(data_temp['vel_axis']['1667'])
	vel_1720 = np.array(data_temp['vel_axis']['1720'])

	tau_1612 = np.array(data_temp['tau_spectrum']['1612'])
	tau_1665 = np.array(data_temp['tau_spectrum']['1665'])
	tau_1667 = np.array(data_temp['tau_spectrum']['1667'])
	tau_1720 = np.array(data_temp['tau_spectrum']['1720'])	
	
	if data['Texp_spectrum']['1665'] != []:
		Texp_1612 = np.array(data_temp['Texp_spectrum']['1612'])
		Texp_1665 = np.array(data_temp['Texp_spectrum']['1665'])
		Texp_1667 = np.array(data_temp['Texp_spectrum']['1667'])
		Texp_1720 = np.array(data_temp['Texp_spectrum']['1720'])

	min_vel -= 10.
	max_vel += 10.

	min_index_1612 = min([np.argmin(abs(vel_1612 - min_vel)), np.argmin(abs(vel_1612 - max_vel))])
	max_index_1612 = max([np.argmin(abs(vel_1612 - min_vel)), np.argmin(abs(vel_1612 - max_vel))])

	min_index_1665 = min([np.argmin(abs(vel_1665 - min_vel)), np.argmin(abs(vel_1665 - max_vel))])
	max_index_1665 = max([np.argmin(abs(vel_1665 - min_vel)), np.argmin(abs(vel_1665 - max_vel))])	

	min_index_1667 = min([np.argmin(abs(vel_1667 - min_vel)), np.argmin(abs(vel_1667 - max_vel))])
	max_index_1667 = max([np.argmin(abs(vel_1667 - min_vel)), np.argmin(abs(vel_1667 - max_vel))])

	min_index_1720 = min([np.argmin(abs(vel_1720 - min_vel)), np.argmin(abs(vel_1720 - max_vel))])
	max_index_1720 = max([np.argmin(abs(vel_1720 - min_vel)), np.argmin(abs(vel_1720 - max_vel))])

	data_temp['vel_axis']['1612'] = vel_1612[min_index_1612:max_index_1612 + 1]
	data_temp['vel_axis']['1665'] = vel_1665[min_index_1665:max_index_1665 + 1]
	data_temp['vel_axis']['1667'] = vel_1667[min_index_1667:max_index_1667 + 1]
	data_temp['vel_axis']['1720'] = vel_1720[min_index_1720:max_index_1720 + 1]

	data_temp['tau_spectrum']['1612'] = tau_1612[min_index_1612:max_index_1612 + 1]
	data_temp['tau_spectrum']['1665'] = tau_1665[min_index_1665:max_index_1665 + 1]
	data_temp['tau_spectrum']['1667'] = tau_1667[min_index_1667:max_index_1667 + 1]
	data_temp['tau_spectrum']['1720'] = tau_1720[min_index_1720:max_index_1720 + 1]

	if data['Texp_spectrum']['1665'] != []:
		data_temp['Texp_spectrum']['1612']=Texp_1612[min_index_1612:max_index_1612+1]
		data_temp['Texp_spectrum']['1665']=Texp_1665[min_index_1665:max_index_1665+1]
		data_temp['Texp_spectrum']['1667']=Texp_1667[min_index_1667:max_index_1667+1]
		data_temp['Texp_spectrum']['1720']=Texp_1720[min_index_1720:max_index_1720+1]

	return data_temp
def FindBaseline(x_axis, y_axis):
	def mpfitBaseline(p, fjac, x_axis_subset, y_axis_subset):
		[m, c] = p
		model_y = [m * x + c for x in x_axis_subset]
		residuals = np.array([y_axis_subset[x] - model_y[x] for x in range(len(x_axis_subset)) if np.isnan(y_axis_subset[x]) == False and np.isinf(y_axis_subset[x]) == False])

		return [0, residuals]
	
	guess = [1., 1.]
	parinfo = [
	{'parname':'m','step':1.,'limited':[1,1],'limits':[-1.E10,1.E10]},
	{'parname':'c','step':1.,'limited':[1,1],'limits':[-1.E10,1.E10]}]

	m_list = []
	c_list = []
	x_list = []

	if len(x_axis) < 10 and len(x_axis) > 3:
		fa = {'x_axis_subset': x_axis, 'y_axis_subset': y_axis}
		mp = mpfit(mpfitBaseline, guess, parinfo = parinfo, functkw = fa, maxiter = 1000, quiet = True)
		model_y = [mp.params[0] * x + mp.params[1] for x in x_axis]
		
		return(statistics.median(model_y), FindRMS(y_axis))
	elif len(x_axis) <= 3:
		print('error finding baseline: x_axis entered is too short')
	else:
		for start_ind in range(len(x_axis) - 5):
			x_axis_subset = x_axis[start_ind:start_ind + 5]
			y_axis_subset = y_axis[start_ind:start_ind + 5]
			fa = {'x_axis_subset': x_axis_subset, 'y_axis_subset': y_axis_subset}
			mp = mpfit(mpfitBaseline, guess, parinfo = parinfo, functkw = fa, maxiter = 1000, quiet = True)
			m_list.append(mp.params[0])
			c_list.append(mp.params[1])
			x_list.append(x_axis[start_ind + 2])
			model_y = [mp.params[0] * x + mp.params[1] for x in x_axis_subset]
		
		m_rms = FindRMS(m_list)
		y_list = [m_list[x] * x_list[x] + c_list[x] for x in range(len(m_list)) if abs(m_list[x]) < 2 * m_rms]
		
		if len(y_list) == 0:
			y_list = [m_list[x] * x_list[x] + c_list[x] for x in range(len(m_list))]
		
		return (statistics.median(y_list), FindRMS(y_axis))
def FindRMS(spectrum):
	x = len(spectrum)
	a = int(x / 10)
	rms_list = []
	for _set in range(9):
		rms = np.std(spectrum[(_set * a):(_set * a) + (2 * a)])
		rms_list.append(rms)
	median_rms = np.median(rms_list)
	return median_rms
def ReduceList(master_list, merge_size, group_spacing):	
	'''
	Merges values in master_list separated by less than merge_size, and groups features separated by 
	less than group_spacing into blended features.
	Returns a list of lists: i.e. [[a], [b, c, d], [e]] where 'a' and 'e' are isolated
	features and 'b', 'c' and 'd' are close enough to overlap in velocity.

	Parameters:
	master_list - list of velocities
	merge_size - any velocities separated by less than this distance will be merged into one velocity. This 
			is performed in 4 stages, first using merge_size / 4 so that the closest velocities are merged 
			first. Merged velocities are replaced by their mean. 
	group_spacing - any velocities separated by less than this value will be grouped together so they can 
			be fit as blended features. Smaller values are likely to prevent the accurate identification 
			of blended features, while larger values will increase running time.

	Returns nested list of velocities:
		reduced_vel_list = [[v1], [v2, v3], [v4]]
		
		where v1 and v4 are isolated features that can be fit independently, but v2 and v3 are close 
		enough in velocity that they must be fit together as a blended feature.
	'''
	try:
		master_list = sorted([val for sublist in master_list for val in sublist])
	except TypeError:
		pass

	master_list = np.array(master_list)
	
	# Step 1: merge based on merge_size
	new_merge_list = np.sort(master_list.flatten())

	for merge in [merge_size / 4, 2 * merge_size / 4, 3 * merge_size / 4, merge_size]:
		new_merge_list = MergeFeatures(new_merge_list, merge, 'merge')
	
	# Step 2: identify components likely to overlap to be fit together
	final_merge_list = MergeFeatures(new_merge_list, group_spacing, 'group')

	return final_merge_list
def MergeFeatures(master_list, size, action):
	'''
	Does the work for ReduceList

	Parameters:
	master_list - list of velocities generated by AGD()
	size - Distance in km/sec for the given action
	action - Action to perform: 'merge' or 'group'

	Returns nested list of velocities:
		reduced_vel_list = [[v1], [v2, v3], [v4]]
		
		where v1 and v4 are isolated features that can be fit independently, but v2 and v3 are close 
		enough in velocity that they must be fit together as a blended feature.
	'''	
	new_merge_list = []
	check = 0
	while check < len(master_list):
		skip = 1
		single = True

		if action == 'merge':
			while check + skip < len(master_list) and master_list[check + skip] - master_list[check] < size:
				skip += 1
				single = False
			if single == True:
				new_merge_list = np.append(new_merge_list, master_list[check])
			else:
				new_merge_list = np.append(new_merge_list, mean(master_list[check:check + skip]))
			check += skip

		elif action == 'group':
			while check + skip < len(master_list) and master_list[check + skip] - master_list[check + skip - 1] < size:
				skip += 1
			new_merge_list.append(master_list[check:check + skip].tolist())
			check += skip
		else:
			print('Error defining action in MergeFeatures')

	return new_merge_list

#################
# Fit Gaussians #
#################

def FitGaussians(data = None, sig_ranges = None, plot_num = None, quiet = True, Bayes_threshold = 10, dv = None, con_test_limit = None, tau_tol = 5., max_cores = None):
	plot_num = 0
	source_name = data['source_name']
	null_evidence = nullEvidence(data)
	accepted_med_parameters = []
	accepted_parameters = []

	for group in sig_ranges:

		prev_evidence = null_evidence
		num_gauss_to_fit = 1.
		min_vel = min(group)
		max_vel = max(group)
		modified_data = TrimData(data, min_vel, max_vel)
		
		keep_going = True
		last_accepted_med_parameters = []
		last_accepted_parameters = []

		while keep_going == True:
			# print('\n\n***********************\n\ncurrently fitting ' + str(num_gauss_to_fit) + ' Gaussian(s)  ' + str(datetime.datetime.now()))
			sample_posterior = SamplePosterior(data = modified_data, num_gauss = num_gauss_to_fit, quiet = quiet, plot_num = plot_num, min_vel = min_vel, max_vel = max_vel, dv = dv, prev_params = last_accepted_med_parameters, accepted_parameters = accepted_med_parameters, con_test_limit = con_test_limit, tau_tol = tau_tol, max_cores = max_cores)
			if sample_posterior != None:
				(parameters, median_parameters, model_evidence, plot_num, flatchain) = sample_posterior
			
				# print('fitting completed, parameters: ' + str(median_parameters))
				ln_Bayes_factor = model_evidence - prev_evidence
				# print('ln Bayes factor: ' + str(ln_Bayes_factor) + ' ' + str(datetime.datetime.now()))

				if ln_Bayes_factor > np.log(Bayes_threshold):
					prev_evidence = model_evidence
					last_accepted_med_parameters = median_parameters
					last_accepted_parameters = parameters
					# print('accepted ' + str(datetime.datetime.now()))
					if data['Texp_spectrum']['1665'] != []: # i.e. not blank, so Texp data is present
						num_param = 10
					else:
						num_param = 6
					num_gauss_to_fit += 1
					if quiet == False:
						CornerPlots(flatchain = flatchain, num_param = num_param, source_name = source_name)
						PlotModel(last_accepted_med_parameters, modified_data, plot_num, 'MCMC' + str(Bayes_threshold))
						plot_num += 1
				else:
					# print('rejected ' + str(datetime.datetime.now()))
					if quiet == False:
						PlotModel(last_accepted_med_parameters, modified_data, plot_num, 'REJECTED_MCMC' + str(Bayes_threshold))
						plot_num += 1
					keep_going = False
			else:
				# print('No finite members of Likelihood')
				keep_going = False
		accepted_med_parameters = list(itertools.chain(accepted_med_parameters, last_accepted_med_parameters))
		accepted_parameters = list(itertools.chain(accepted_parameters, last_accepted_parameters))
	accepted_med_parameters = [x for x in accepted_med_parameters]
	accepted_parameters = [x for x in accepted_parameters]
	if len(accepted_med_parameters) == 0:
		return ([None], [None])
	else:
		return (accepted_parameters, accepted_med_parameters)
def MakeModel(p = None, data = None, accepted_parameters = []):
	'''
	Constructs a Gaussian model based on parameters given in p.

	Parameters:
	p - Parameters of Gaussian component(s): [vel_1, fwhm_1, tau_1612_1, tau_1665_1, tau_1667_1, 
			tau_1720_1, Tex_1612_1, Tex_1665_1, Tex_1667_1, Tex_1720_1, ..., _N] for N components
	vel_XXXX - velocity axes of the 4 OH transitions

	Returns Gaussian models for each of the 4 OH transitions
	'''
	
	vel_1612 = data['vel_axis']['1612']
	vel_1665 = data['vel_axis']['1665']
	vel_1667 = data['vel_axis']['1667']
	vel_1720 = data['vel_axis']['1720']
	
	if accepted_parameters != []:
		if data['Texp_spectrum']['1665'] != []:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = MakeModel(p = accepted_parameters, data = data)
		else:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p = accepted_parameters, data = data)
	else:
		tau_model_1612 = np.zeros(len(vel_1612))
		tau_model_1665 = np.zeros(len(vel_1665))
		tau_model_1667 = np.zeros(len(vel_1667))
		tau_model_1720 = np.zeros(len(vel_1720))

		if data['Texp_spectrum']['1665'] != []:

			Texp_model_1612 = np.zeros(len(vel_1612))
			Texp_model_1665 = np.zeros(len(vel_1665))
			Texp_model_1667 = np.zeros(len(vel_1667))
			Texp_model_1720 = np.zeros(len(vel_1720))
	
	if data['Texp_spectrum']['1665'] != []:
		for component in range(int(len(p) / 10)): 
			[vel, fwhm, tau_1612, tau_1665, tau_1667, tau_1720, Texp_1612, Texp_1665, Texp_1667, Texp_1720] = p[component * 10:component * 10 + 10]

			tau_model_1612 += Gaussian(vel, fwhm, tau_1612)(vel_1612)
			tau_model_1665 += Gaussian(vel, fwhm, tau_1665)(vel_1665)
			tau_model_1667 += Gaussian(vel, fwhm, tau_1667)(vel_1667)
			tau_model_1720 += Gaussian(vel, fwhm, tau_1720)(vel_1720)

			Texp_model_1612 += Gaussian(vel, fwhm, Texp_1612)(vel_1612)
			Texp_model_1665 += Gaussian(vel, fwhm, Texp_1665)(vel_1665)
			Texp_model_1667 += Gaussian(vel, fwhm, Texp_1667)(vel_1667)
			Texp_model_1720 += Gaussian(vel, fwhm, Texp_1720)(vel_1720)

		return (tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720)	
	else:
		for component in range(int(len(p) / 6)): 
			[vel, fwhm, tau_1612, tau_1665, tau_1667, tau_1720] = p[component * 6:component * 6 + 6]

			tau_model_1612 += Gaussian(vel, fwhm, tau_1612)(vel_1612)
			tau_model_1665 += Gaussian(vel, fwhm, tau_1665)(vel_1665)
			tau_model_1667 += Gaussian(vel, fwhm, tau_1667)(vel_1667)
			tau_model_1720 += Gaussian(vel, fwhm, tau_1720)(vel_1720)

		return (tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720)
def Gaussian(mean = None, fwhm = None, height = None, sigma = None, amp = None):
	'''
	Generates a Gaussian profile with the given parameters.
	'''

	if sigma == None:
		sigma = fwhm / (2. * math.sqrt(2. * np.log(2.)))

	if height == None:
		height = amp / (sigma * math.sqrt(2.* math.pi))

	return lambda x: height * np.exp(-((x - mean)**2.) / (2.*sigma**2.))
def PlotModel(p = None, data = None, plot_num = 'Test', plot_type = None):
	source_name = data['source_name']

	vel_1612 = data['vel_axis']['1612']
	vel_1665 = data['vel_axis']['1665']
	vel_1667 = data['vel_axis']['1667']
	vel_1720 = data['vel_axis']['1720']

	tau_1612 = data['tau_spectrum']['1612']
	tau_1665 = data['tau_spectrum']['1665']
	tau_1667 = data['tau_spectrum']['1667']
	tau_1720 = data['tau_spectrum']['1720']
	
	if data['Texp_spectrum']['1665'] != []:
		Texp_1612 = data['Texp_spectrum']['1612']
		Texp_1665 = data['Texp_spectrum']['1665']
		Texp_1667 = data['Texp_spectrum']['1667']
		Texp_1720 = data['Texp_spectrum']['1720']

		(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = MakeModel(p = p, data = data)
		fig, axes = plt.subplots(nrows = 5, ncols = 2, sharex = True)

		axes[0,0].plot(vel_1612, tau_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[0,0].plot(vel_1612, tau_model_1612, color = 'black', linewidth = 1)
		axes[1,0].plot(vel_1665, tau_1665, color = 'green', label = '1665 MHz', linewidth = 1)
		axes[1,0].plot(vel_1665, tau_model_1665, color = 'black', linewidth = 1)
		axes[2,0].plot(vel_1667, tau_1667, color = 'red', label = '1667 MHz', linewidth = 1)
		axes[2,0].plot(vel_1667, tau_model_1667, color = 'black', linewidth = 1)
		axes[3,0].plot(vel_1720, tau_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[3,0].plot(vel_1720, tau_model_1720, color = 'black', linewidth = 1)
		axes[4,0].plot(vel_1612, tau_1612 - tau_model_1612, color = 'blue', linewidth = 1)
		axes[4,0].plot(vel_1665, tau_1665 - tau_model_1665, color = 'green', linewidth = 1)
		axes[4,0].plot(vel_1667, tau_1667 - tau_model_1667, color = 'red', linewidth = 1)
		axes[4,0].plot(vel_1720, tau_1720 - tau_model_1720, color = 'cyan', linewidth = 1)
		axes[0,1].plot(vel_1612, Texp_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[0,1].plot(vel_1612, Texp_model_1612, color = 'black', linewidth = 1)
		axes[1,1].plot(vel_1665, Texp_1665, color = 'green', label = '1665 MHz', linewidth = 1)
		axes[1,1].plot(vel_1665, Texp_model_1665, color = 'black', linewidth = 1)
		axes[2,1].plot(vel_1667, Texp_1667, color = 'red', label = '1667 MHz', linewidth = 1)
		axes[2,1].plot(vel_1667, Texp_model_1667, color = 'black', linewidth = 1)
		axes[3,1].plot(vel_1720, Texp_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[3,1].plot(vel_1720, Texp_model_1720, color = 'black', linewidth = 1)
		axes[4,1].plot(vel_1612, Texp_1612 - Texp_model_1612, color = 'blue', linewidth = 1)
		axes[4,1].plot(vel_1665, Texp_1665 - Texp_model_1665, color = 'green', linewidth = 1)
		axes[4,1].plot(vel_1667, Texp_1667 - Texp_model_1667, color = 'red', linewidth = 1)
		axes[4,1].plot(vel_1720, Texp_1720 - Texp_model_1720, color = 'cyan', linewidth = 1)
	else:
		(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p = p, data = data)
		fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)

		axes[0].plot(vel_1612, tau_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[0].plot(vel_1612, tau_model_1612, color = 'black', linewidth = 1)
		axes[1].plot(vel_1665, tau_1665, color = 'green', label = '1665 MHz', linewidth = 1)
		axes[1].plot(vel_1665, tau_model_1665, color = 'black', linewidth = 1)
		axes[2].plot(vel_1667, tau_1667, color = 'red', label = '1667 MHz', linewidth = 1)
		axes[2].plot(vel_1667, tau_model_1667, color = 'black', linewidth = 1)
		axes[3].plot(vel_1720, tau_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[3].plot(vel_1720, tau_model_1720, color = 'black', linewidth = 1)
		axes[4].plot(vel_1612, tau_1612 - tau_model_1612, color = 'blue', linewidth = 1)
		axes[4].plot(vel_1665, tau_1665 - tau_model_1665, color = 'green', linewidth = 1)
		axes[4].plot(vel_1667, tau_1667 - tau_model_1667, color = 'red', linewidth = 1)
		axes[4].plot(vel_1720, tau_1720 - tau_model_1720, color = 'cyan', linewidth = 1)	

	if plot_num == 'Final':
		try:
			os.makedir('BGD_Output')
		except:
			pass
		try:
			os.makedir('BGD_Output/Final_Models')
		except:
			pass
		plt.savefig('BGD_Output/Final_Models/' + str(plot_type) + '_' + source_name + '_Final_model.pdf')
		# plt.show()
	else:
		try:
			os.makedir('BGD_Output')
		except:
			pass
		try:
			os.makedir('BGD_Output/Models')
		except:
			pass
		plt.savefig('BGD_Output/Models/' + str(plot_type) + '_' + source_name + '_Plot_' + str(plot_num) + '_model.pdf')
	# plt.show()
	plt.close()
def PlotChain(data = None, ndim = None, chain = None, plot_num = None, phase = None):
	for parameter in range(ndim):
		plt.figure()
		for walker in range(chain.shape[0]):
			plt.plot(range(chain.shape[1]), chain[walker,:,parameter])
		plt.title(data['source_name'] + ' for param ' + str(parameter) + ': burn in')
		
		try:
			os.makedir('BGD_Output')
		except:
			pass
		try:
			os.makedir('BGD_Output/Chain_Plots')
		except:
			pass

		plt.savefig('BGD_Output/Chain_Plots/Chain_plot_' + str(phase) + '_' + data['source_name'] + '_' + str(parameter) + '_Plot_' + str(plot_num) + '.pdf')
		plot_num += 1
		plt.close()
	return plot_num
def SamplePosterior(data = None, fitted_params = None, quiet = True, plot_num = None, min_vel = None, max_vel = None, num_gauss = None, dv = None, prev_params = None, accepted_parameters = [], con_test_limit = None, tau_tol = None, max_cores = None):
	
	if data['Texp_spectrum']['1665'] != []:
		if fitted_params != None:
			num_gauss = int(len(fitted_params) / 10)
		ndim = int(10 * num_gauss)
	else:
		if fitted_params != None:
			num_gauss = int(len(fitted_params) / 6)
		ndim = int(6 * num_gauss)	

	num_gauss = int(num_gauss) 
	nwalkers = int(32 * ndim)
	burn_iterations = 5000
	iterations = 1000

	# defining initial positions
	# print('\tdefining intitial positions '+ str(datetime.datetime.now()))
	
	if fitted_params == None:
		p0 = p0Gen(data = data, min_vel = min_vel, max_vel = max_vel, num_gauss = num_gauss, dv = dv, prev_params = prev_params, tau_tol = tau_tol)
		if p0 == None:
			bad_guess = []
			for component in range(num_gauss):
				component_bad_guess = np.ones(int(ndim/num_gauss)).tolist()
				component_bad_guess[0] = random.uniform(min_vel, max_vel)
				bad_guess += component_bad_guess
			p0 = [FixInitialValue(p = [x + random.uniform(-1e-5, 1e-5) for x in bad_guess], data = data, min_vel = min_vel, max_vel = max_vel, tau_tol = tau_tol) for y in range(int(nwalkers))]		
	else:
		p0 = [FixInitialValue(p = [x + random.uniform(-1e-5, 1e-5) for x in fitted_params], data = data, min_vel = min_vel, max_vel = max_vel, tau_tol = tau_tol) for y in range(int(nwalkers))]

	p0 = [[x + random.uniform(-1e-6, 1e-6) for x in y] for y in p0]
	
	nwalkers = len(p0)


	# initiating and running mcmc
	# print('\trunning MCMC '+ str(datetime.datetime.now()))

	# set number of cores:
	pool = MPIPool() 
	if not pool.is_master():
		pool.wait()
		sys.exit(0)

	# initialise sampler object
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [data, min_vel, max_vel, accepted_parameters, dv, tau_tol], pool = pool)
	# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = [data, min_vel, max_vel, accepted_parameters, dv, tau_tol])
	try:
		pos, prob, state = sampler.run_mcmc(p0, burn_iterations)
	except ValueError: # sometimes there is an error within emcee (random.randint)
		pos, prob, state = sampler.run_mcmc(p0, burn_iterations)

	# plot walker chains during burn in phase
	if quiet == False:
		plot_num = PlotChain(data = data, ndim = ndim, chain = sampler.chain, plot_num = plot_num, phase = 'Burn')

	########## v Convergence Test v ##########

	(test_result, repaired_pos) = ConvergenceTest(data = data, sampler_chain = sampler.chain, num_gauss = num_gauss, test_limit = con_test_limit)

	if test_result == 'Fail':
		# print('Burning phase failed to converge. Burning again.')
		try:
			pos, prob, state = sampler.run_mcmc(repaired_pos, burn_iterations) # burns again
		except ValueError: # sometimes there is an error within emcee (random.randint)
			pos, prob, state = sampler.run_mcmc(repaired_pos, burn_iterations) # burns again
		
		(test_result, repaired_pos) = ConvergenceTest(data = data, sampler_chain = sampler.chain, num_gauss = num_gauss, test_limit = con_test_limit)
		if test_result == 'Fail':
			# print('Burning phase failed to converge. Burning yet again.')
			try:
				pos, prob, state = sampler.run_mcmc(repaired_pos, burn_iterations) # burns again
			except ValueError: # sometimes there is an error within emcee (random.randint)
				pos, prob, state = sampler.run_mcmc(repaired_pos, burn_iterations) # burns again
			
			# (test_result, repaired_pos) = ConvergenceTest(data = data, sampler_chain = sampler.chain, num_gauss = num_gauss, test_limit = con_test_limit)
			# if test_result == 'Fail':
				# print('Warning: Markov chain failed to converge for ' + data['source_name'])

	########## ^ Convergence Test ^ ##########

	# run final time
	sampler.reset()
	sampler.run_mcmc(pos, iterations)

	# plot walker chains during final phase
	if quiet == False:
		plot_num = PlotChain(data = data, ndim = ndim, chain = sampler.chain, plot_num = plot_num, phase = 'Final')

	# remove steps where lnprob = -inf
	original_flat_chain = sampler.flatchain
	original_flat_lnprob = sampler.flatlnprobability

	new_flat_chain = []
	new_flat_lnprob = []

	for step in range(len(original_flat_lnprob)):
		if original_flat_lnprob[step] != -np.inf:
			new_flat_chain.append(original_flat_chain[step])
			new_flat_lnprob.append(original_flat_lnprob[step])

	new_flat_chain = np.array(new_flat_chain)
	new_flat_lnprob = np.array(new_flat_lnprob)

	# find best parameters and errors
	if len(new_flat_chain) != 0:
		(parameters, log_evidence) = BestParams(chain = new_flat_chain, lnprob = new_flat_lnprob)
		median_parameters = [x[1] for x in parameters]

		return (parameters, median_parameters, log_evidence, plot_num, new_flat_chain)
	else:
		return None
def CornerPlots(flatchain = None, num_param = None, source_name = None):
	'''
	num_param = 6 for absorption only, = 10 for absorption and emission data
	'''
	for feature in range(int(flatchain.shape[1] / num_param)):
		vel_chain = flatchain[:, feature * num_param + 0]
		fwhm_chain = flatchain[:, feature * num_param + 1]
		tau_1612_chain = flatchain[:, feature * num_param + 2]
		tau_1665_chain = flatchain[:, feature * num_param + 3]
		tau_1667_chain = flatchain[:, feature * num_param + 4]
		tau_1720_chain = flatchain[:, feature * num_param + 5]
		if int(num_param) == 10:
			Texp_1612_chain = flatchain[:, feature * num_param + 6]
			Texp_1665_chain = flatchain[:, feature * num_param + 7]
			Texp_1667_chain = flatchain[:, feature * num_param + 8]
			Texp_1720_chain = flatchain[:, feature * num_param + 9]

		feature_vel = round(corner.quantile(vel_chain, [0.5])[0])

		try:
			corner.corner(np.transpose([vel_chain, fwhm_chain, tau_1612_chain, tau_1665_chain, tau_1667_chain, tau_1720_chain]), show_titles = True, range = (0.9, 0.9, 0.9, 0.9, 0.9, 0.9), labels = ['Vel', 'fwhm', 'tau(1612)', 'tau(1665)', 'tau(1667)', 'tau(1720)'])
			
			try:
				os.makedir('BGD_Output')
			except:
				pass
			try:
				os.makedir('BGD_Output/Corner_Plots')
			except:
				pass

			plt.savefig('BGD_Output/Corner_Plots/tau_' + str(source_name) + '_vel_' + str(feature_vel) + '.pdf')
			plt.close()

			if int(num_param) == 10:
				corner.corner(np.transpose([vel_chain, fwhm_chain, Texp_1612_chain, Texp_1665_chain, Texp_1667_chain, Texp_1720_chain]), show_titles = True, range = (0.9, 0.9, 0.9, 0.9, 0.9, 0.9), labels = ['Vel', 'fwhm', 'Texp(1612)', 'Texp(1665)', 'Texp(1667)', 'Texp(1720)'])
			
				try:
					os.makedir('BGD_Output')
				except:
					pass
				try:
					os.makedir('BGD_Output/Corner_Plots')
				except:
					pass

				plt.savefig('BGD_Output/Corner_Plots/Texp_' + str(source_name) + '_vel_' + str(feature_vel) + '.pdf')
				plt.close()
		except:
			pass
def FixInitialValue(p = None, data = None, min_vel = None, max_vel = None, tau_tol = 5):
	# print('\nFixing initial value: ' + str(p))
	vel_list = []
	if data['Texp_spectrum']['1665'] != []:
		num_param = 10
	else:
		num_param = 6

	for vel_ind in range(0,len(p),num_param):
		vel_list.append(p[vel_ind])
	vel_list = sorted(vel_list)
	
	for vel_ind in range(0,len(p),num_param):
		# print('\treplacing ' + str(p[vel_ind]) + ' with ' + str(vel_list[ind(vel_ind/num_param)]) )
		p[vel_ind] = vel_list[int(vel_ind/num_param)]
	
	for component in range(int(len(p)/num_param)):
		if p[component * num_param + 0] < min_vel:
			p[component * num_param + 0] = min_vel
		if p[component * num_param + 0] > max_vel: # vel
			p[component * num_param + 0] = max_vel
		if p[component * num_param + 1] <= 1.e-3: # fwhm
			p[component * num_param + 1] = 0.01
		if p[component * num_param + 2] > tau_tol * abs(max(data['tau_spectrum']['1612'])): # tau_1612
			p[component * num_param + 2] = (tau_tol * 0.9) * abs(max(data['tau_spectrum']['1612']))
		if p[component * num_param + 2] < -tau_tol * abs(min(data['tau_spectrum']['1612'])): # tau_1612
			p[component * num_param + 2] = -(tau_tol * 0.9) * abs(min(data['tau_spectrum']['1612']))
		if p[component * num_param + 3] > tau_tol * abs(max(data['tau_spectrum']['1665'])): # tau_1665
			p[component * num_param + 3] = (tau_tol * 0.9) * abs(max(data['tau_spectrum']['1665']))
		if p[component * num_param + 3] < -tau_tol * abs(min(data['tau_spectrum']['1665'])): # tau_1665
			p[component * num_param + 3] = -(tau_tol * 0.9) * abs(min(data['tau_spectrum']['1665']))
		if p[component * num_param + 4] > tau_tol * abs(max(data['tau_spectrum']['1667'])): # tau_1667
			p[component * num_param + 4] = (tau_tol * 0.9) * abs(max(data['tau_spectrum']['1667']))
		if p[component * num_param + 4] < -tau_tol * abs(min(data['tau_spectrum']['1667'])): # tau_1667
			p[component * num_param + 4] = -(tau_tol * 0.9) * abs(min(data['tau_spectrum']['1667']))
		if p[component * num_param + 5] > tau_tol * abs(max(data['tau_spectrum']['1720'])): # tau_1720
			p[component * num_param + 5] = (tau_tol * 0.9) * abs(max(data['tau_spectrum']['1720']))
		if p[component * num_param + 5] < -tau_tol * abs(min(data['tau_spectrum']['1720'])): # tau_1720
			p[component * num_param + 5] = -(tau_tol * 0.9) * abs(min(data['tau_spectrum']['1720']))
			
		if data['Texp_spectrum']['1665'] != []:
			if p[component * num_param + 6] > 5. * abs(max(data['Texp_spectrum']['1612'])): # Texp_1612
				p[component * num_param + 6] = 4. * abs(max(data['Texp_spectrum']['1612']))
			if p[component * num_param + 6] < -5. * abs(min(data['Texp_spectrum']['1612'])): # Texp_1612
				p[component * num_param + 6] = -4. * abs(min(data['Texp_spectrum']['1612']))
			if p[component * num_param + 7] > 5. * abs(max(data['Texp_spectrum']['1665'])): # Texp_1665
				p[component * num_param + 7] = 4. * abs(max(data['Texp_spectrum']['1665']))
			if p[component * num_param + 7] < -5. * abs(min(data['Texp_spectrum']['1665'])): # Texp_1665
				p[component * num_param + 7] = -4. * abs(min(data['Texp_spectrum']['1665']))
			if p[component * num_param + 8] > 5. * abs(max(data['Texp_spectrum']['1667'])): # Texp_1667
				p[component * num_param + 8] = 4. * abs(max(data['Texp_spectrum']['1667']))
			if p[component * num_param + 8] < -5. * abs(min(data['Texp_spectrum']['1667'])): # Texp_1667
				p[component * num_param + 8] = -4. * abs(min(data['Texp_spectrum']['1667']))
			if p[component * num_param + 9] > 5. * abs(max(data['Texp_spectrum']['1720'])): # Texp_1720
				p[component * num_param + 9] = 4. * abs(max(data['Texp_spectrum']['1720']))
			if p[component * num_param + 9] < -5. * abs(min(data['Texp_spectrum']['1720'])): # Texp_1720
				p[component * num_param + 9] = -4. * abs(min(data['Texp_spectrum']['1720']))
	return p
def p0Gen(data = None, min_vel = None, max_vel = None, num_gauss = None, dv = None, prev_params = None, tau_tol = None):
	def mpfitp0(p, fjac, data):
		if data['Texp_spectrum']['1665'] != []:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = MakeModel(p, data)
			residuals = np.concatenate((
					(tau_model_1612 - data['tau_spectrum']['1612']) / data['tau_rms']['1612'], 
					(tau_model_1665 - data['tau_spectrum']['1665']) / data['tau_rms']['1665'], 
					(tau_model_1667 - data['tau_spectrum']['1667']) / data['tau_rms']['1667'], 
					(tau_model_1720 - data['tau_spectrum']['1720']) / data['tau_rms']['1720'], 
					(Texp_model_1612 - data['Texp_spectrum']['1612']) / data['Texp_rms']['1612'], 
					(Texp_model_1665 - data['Texp_spectrum']['1665']) / data['Texp_rms']['1665'], 
					(Texp_model_1667 - data['Texp_spectrum']['1667']) / data['Texp_rms']['1667'], 
					(Texp_model_1720 - data['Texp_spectrum']['1720']) / data['Texp_rms']['1720'])) 
		else:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p, data)
			residuals = np.concatenate((
					(tau_model_1612 - data['tau_spectrum']['1612']) / data['tau_rms']['1612'], 
					(tau_model_1665 - data['tau_spectrum']['1665']) / data['tau_rms']['1665'], 
					(tau_model_1667 - data['tau_spectrum']['1667']) / data['tau_rms']['1667'], 
					(tau_model_1720 - data['tau_spectrum']['1720']) / data['tau_rms']['1720'])) 
		
		return [0, residuals]
	def InterestingVel(vel_axis = None, spectrum = None, spectrum_rms = None, min_vel = None, max_vel = None, ranges = None):
		id_vel_list = []
		# Flag features
		for _range in ranges:
			if _range%2 != 0:
				_range += 1

			dx = Derivative(vel_axis, spectrum, _range)
			dx2 = Derivative(vel_axis, dx, 2)

			rms_dx = FindRMS(dx)
			rms_dx2 = FindRMS(dx2)

			spectrum_zero = [abs(x) < 2. * spectrum_rms for x in spectrum]
			spectrum_pos = [x > 2. * spectrum_rms for x in spectrum]
			
			dx_pos = [x > 2. * rms_dx for x in dx]
			dx2_pos = [x > 2. * rms_dx2 for x in dx2]
			
			dx_zero = Zeros(vel_axis, dx, rms_dx)
			dx2_zero = Zeros(vel_axis, dx2, rms_dx2)		

			vel_list1 = [vel_axis[z] for z in range(int(len(dx_pos) - 1)) if dx_zero[z] == True and spectrum_pos[z] != dx2_pos[z]]
			vel_list2 = [vel_axis[z] for z in range(1,int(len(dx2_zero) - 1)) if dx2_zero[z] == True and spectrum_zero[z] == False and np.any([x <= dx2_zero[z+1] and x >= dx2_zero[z-1] for x in dx_zero]) == False]
			
			vel_list = np.concatenate((vel_list1, vel_list2))
			
			id_vel_list.append(vel_list)
		id_vel_list = sorted([val for sublist in id_vel_list for val in sublist])

		if len(id_vel_list) != 0:
			id_vel_list = [x for x in id_vel_list if x >= min_vel and x <= max_vel]
			return id_vel_list
		else:
			# print('No interesting velocities identified')
			return None
	
	nwalkers = num_gauss * 30
	interesting_vel_master = []

	if data['Texp_spectrum']['1665'] != []:
		spectrum_list = ['tau', 'Texp']
	else:
		spectrum_list = ['tau']
	
	for spectrum in spectrum_list:
		for freq in ['1612', '1665', '1667', '1720']:
			interesting_vel = InterestingVel(
				vel_axis = data['vel_axis'][freq], 
				spectrum = data[spectrum + '_spectrum'][freq],
				spectrum_rms = data[spectrum + '_rms'][freq],
				min_vel = min_vel, 
				max_vel = max_vel, 
				ranges = [min([len(data['vel_axis'][freq]) - 5, 10])])
			if interesting_vel != None:
				interesting_vel_master.append(interesting_vel)

	if len(interesting_vel_master) == 0:
		interesting_vel_master = (np.arange(min_vel, max_vel + dv, dv)).tolist() # should be ok as if no velocities were identified it's likely that it won't get past one gaussian
	else:
		interesting_vel_master = ReduceList(interesting_vel_master, 3 * dv, 1)
		interesting_vel_master = sorted([val for sublist in interesting_vel_master for val in sublist])

	if num_gauss > 2:
		vel_list = []
		
		if data['Texp_spectrum']['1665'] != []:
			for vel_ind in range(0,len(prev_params),10):
				vel_list.append(prev_params[vel_ind])
		else:
			for vel_ind in range(0,len(prev_params),6):
				vel_list.append(prev_params[vel_ind])
		
		vel_list = tuple(vel_list)
		interesting_vel_master = [tuple([x]) for x in interesting_vel_master]

		vel_combos = list(itertools.product([vel_list], interesting_vel_master))
		for x in range(len(vel_combos)):
			vel_combos[x] = [d for e in vel_combos[x] for d in e]

	else:
		vel_combos = list(itertools.product(interesting_vel_master, repeat = num_gauss))
	
	vel_combos = [sorted(x) for x in vel_combos]

	parinfo = []
	if data['Texp_spectrum']['1665'] != []:
		for component in range(num_gauss):
			parinfo += [{'parname': 'velocity', 'step': 1.e-2, 'limited': [1,1], 'limits': [min_vel,max_vel]}, 
					{'parname': 'fwhm', 'step': 1.e-3, 'limited': [1,1], 'limits':[0,5]},
					{'parname': 'tau_1612', 'step': 1.e-5},
					{'parname': 'tau_1665', 'step': 1.e-5},
					{'parname': 'tau_1667', 'step': 1.e-5},
					{'parname': 'tau_1720', 'step': 1.e-5},
					{'parname': 'Texp_1612', 'step': 1.e-5},
					{'parname': 'Texp_1665', 'step': 1.e-5},
					{'parname': 'Texp_1667', 'step': 1.e-5},
					{'parname': 'Texp_1720', 'step': 1.e-5}]
	else:
		for component in range(num_gauss):
			parinfo += [{'parname': 'velocity', 'step': 1.e-2, 'limited': [1,1], 'limits': [min_vel,max_vel]}, 
					{'parname': 'fwhm', 'step': 1.e-3, 'limited': [1,1], 'limits':[0,5]},
					{'parname': 'tau_1612', 'step': 1.e-5},
					{'parname': 'tau_1665', 'step': 1.e-5},
					{'parname': 'tau_1667', 'step': 1.e-5},
					{'parname': 'tau_1720', 'step': 1.e-5}]

	best_combo = None
	best_params = None
	best_llh = -np.inf

	# print('allowed velocity range = ' + str([min_vel - 2, max_vel + 2]))

	for combo in vel_combos:
		guess = []
		for vel in combo:
			if data['Texp_spectrum']['1665'] != []:
				guess += [vel, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
			else:
				guess += [vel, 1, 0.1, 0.1, 0.1, 0.1]
	
		mp = mpfit(mpfitp0, guess, parinfo = parinfo, functkw = {'data': data}, maxiter = 1000, quiet = True)
		fitted_params = mp.params
		llh = lnprob(x = fitted_params, data = data, min_vel = min_vel, max_vel = max_vel, dv = dv, tau_tol = tau_tol)
		# print('\tfitted params: ' + str(fitted_params) + ' llh: ' + str(llh))
		if llh == -np.inf:
			PrPrior(x = fitted_params, data = data, min_vel = min_vel, max_vel = max_vel, report = True, dv = dv, tau_tol = tau_tol)
			fitted_params = FixInitialValue(p = fitted_params, data = data, min_vel = min_vel, max_vel = max_vel, tau_tol = tau_tol)
			llh = lnprob(x = fitted_params, data = data, min_vel = min_vel, max_vel = max_vel, dv = dv, tau_tol = tau_tol)
			# print('\'fixed\' parameters: ' + str(fitted_params) + ' llh: ' + str(llh))
			if llh == -np.inf:
				PrPrior(x = fitted_params, data = data, min_vel = min_vel, max_vel = max_vel, report = True, dv = dv, tau_tol = tau_tol)
		
		if llh > best_llh:
			best_combo = combo
			best_params = mp.params
			best_llh = llh
	
	if best_combo != None:
		return [[x * np.random.uniform(0.9, 1.1) for x in best_params] for y in range(nwalkers)]
	else:
		# print('p0Gen failed!')
		return None
def Zeros(x_axis, y_axis, y_rms):
	'''
	produces a boolean array of whether or not an x value is a zero
	'''
	def mpfitZero(p, fjac, x_axis_subset, y_axis_subset):
		[m, c] = p
		model_y = [m * x + c for x in x_axis_subset]
		residuals = np.array([y_axis_subset[x] - model_y[x] for x in range(len(x_axis_subset))])

		return [0, residuals]
	
	gradient_min = abs(2.* y_rms / (x_axis[10] - x_axis[0]))

	zeros = np.zeros(len(x_axis))
	for x in range(5, int(len(x_axis) - 5)):
		x_axis_subset = x_axis[x-5:x+6]
		y_axis_subset = y_axis[x-5:x+6]

		guess = [1., 1.]
		parinfo = [	{'parname':'gradient','step':0.001}, 
					{'parname':'y intercept','step':0.001}]
		fa = {'x_axis_subset': x_axis_subset, 'y_axis_subset': y_axis_subset}

		mp = mpfit(mpfitZero, guess, parinfo = parinfo, functkw = fa, maxiter = 10000, quiet = True)
		[grad_fit, y_int_fit] = mp.params

		if abs(grad_fit) >= gradient_min:

			# find y values on either side of x to test sign. True = pos
			if grad_fit * x_axis[x-1] + y_int_fit > 0 and grad_fit * x_axis[x+1] + y_int_fit < 0:
				zeros[x] = 1
			elif grad_fit * x_axis[x-1] + y_int_fit < 0 and grad_fit * x_axis[x+1] + y_int_fit > 0:
				zeros[x] = 1

	return zeros
def Derivative(vel_axis, spectrum, _range):
	def mpfitDer(p, fjac, x, y):
		'''
		x and y should be small arrays of length '_range' (from parent function). 
		'''
		status = 0

		[m, c] = p # gradient and y intercept of line
		model_y = m * np.array(x) + c
		residuals = (np.array(y) - model_y)

		return [status, residuals]
	
	extra = [0] * int(_range / 2) # _range will be even
	dx = []

	for start in range(int(len(vel_axis) - _range)):
		x = vel_axis[start:int(start + _range + 1)]
		y = spectrum[start:int(start + _range + 1)]

		guess = [(y[0] - y[-1]) / (x[0] - x[-1]), 0]
		parinfo = [	{'parname':'gradient','step':0.0001, 'limited': [1, 1], 'limits': [-20., 20.]}, 
					{'parname':'y intercept','step':0.001, 'limited': [1, 1], 'limits': [-1000., 1000.]}]
		fa = {'x': x, 'y': y}
		
		mp = mpfit(mpfitDer, guess, parinfo = parinfo, functkw = fa, maxiter = 10000, quiet = True)
		gradient = mp.params[0]
		dx.append(gradient)
	dx = np.concatenate((extra, dx, extra))
	return dx
def Tex(Texp = None, tau = None, Tbg = None):
	Tex = Texp / (1 - np.exp(-tau)) + Tbg
	return Tex
def Texp(tau = None, Tbg = None, Tex = None):
	Texp = (Tex - Tbg) * (1 - np.exp(-tau))
	return Texp
def lnprob(x = None, data = None, min_vel = None, max_vel = None, accepted_parameters = [], dv = None, tau_tol = None):

	ln_prior = PrPrior(x = x, data = data, min_vel = min_vel, max_vel = max_vel, dv = dv, tau_tol = tau_tol)
	ln_llh = ln_prior

	if any(np.isnan(x)) or any(np.isinf(x)) or np.isinf(ln_prior) or np.isnan(ln_prior):
		return -np.inf
	else:
		vel_1612 = data['vel_axis']['1612']
		vel_1665 = data['vel_axis']['1665']
		vel_1667 = data['vel_axis']['1667']
		vel_1720 = data['vel_axis']['1720']

		tau_1612 = data['tau_spectrum']['1612']
		tau_1665 = data['tau_spectrum']['1665']
		tau_1667 = data['tau_spectrum']['1667']
		tau_1720 = data['tau_spectrum']['1720']

		tau_rms_1612 = data['tau_rms']['1612']
		tau_rms_1665 = data['tau_rms']['1665']
		tau_rms_1667 = data['tau_rms']['1667']
		tau_rms_1720 = data['tau_rms']['1720']

		[N_1612, N_1665, N_1667, N_1720] = [len(vel_1612), len(vel_1665), len(vel_1667), len(vel_1720)]

		if data['Texp_spectrum']['1665'] != []:

			Texp_1612 = data['Texp_spectrum']['1612']
			Texp_1665 = data['Texp_spectrum']['1665']
			Texp_1667 = data['Texp_spectrum']['1667']
			Texp_1720 = data['Texp_spectrum']['1720']

			Texp_rms_1612 = data['Texp_rms']['1612']
			Texp_rms_1665 = data['Texp_rms']['1665']
			Texp_rms_1667 = data['Texp_rms']['1667']
			Texp_rms_1720 = data['Texp_rms']['1720']

			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = MakeModel(p = x, data = data, accepted_parameters = accepted_parameters)

			sse_Texp_1612 = sum([(Texp_1612[a] - Texp_model_1612[a])**2. for a in range(len(Texp_1612))])
			sse_Texp_1665 = sum([(Texp_1665[a] - Texp_model_1665[a])**2. for a in range(len(Texp_1665))])
			sse_Texp_1667 = sum([(Texp_1667[a] - Texp_model_1667[a])**2. for a in range(len(Texp_1667))])
			sse_Texp_1720 = sum([(Texp_1720[a] - Texp_model_1720[a])**2. for a in range(len(Texp_1720))])

			llh_Texp_1612 = -(N_1612*np.log(math.sqrt(2.*math.pi)*Texp_rms_1612)) - (sse_Texp_1612/(2.*Texp_rms_1612**2.))
			llh_Texp_1665 = -(N_1665*np.log(math.sqrt(2.*math.pi)*Texp_rms_1665)) - (sse_Texp_1665/(2.*Texp_rms_1665**2.))
			llh_Texp_1667 = -(N_1667*np.log(math.sqrt(2.*math.pi)*Texp_rms_1667)) - (sse_Texp_1667/(2.*Texp_rms_1667**2.))
			llh_Texp_1720 = -(N_1720*np.log(math.sqrt(2.*math.pi)*Texp_rms_1720)) - (sse_Texp_1720/(2.*Texp_rms_1720**2.))


			ln_llh += llh_Texp_1612 + llh_Texp_1665 + llh_Texp_1667 + llh_Texp_1720
		
		else:
			(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p = x, data = data, accepted_parameters = accepted_parameters)


		sse_tau_1612 = sum([(tau_1612[a] - tau_model_1612[a])**2. for a in range(len(tau_1612))])
		sse_tau_1665 = sum([(tau_1665[a] - tau_model_1665[a])**2. for a in range(len(tau_1665))])
		sse_tau_1667 = sum([(tau_1667[a] - tau_model_1667[a])**2. for a in range(len(tau_1667))])
		sse_tau_1720 = sum([(tau_1720[a] - tau_model_1720[a])**2. for a in range(len(tau_1720))])

		llh_tau_1612 = -(N_1612*np.log(math.sqrt(2.*math.pi)*tau_rms_1612)) - (sse_tau_1612/(2.*tau_rms_1612**2.))
		llh_tau_1665 = -(N_1665*np.log(math.sqrt(2.*math.pi)*tau_rms_1665)) - (sse_tau_1665/(2.*tau_rms_1665**2.))
		llh_tau_1667 = -(N_1667*np.log(math.sqrt(2.*math.pi)*tau_rms_1667)) - (sse_tau_1667/(2.*tau_rms_1667**2.))
		llh_tau_1720 = -(N_1720*np.log(math.sqrt(2.*math.pi)*tau_rms_1720)) - (sse_tau_1720/(2.*tau_rms_1720**2.))


		ln_llh += llh_tau_1612 + llh_tau_1665 + llh_tau_1667 + llh_tau_1720
		# print('\t\t' + str(llh_tau_1612) + '\t' + str(llh_tau_1665) + '\t' + str(llh_tau_1667) + '\t' + str(llh_tau_1720))

		return ln_llh
def TestPriors(data = None, dv = None, tau_tol = None):
	'''
	Priors should be normalised to integrate to 1 over parameter space
	'''
	min_vel = max([min(data['vel_axis']['1612']), min(data['vel_axis']['1665']), min(data['vel_axis']['1667']), min(data['vel_axis']['1720'])])
	max_vel = min([max(data['vel_axis']['1612']), max(data['vel_axis']['1665']), max(data['vel_axis']['1667']), max(data['vel_axis']['1720'])])
	
	trials = 5000

	# check velocity
	x_set = [[np.random.uniform(min_vel-10, max_vel+10), 0, 0, 0, 0, 0] for x in range(trials)]
	velocities = [x[0] for x in x_set]
	priors = [PrPriorVel(x = x, model_dim = 6, data = data, min_vel = min_vel + 0.25*(max_vel - min_vel), max_vel = max_vel - 0.25*(max_vel - min_vel)) for x in x_set]
	
	ln_Pr_prior_vel_integral = Integral(velocities, priors)
	print('ln_Pr_prior_vel_integral: ' + str(ln_Pr_prior_vel_integral))
	plt.figure()
	plt.scatter(velocities, priors, color = 'black')
	plt.title('Velocity')
	plt.show()
	plt.close()

	# check fwhm
	fwhm_set = [np.random.uniform(-5, 20) for x in range(trials)]
	priors = [PrPriorFWHM(fwhm = fwhm, dv = dv) for fwhm in fwhm_set]

	ln_Pr_prior_fwhm_integral = Integral(fwhm_set, priors)
	print('ln_Pr_prior_fwhm_integral: ' + str(ln_Pr_prior_fwhm_integral))
	plt.figure()
	plt.scatter(fwhm_set, priors, color = 'black')
	plt.title('FWHM')
	plt.show()
	plt.close()

	# check taus

	max_tau_1612 = 6. * abs(max(data['tau_spectrum']['1612']))
	max_tau_1665 = 6. * abs(max(data['tau_spectrum']['1665']))
	max_tau_1667 = 6. * abs(max(data['tau_spectrum']['1667']))
	max_tau_1720 = 6. * abs(max(data['tau_spectrum']['1720']))

	min_tau_1612 = -6. * abs(min(data['tau_spectrum']['1612']))
	min_tau_1665 = -6. * abs(min(data['tau_spectrum']['1665']))
	min_tau_1667 = -6. * abs(min(data['tau_spectrum']['1667']))
	min_tau_1720 = -6. * abs(min(data['tau_spectrum']['1720']))

	tau_set = [[np.random.uniform(min_tau_1612, max_tau_1612), np.random.uniform(min_tau_1665, max_tau_1665), np.random.uniform(min_tau_1667, max_tau_1667), np.random.uniform(min_tau_1720, max_tau_1720)] for x in range(trials)]
	tau_sum_set = [_set[1]/5. + _set[2]/9. - _set[0] - _set[3] for _set in tau_set]
	priors = [PrPriorTau(taus = _set, data = data, tau_tol = tau_tol) for _set in tau_set]

	integral = Integral(tau_sum_set, priors)
	print('integral = ' + str(integral))

	plt.figure()
	plt.scatter(tau_sum_set, priors, color = 'black')
	# plt.plot(sorted(tau_sum_set), model, color = 'red')
	plt.title('Tau')
	plt.show()
	plt.close()

	
	if data['Texp_spectrum']['1665'] != []:
		# check Texps
		max_Texp_1612 = 6. * abs(max(data['Texp_spectrum']['1612']))
		max_Texp_1665 = 6. * abs(max(data['Texp_spectrum']['1665']))
		max_Texp_1667 = 6. * abs(max(data['Texp_spectrum']['1667']))
		max_Texp_1720 = 6. * abs(max(data['Texp_spectrum']['1720']))

		min_Texp_1612 = -6. * abs(min(data['Texp_spectrum']['1612']))
		min_Texp_1665 = -6. * abs(min(data['Texp_spectrum']['1665']))
		min_Texp_1667 = -6. * abs(min(data['Texp_spectrum']['1667']))
		min_Texp_1720 = -6. * abs(min(data['Texp_spectrum']['1720']))

		Texp_set = [[np.random.uniform(min_Texp_1612, max_Texp_1612), np.random.uniform(min_Texp_1665, max_Texp_1665), np.random.uniform(min_Texp_1667, max_Texp_1667), np.random.uniform(min_Texp_1720, max_Texp_1720)] for x in range(trials)]
		priors = [PrPriorTexp(Texps = _set, data = data) for _set in Texp_set]
	
		Texp_1612 = [_set[0] for _set in Texp_set]
		Texp_1665 = [_set[1] for _set in Texp_set]
		Texp_1667 = [_set[2] for _set in Texp_set]
		Texp_1720 = [_set[3] for _set in Texp_set]

		ln_Pr_prior_Texp_integral = Integral(Texp_1612, priors) + Integral(Texp_1665, priors) + Integral(Texp_1667, priors) + Integral(Texp_1720, priors)
		print('ln_Pr_prior_Texp_integral: ' + str(ln_Pr_prior_Texp_integral))
		plt.figure()
		plt.scatter(Texp_1612, priors, color = 'blue')
		plt.scatter(Texp_1665, priors, color = 'green')
		plt.scatter(Texp_1667, priors, color = 'red')
		plt.scatter(Texp_1720, priors, color = 'cyan')
		plt.title('Texp')
		plt.show()
		plt.close()

		# check Tex

		Tex_set = [[np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(-20, 20)] for x in range(trials)]
		Tex_sum_set = [1.612231 / _set[0] - 1.665402 / _set[1] - 1.667359 / _set[2] + 1.72053 / _set[3] for _set in Tex_set]
		priors = [PrPriorTex(Texs = _set, Tex_errors = [1., 1., 1., 1.], data = data) for _set in Tex_set]
	
		Tex_1612 = [_set[0] for _set in Tex_set]
		Tex_1665 = [_set[1] for _set in Tex_set]
		Tex_1667 = [_set[2] for _set in Tex_set]
		Tex_1720 = [_set[3] for _set in Tex_set]

		ln_Pr_prior_Tex_integral = Integral(Tex_1612, priors) + Integral(Tex_1665, priors) + Integral(Tex_1667, priors) + Integral(Tex_1720, priors)
		print('ln_Pr_prior_Tex_integral: ' + str(ln_Pr_prior_Tex_integral))
		plt.figure()
		plt.scatter(Tex_1612, priors, color = 'blue')
		plt.scatter(Tex_1665, priors, color = 'green')
		plt.scatter(Tex_1667, priors, color = 'red')
		plt.scatter(Tex_1720, priors, color = 'cyan')
		plt.title('Tex')
		plt.show()
		plt.close()

		plt.figure()
		plt.scatter(Tex_sum_set, priors, color = 'black')
		plt.title('Tex')
		plt.show()
		plt.close()
def PrPriorVel(x = None, model_dim = None, data = None, min_vel = None, max_vel = None, report = False):
	ln_Pr_prior_vel = 0

	if min_vel != None:
		prev_vel = min_vel
	else:
		prev_vel = max([min(data['vel_axis']['1612']), min(data['vel_axis']['1665']), min(data['vel_axis']['1667']), min(data['vel_axis']['1720'])])
	
	for vel_ind in range(0, len(x), model_dim):
		vel = x[vel_ind]
		if vel >= prev_vel:
			if len(x) > vel_ind + model_dim:
				next_vel = x[vel_ind + model_dim]
			else:
				if max_vel != None:
					next_vel = max_vel
				else:
					next_vel = min([max(data['vel_axis']['1612']), max(data['vel_axis']['1665']), max(data['vel_axis']['1667']), max(data['vel_axis']['1720'])])
			if vel <= next_vel:
				ln_Pr_prior_vel += -np.log(next_vel - prev_vel)
				prev_vel = vel
			else:
				if report == True:
					print('ln_Pr_prior = -inf because velocity not in allowable range')
					print('vel = ' + str(vel))
					print('next vel = ' + str(next_vel))
				ln_Pr_prior_vel = -np.inf
				break				
		else:
			if report == True:
				print('ln_Pr_prior = -inf because velocity not in allowable range')
				print('vel = ' + str(vel))
				print('previous vel = ' + str(prev_vel))
			ln_Pr_prior_vel = -np.inf
			break
	return ln_Pr_prior_vel
def PrPriorFWHM(fwhm = None, report = False, fwhm_soft_lim = 5., fwhm_hard_lim = 7., dv = None):
	
	fwhm_height = 2. / (fwhm_soft_lim + fwhm_hard_lim - 3.*dv)

	if fwhm < 3. * dv:
		Pr_prior_fwhm = fwhm * fwhm_height / (3.*dv)
	elif fwhm < fwhm_soft_lim:
		Pr_prior_fwhm = fwhm_height
	elif fwhm < fwhm_hard_lim:
		Pr_prior_fwhm = (fwhm_height/(fwhm_hard_lim - fwhm_soft_lim)) * (fwhm_hard_lim - fwhm)
	else:
		if report == True:
			print('ln_Pr_prior = -inf because fwhm is not in allowable range')
			print('fwhm = ' + str(fwhm))
		Pr_prior_fwhm = 0
	return np.log(Pr_prior_fwhm)
def PrPriorTau(taus = None, data = None, report = False, tau_tol = 5):

	[tau_1612, tau_1665, tau_1667, tau_1720] = taus
	
	max_tau_1612 = tau_tol * abs(max(data['tau_spectrum']['1612']))
	max_tau_1665 = tau_tol * abs(max(data['tau_spectrum']['1665']))
	max_tau_1667 = tau_tol * abs(max(data['tau_spectrum']['1667']))
	max_tau_1720 = tau_tol * abs(max(data['tau_spectrum']['1720']))

	min_tau_1612 = -tau_tol * abs(min(data['tau_spectrum']['1612']))
	min_tau_1665 = -tau_tol * abs(min(data['tau_spectrum']['1665']))
	min_tau_1667 = -tau_tol * abs(min(data['tau_spectrum']['1667']))
	min_tau_1720 = -tau_tol * abs(min(data['tau_spectrum']['1720']))

	tau_rms_1612 = data['tau_rms']['1612']
	tau_rms_1665 = data['tau_rms']['1665']
	tau_rms_1667 = data['tau_rms']['1667']
	tau_rms_1720 = data['tau_rms']['1720']

	if tau_1612 < min_tau_1612 or tau_1665 < min_tau_1665 or tau_1667 < min_tau_1667 or tau_1720 < min_tau_1720 or tau_1612 > max_tau_1612 or tau_1665 > max_tau_1665 or tau_1667 > max_tau_1667 or tau_1720 > max_tau_1720:
		ln_Pr_prior_tau = -np.inf
	else:
		# tau sum rule
		tau_sum = tau_1665/5. + tau_1667/9. - tau_1612 - tau_1720
		tau_sigma = sum([tau_rms_1665/5., tau_rms_1667/9., tau_rms_1612, tau_rms_1720])
		Pr_prior_tau = Gaussian(mean = 0, sigma = tau_sigma, amp = 1)(np.array([tau_sum]))[0]
		ln_Pr_prior_tau = np.log(Pr_prior_tau)
	
	return ln_Pr_prior_tau
def PrPriorTexp(Texps = None, data = None, report = False):
	[Texp_1612, Texp_1665, Texp_1667, Texp_1720] = Texps

	max_Texp_1612 = 5. * abs(max(data['Texp_spectrum']['1612']))
	max_Texp_1665 = 5. * abs(max(data['Texp_spectrum']['1665']))
	max_Texp_1667 = 5. * abs(max(data['Texp_spectrum']['1667']))
	max_Texp_1720 = 5. * abs(max(data['Texp_spectrum']['1720']))

	min_Texp_1612 = -5. * abs(min(data['Texp_spectrum']['1612']))
	min_Texp_1665 = -5. * abs(min(data['Texp_spectrum']['1665']))
	min_Texp_1667 = -5. * abs(min(data['Texp_spectrum']['1667']))
	min_Texp_1720 = -5. * abs(min(data['Texp_spectrum']['1720']))

	if Texp_1612 < min_Texp_1612 or Texp_1665 < min_Texp_1665 or Texp_1667 < min_Texp_1667 or Texp_1720 < min_Texp_1720 or Texp_1612 > max_Texp_1612 or Texp_1665 > max_Texp_1665 or Texp_1667 > max_Texp_1667 or Texp_1720 > max_Texp_1720:
		ln_Pr_prior_Texp = -np.inf
	else:
		ln_Pr_prior_Texp = -np.log((max_Texp_1612 - min_Texp_1612) * (max_Texp_1665 - min_Texp_1665) * (max_Texp_1667 - min_Texp_1667) * (max_Texp_1720 - min_Texp_1720))

	return ln_Pr_prior_Texp
def PrPriorTex(Texps = None, taus = None, data = None, report = False):
	[Texp_1612, Texp_1665, Texp_1667, Texp_1720] = Texps
	[tau_1612, tau_1665, tau_1667, tau_1720] = taus

	Texp_rms_1612 = data['Texp_rms']['1612']
	Texp_rms_1665 = data['Texp_rms']['1665']
	Texp_rms_1667 = data['Texp_rms']['1667']
	Texp_rms_1720 = data['Texp_rms']['1720']

	tau_rms_1612 = data['tau_rms']['1612']
	tau_rms_1665 = data['tau_rms']['1665']
	tau_rms_1667 = data['tau_rms']['1667']
	tau_rms_1720 = data['tau_rms']['1720']

	Tbg_1612 = data['Tbg']['1612']
	Tbg_1665 = data['Tbg']['1665']
	Tbg_1667 = data['Tbg']['1667']
	Tbg_1720 = data['Tbg']['1720']

	Tex_1612 = Tex(tau = tau_1612, Tbg = Tbg_1612, Texp = Texp_1612)
	Tex_1665 = Tex(tau = tau_1665, Tbg = Tbg_1665, Texp = Texp_1665)
	Tex_1667 = Tex(tau = tau_1667, Tbg = Tbg_1667, Texp = Texp_1667)
	Tex_1720 = Tex(tau = tau_1720, Tbg = Tbg_1720, Texp = Texp_1720)

	Tex_1612_error = abs(Tex_1612 * (Texp_rms_1612 / Texp_1612 + tau_rms_1612 / tau_1612))
	Tex_1665_error = abs(Tex_1665 * (Texp_rms_1665 / Texp_1665 + tau_rms_1665 / tau_1665))
	Tex_1667_error = abs(Tex_1667 * (Texp_rms_1667 / Texp_1667 + tau_rms_1667 / tau_1667))
	Tex_1720_error = abs(Tex_1720 * (Texp_rms_1720 / Texp_1720 + tau_rms_1720 / tau_1720))
	
	Tex_sum = 1.612231 / Tex_1612 - 1.665402 / Tex_1665 - 1.667359 / Tex_1667 + 1.72053 / Tex_1720
	Tex_sigma = abs((	1.612231 * Tex_1612_error / abs(Tex_1612) + 
				1.665402 * Tex_1665_error / abs(Tex_1665) + 
				1.667359 * Tex_1667_error / abs(Tex_1667) + 
				1.720530 * Tex_1720_error / abs(Tex_1720)) * math.sqrt(sum([Tex_1612**2., Tex_1665**2., Tex_1667**2., Tex_1720**2.]) / 4.))
	
	Pr_prior_Tex = Gaussian(mean = 0, sigma = Tex_sigma, amp = 1)(np.array([Tex_sum]))[0]
	return np.log(Pr_prior_Tex)
def PrPrior(x = None, data = None, min_vel = None, max_vel = None, report = False, fwhm_soft_lim = 5., fwhm_hard_lim = 7., dv = None, tau_tol = None):

	if data['Texp_spectrum']['1665'] != []:
		model_dim = 10
	else:
		model_dim = 6

	# initialise prior
	ln_Pr_prior = 0 # I'm adding to this to multiply probabilities. It should start at 0 (Pr = 1)

	##################
	#                #
	# Velocity Prior #
	#                #
	##################

	ln_Pr_prior += PrPriorVel(x = x, model_dim = model_dim, data = data, min_vel = min_vel, max_vel = max_vel)

	##############
	#            #
	# FWHM Prior #
	#            #
	##############

	for comp in range(int(len(x) / model_dim)):
		
		fwhm = x[comp * model_dim + 1]
		tau_1612 = x[comp * model_dim + 2]
		tau_1665 = x[comp * model_dim + 3]
		tau_1667 = x[comp * model_dim + 4]
		tau_1720 = x[comp * model_dim + 5]
		if data['Texp_spectrum']['1665'] != []:
			Texp_1612 = x[comp * model_dim + 6]
			Texp_1665 = x[comp * model_dim + 7]
			Texp_1667 = x[comp * model_dim + 8]
			Texp_1720 = x[comp * model_dim + 9]

		ln_Pr_prior += PrPriorFWHM(fwhm = fwhm, dv = dv)

		#############
		#           #
		# Tau Prior #
		#           #
		#############

		ln_Pr_prior += PrPriorTau(taus = [tau_1612, tau_1665, tau_1667, tau_1720], data = data, tau_tol = tau_tol)

		##############
		#            #
		# Texp Prior #
		#            #
		##############

		if data['Texp_spectrum']['1665'] != []:
			ln_Pr_prior += PrPriorTexp(Texps = [Texp_1612, Texp_1665, Texp_1667, Texp_1720], data = data)

			#############
			#           #
			# Tex Prior #
			#           #
			#############

			ln_Pr_prior += PrPriorTex(Texps = [Texp_1612, Texp_1665, Texp_1667, Texp_1720], taus = [tau_1612, tau_1665, tau_1667, tau_1720], data = data)

	return ln_Pr_prior
def BestParams(chain = None, lnprob = None):
	num_steps = len(chain)
	num_param = len(chain[0])

	final_array = [list(reversed(sorted(lnprob)))]
	final_darray = [list(reversed(sorted(lnprob)))]


	for param in range(num_param):
		param_chain = [chain[x][param] for x in range(num_steps)]
		final_array = np.concatenate((final_array, [[x for _,x in list(reversed(sorted(zip(lnprob, param_chain))))]]))
		zipped = sorted(zip(param_chain, lnprob))
		sorted_param_chain, sorted_lnprob = zip(*zipped)

		dparam_chain = [0] + [sorted_param_chain[x] - sorted_param_chain[x-1] for x in range(1, len(sorted_param_chain))]
		sorted_dparam_chain = [[x for _,x in list(reversed(sorted(zip(sorted_lnprob, dparam_chain))))]]
		final_darray = np.concatenate((final_darray, sorted_dparam_chain), axis = 0)

	contributions_to_evidence = np.zeros(num_steps)

	for step in range(num_steps):
		for param in range(num_param):
			contributions_to_evidence[step] += final_darray[param][step]

	accumulated_evidence = np.zeros(num_steps)
	step = 0

	for cont in contributions_to_evidence:
		if step != 0:
			accumulated_evidence[step] = np.logaddexp(accumulated_evidence[step - 1], cont)
		else:
			accumulated_evidence[step] = cont
		step += 1

	total_evidence = accumulated_evidence[-1]
	evidence_68 = total_evidence + np.log(0.6825)

	sigma_index = np.argmin(abs(accumulated_evidence - evidence_68))

	results = np.zeros([num_param, 3])

	for param in range(num_param):
		results[param][0] = min(final_array[param + 1][:sigma_index + 1])
		results[param][1] = np.median(final_array[param + 1][:sigma_index + 1])
		results[param][2] = max(final_array[param + 1][:sigma_index + 1])

	return (results, total_evidence)
def nullEvidence(data = None):
	'''
	Calculates the evidence of the null model. This is the special case where the model is a horizontal line with one parameter: y intercept. This gives a total of 8 parameters.
	The evidence is found by sampling the posterior distribution, and integrating that distribution.	
	'''
	def lnprob_null(x, data):
		# print('null\t' + str(x))
		if any(np.isnan(x)) or any(np.isinf(x)) or any(abs(np.array(x)) > 10.):
			return -np.inf
		else:
			tau_1612 = data['tau_spectrum']['1612']
			tau_1665 = data['tau_spectrum']['1665']
			tau_1667 = data['tau_spectrum']['1667']
			tau_1720 = data['tau_spectrum']['1720']

			rms_tau_1612 = data['tau_rms']['1612']
			rms_tau_1665 = data['tau_rms']['1665']
			rms_tau_1667 = data['tau_rms']['1667']
			rms_tau_1720 = data['tau_rms']['1720']

			tau_model_1612 = x[0] * np.ones(len(tau_1612))
			tau_model_1665 = x[1] * np.ones(len(tau_1665))
			tau_model_1667 = x[2] * np.ones(len(tau_1667))
			tau_model_1720 = x[3] * np.ones(len(tau_1720))	

			N_1612 = len(tau_1612)
			N_1665 = len(tau_1665)
			N_1667 = len(tau_1667)
			N_1720 = len(tau_1720)	

			llh_tau_1612 = -sum((tau_1612 - tau_model_1612)**2) / (2 * rms_tau_1612**2) + N_1612 * np.log(1 / (rms_tau_1612 * math.sqrt(2 * math.pi)))
			llh_tau_1665 = -sum((tau_1665 - tau_model_1665)**2) / (2 * rms_tau_1665**2) + N_1665 * np.log(1 / (rms_tau_1665 * math.sqrt(2 * math.pi)))
			llh_tau_1667 = -sum((tau_1667 - tau_model_1667)**2) / (2 * rms_tau_1667**2) + N_1667 * np.log(1 / (rms_tau_1667 * math.sqrt(2 * math.pi)))
			llh_tau_1720 = -sum((tau_1720 - tau_model_1720)**2) / (2 * rms_tau_1720**2) + N_1720 * np.log(1 / (rms_tau_1720 * math.sqrt(2 * math.pi)))
			
			if data['Texp_spectrum']['1665'] != []:

				Texp_1612 = data['Texp_spectrum']['1612']
				Texp_1665 = data['Texp_spectrum']['1665']
				Texp_1667 = data['Texp_spectrum']['1667']
				Texp_1720 = data['Texp_spectrum']['1720']

				rms_Texp_1612 = data['Texp_rms']['1612']
				rms_Texp_1665 = data['Texp_rms']['1665']
				rms_Texp_1667 = data['Texp_rms']['1667']
				rms_Texp_1720 = data['Texp_rms']['1720']

				Texp_model_1612 = x[4] * np.ones(len(Texp_1612))
				Texp_model_1665 = x[5] * np.ones(len(Texp_1665))
				Texp_model_1667 = x[6] * np.ones(len(Texp_1667))
				Texp_model_1720 = x[7] * np.ones(len(Texp_1720))

				llh_Texp_1612 = -sum((Texp_1612 - Texp_model_1612)**2) / (2 * rms_Texp_1612**2) + N_1612 * np.log(1 / (rms_Texp_1612 * math.sqrt(2 * math.pi)))
				llh_Texp_1665 = -sum((Texp_1665 - Texp_model_1665)**2) / (2 * rms_Texp_1665**2) + N_1665 * np.log(1 / (rms_Texp_1665 * math.sqrt(2 * math.pi)))
				llh_Texp_1667 = -sum((Texp_1667 - Texp_model_1667)**2) / (2 * rms_Texp_1667**2) + N_1667 * np.log(1 / (rms_Texp_1667 * math.sqrt(2 * math.pi)))
				llh_Texp_1720 = -sum((Texp_1720 - Texp_model_1720)**2) / (2 * rms_Texp_1720**2) + N_1720 * np.log(1 / (rms_Texp_1720 * math.sqrt(2 * math.pi)))
				
				ln_llh = llh_tau_1612 + llh_tau_1665 + llh_tau_1667 + llh_tau_1720 + llh_Texp_1612 + llh_Texp_1665 + llh_Texp_1667 + llh_Texp_1720
			else:
				ln_llh = llh_tau_1612 + llh_tau_1665 + llh_tau_1667 + llh_tau_1720
			return ln_llh

	nwalkers = 200
	if data['Texp_spectrum']['1665'] != []:
		ndim = 8
	else:
		ndim = 4
	burn_iterations = 1000
	iterations = 100

	# defining initial positions
	if data['Texp_spectrum']['1665'] != []:
		p0 = [1., 1., 1., 1., 0., 0., 0., 0.]
	else:
		p0 = [1., 1., 1., 1.]
	# Then add a random offset for each walker
	p0 = [[np.random.randn() / 100 + y for y in p0] for x in range(nwalkers)]
	# initiating and running mcmc
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_null, args = [data])
	pos, prob, state = sampler.run_mcmc(p0, burn_iterations)

	# run final time
	sampler.reset()
	sampler.run_mcmc(pos, iterations)

	# Find evidence from posterior
	post_dist = sampler.flatlnprobability
	flat_chain = sampler.flatchain
	(_, log_evidence) = BestParams(lnprob = post_dist, chain = flat_chain)
	return log_evidence
def Integral(x_axis, y_axis):
	'''
	It's assumed that the y_axis is logarithmic while the x_axis is not.
	'''
	# sort axes by x_axis
	y_axis = [x for _,x in sorted(zip(x_axis, y_axis))]
	x_axis = sorted(x_axis)	

	x_axis = [x_axis[x] for x in range(len(x_axis)) if np.isinf(y_axis[x]) == False]
	y_axis = [y_axis[x] for x in range(len(y_axis)) if np.isinf(y_axis[x]) == False]

	log_integral = -np.inf

	prev_x = x_axis[0]
	prev_y = y_axis[0]
	for a in range(1,len(x_axis)):
		current_x = x_axis[a]
		current_y = y_axis[a]
		
		piece_integral = np.logaddexp(current_y, prev_y) + np.log((current_x - prev_x) / 2.)
		log_integral = np.logaddexp(log_integral, piece_integral)
		prev_x = current_x
		prev_y = current_y
	return log_integral
def ConvergenceTest(data = None, sampler_chain = None, num_gauss = None, test_limit = 15):
	'''
	Tests if the variance across chains is comparable to the variance within the chains.

	Returns 'Pass' or 'Fail'
	'''
	if data['Texp_spectrum']['1665'] != []:
		model_dim = 10
	else:
		model_dim = 6
	orig_num_walkers = sampler_chain.shape[0]
	counter = 0

	# remove dead walkers
	for walker in reversed(range(sampler_chain.shape[0])):
		if sampler_chain[walker,0,0] == sampler_chain[walker,-1,0]: # vel doesn't change
			sampler_chain = np.delete(sampler_chain, walker, 0)
			counter += 1
	# replace removed walkers
	if counter > 0 and counter < orig_num_walkers / 2:
		for x in range(counter):
			sampler_chain = np.concatenate((sampler_chain, [sampler_chain[0]]), axis = 0)
	elif counter > orig_num_walkers / 2:
		return ('Fail', None)

	# test convergence in velocity
	for component in range(num_gauss):

		var_within_chains = np.median([np.var(sampler_chain[x,-100:-1,component * model_dim]) for x in range(sampler_chain.shape[0])])
		var_across_chains = np.median([np.var(sampler_chain[:,-x - 1,component * model_dim]) for x in range(100)])
		ratio = max([var_within_chains, var_across_chains]) / min([var_within_chains, var_across_chains])
		max_var = max([var_within_chains, var_across_chains])

		if ratio > test_limit and max_var < 1.:
			return ('Fail', sampler_chain[:,-1,:])

	return ('Pass', sampler_chain[:,-1,:])

##################
# Return Results #
##################

def PlotFinalModel(final_parameters = None, data = None, file_preamble = None):
	'''
	final_parameters is an Xx3 array, axis 1 has the 0.16, 0.50 and 0.84 quantiles from the posterior distribution, which approximate +/- 1 sigma
	'''

	parameters_16 = [x[0] for x in final_parameters]
	parameters_50 = [x[1] for x in final_parameters]
	parameters_84 = [x[2] for x in final_parameters]

	source_name = data['source_name']

	vel_1612 = data['vel_axis']['1612']
	vel_1665 = data['vel_axis']['1665']
	vel_1667 = data['vel_axis']['1667']
	vel_1720 = data['vel_axis']['1720']

	tau_1612 = data['tau_spectrum']['1612']
	tau_1665 = data['tau_spectrum']['1665']
	tau_1667 = data['tau_spectrum']['1667']
	tau_1720 = data['tau_spectrum']['1720']
	
	if data['Texp_spectrum']['1665'] != []:
		Texp_1612 = data['Texp_spectrum']['1612']
		Texp_1665 = data['Texp_spectrum']['1665']
		Texp_1667 = data['Texp_spectrum']['1667']
		Texp_1720 = data['Texp_spectrum']['1720']

		
		(tau_model_1612_min, tau_model_1665_min, tau_model_1667_min, tau_model_1720_min, Texp_model_1612_min, Texp_model_1665_min, Texp_model_1667_min, Texp_model_1720_min) = MakeModel(p = parameters_16, data = data)
		(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720, Texp_model_1612, Texp_model_1665, Texp_model_1667, Texp_model_1720) = MakeModel(p = parameters_50, data = data)
		(tau_model_1612_max, tau_model_1665_max, tau_model_1667_max, tau_model_1720_max, Texp_model_1612_max, Texp_model_1665_max, Texp_model_1667_max, Texp_model_1720_max) = MakeModel(p = parameters_84, data = data)

		fig, axes = plt.subplots(nrows = 5, ncols = 2, sharex = True)
		# tau
		axes[0,0].plot(vel_1612, tau_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[0,0].plot(vel_1612, tau_model_1612, color = 'black', linewidth = 1)
		axes[0,0].fill_between(vel_1612, tau_model_1612_min, tau_model_1612_max, color='0.7', zorder=-1)
		axes[1,0].plot(vel_1665, tau_1665, color = 'green', label = '1665 MHz', linewidth = 1)
		axes[1,0].plot(vel_1665, tau_model_1665, color = 'black', linewidth = 1)
		axes[1,0].fill_between(vel_1665, tau_model_1665_min, tau_model_1665_max, color='0.7', zorder=-1)
		axes[2,0].plot(vel_1667, tau_1667, color = 'red', label = '1667 MHz', linewidth = 1)
		axes[2,0].plot(vel_1667, tau_model_1667, color = 'black', linewidth = 1)
		axes[2,0].fill_between(vel_1667, tau_model_1667_min, tau_model_1667_max, color='0.7', zorder=-1)
		axes[3,0].plot(vel_1720, tau_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[3,0].plot(vel_1720, tau_model_1720, color = 'black', linewidth = 1)
		axes[3,0].fill_between(vel_1720, tau_model_1720_min, tau_model_1720_max, color='0.7', zorder=-1)
		# tau residuals
		axes[4,0].plot(vel_1612, tau_1612 - tau_model_1612, color = 'blue', linewidth = 1)
		axes[4,0].plot(vel_1665, tau_1665 - tau_model_1665, color = 'green', linewidth = 1)
		axes[4,0].plot(vel_1667, tau_1667 - tau_model_1667, color = 'red', linewidth = 1)
		axes[4,0].plot(vel_1720, tau_1720 - tau_model_1720, color = 'cyan', linewidth = 1)
		# Texp
		axes[0,1].plot(vel_1612, Texp_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[0,1].plot(vel_1612, Texp_model_1612, color = 'black', linewidth = 1)
		axes[0,1].fill_between(vel_1612, Texp_model_1612_min, Texp_model_1612_max, color='0.7', zorder=-1)
		axes[1,1].plot(vel_1665, Texp_1665, color = 'green', label = '1665 MHz', linewidth = 1)
		axes[1,1].plot(vel_1665, Texp_model_1665, color = 'black', linewidth = 1)
		axes[1,1].fill_between(vel_1665, Texp_model_1665_min, Texp_model_1665_max, color='0.7', zorder=-1)
		axes[2,1].plot(vel_1667, Texp_1667, color = 'red', label = '1667 MHz', linewidth = 1)
		axes[2,1].plot(vel_1667, Texp_model_1667, color = 'black', linewidth = 1)
		axes[2,1].fill_between(vel_1667, Texp_model_1667_min, Texp_model_1667_max, color='0.7', zorder=-1)
		axes[3,1].plot(vel_1720, Texp_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[3,1].plot(vel_1720, Texp_model_1720, color = 'black', linewidth = 1)
		axes[3,1].fill_between(vel_1720, Texp_model_1720_min, Texp_model_1720_max, color='0.7', zorder=-1)
		# Texp residuals
		axes[4,1].plot(vel_1612, Texp_1612 - Texp_model_1612, color = 'blue', linewidth = 1)
		axes[4,1].plot(vel_1665, Texp_1665 - Texp_model_1665, color = 'green', linewidth = 1)
		axes[4,1].plot(vel_1667, Texp_1667 - Texp_model_1667, color = 'red', linewidth = 1)
		axes[4,1].plot(vel_1720, Texp_1720 - Texp_model_1720, color = 'cyan', linewidth = 1)

		for row in range(5):
			for col in range(2):
				if any([axes[row, col].get_yticks()[::2][x] == 0. for x in axes[row, col].get_yticks()[::2]]):
					axes[row, col].set_yticks(axes[row, col].get_yticks()[::2])
				else:
					axes[row, col].set_yticks(axes[row, col].get_yticks()[1::2])

	else:
		(tau_model_1612_min, tau_model_1665_min, tau_model_1667_min, tau_model_1720_min) = MakeModel(p = parameters_16, data = data)
		(tau_model_1612, tau_model_1665, tau_model_1667, tau_model_1720) = MakeModel(p = parameters_50, data = data)
		(tau_model_1612_max, tau_model_1665_max, tau_model_1667_max, tau_model_1720_max) = MakeModel(p = parameters_84, data = data)
		fig, axes = plt.subplots(nrows = 5, ncols = 1, sharex = True)
		# tau
		axes[0].plot(vel_1612, tau_1612, color = 'blue', label = '1612 MHz', linewidth = 1)
		axes[0].plot(vel_1612, tau_model_1612, color = 'black', linewidth = 1)
		axes[0].fill_between(vel_1612, tau_model_1612_min, tau_model_1612_max, color='0.7', zorder=-1)
		axes[1].plot(vel_1665, tau_1665, color = 'green', label = '1665 MHz', linewidth = 1)
		axes[1].plot(vel_1665, tau_model_1665, color = 'black', linewidth = 1)
		axes[1].fill_between(vel_1665, tau_model_1665_min, tau_model_1665_max, color='0.7', zorder=-1)
		axes[2].plot(vel_1667, tau_1667, color = 'red', label = '1667 MHz', linewidth = 1)
		axes[2].plot(vel_1667, tau_model_1667, color = 'black', linewidth = 1)
		axes[2].fill_between(vel_1667, tau_model_1667_min, tau_model_1667_max, color='0.7', zorder=-1)
		axes[3].plot(vel_1720, tau_1720, color = 'cyan', label = '1720 MHz', linewidth = 1)
		axes[3].plot(vel_1720, tau_model_1720, color = 'black', linewidth = 1)
		axes[3].fill_between(vel_1720, tau_model_1720_min, tau_model_1720_max, color='0.7', zorder=-1)
		# tau residuals
		axes[4].plot(vel_1612, tau_1612 - tau_model_1612, color = 'blue', linewidth = 1)
		axes[4].plot(vel_1665, tau_1665 - tau_model_1665, color = 'green', linewidth = 1)
		axes[4].plot(vel_1667, tau_1667 - tau_model_1667, color = 'red', linewidth = 1)
		axes[4].plot(vel_1720, tau_1720 - tau_model_1720, color = 'cyan', linewidth = 1)
		# labels
		axes[4].set_xlabel('Velocity (km/s)')
		axes[2].set_ylabel('Optical Depth', labelpad = 15)
		axes[0].set_title(source_name)
		axes[0].text(0.01, 0.75, '1612 MHz', transform=axes[0].transAxes)
		axes[1].text(0.01, 0.75, '1665 MHz', transform=axes[1].transAxes)
		axes[2].text(0.01, 0.75, '1667 MHz', transform=axes[2].transAxes)
		axes[3].text(0.01, 0.75, '1720 MHz', transform=axes[3].transAxes)
		axes[4].text(0.01, 0.75, 'Residuals', transform=axes[4].transAxes)
		
		for row in range(5):
			if any([axes[row].get_yticks()[::2][x] == 0. for x in axes[row].get_yticks()[::2]]):
				axes[row].set_yticks(axes[row].get_yticks()[::2])
			else:
				axes[row].set_yticks(axes[row].get_yticks()[1::2])

	
	plt.savefig('BGD_Output/Final_Models/' + str(file_preamble) + '_' + source_name + '_Final_model.pdf')
	# plt.show()
	plt.close()
def ResultsReport(final_parameters = None, final_median_parameters = None, data = None, file_preamble = None):
	'''
	Generates a nice report of results
	'''

	short_source_name_dict = {'g003.74+0.64.':'g003', 
			'g006.32+1.97.':'g006', 
			'g007.47+0.06.':'g007', 
			'g334.72-0.65.':'g334', 
			'g336.49-1.48.':'g336', 
			'g340.79-1.02a.':'g340a', 
			'g340.79-1.02b.':'g340b', 
			'g344.43+0.05.':'g344', 
			'g346.52+0.08.':'g346', 
			'g347.75-1.14.':'g347', 
			'g348.44+2.08.':'g348', 
			'g349.73+1.67.':'g349', 
			'g350.50+0.96.':'g350', 
			'g351.61+0.17a.':'g351a', 
			'g351.61+0.17b.':'g351b', 
			'g353.411-0.3.':'g353', 
			'g356.91+0.08.':'g356'}

	try:
		os.makedir('pickles')
	except:
		pass

	pickle.dump(final_parameters, open('pickles/RESULTS_' + str(file_preamble) + '_' + short_source_name_dict[data['source_name']] + '.pickle', 'w'))
	
	PlotFinalModel(final_parameters = final_parameters, data = data, file_preamble = file_preamble)
	print('\nResults for ' + data['source_name'])
	
	if data['Texp_spectrum']['1665'] != []:
		print('\tNumber of features identified: ' + str(len(final_parameters) / 10) + '\n')
	else:
		print('\tNumber of features identified: ' + str(len(final_parameters) / 6) + '\n'	)

	if len(final_parameters) > 5:

		print('\tBackground Temperatures [1612, 1665, 1667, 1720MHz] = ' + str([data['Tbg']['1612'], data['Tbg']['1665'], data['Tbg']['1667'], data['Tbg']['1720']]))
		final_parameters = final_parameters
		print('\tFeature Parameters [16th, 50th, 84th quantiles]:')
		
		if data['Texp_spectrum']['1665'] != []:
			for feature in range(int(len(final_parameters) / 10)):
				print('\t\tfeature number ' + str(feature + 1) + ':')
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84], [Texp_1612_16, Texp_1612_50, Texp_1612_84], [Texp_1665_16, Texp_1665_50, Texp_1665_84], [Texp_1667_16, Texp_1667_50, Texp_1667_84], [Texp_1720_16, Texp_1720_50, Texp_1720_84]] = final_parameters[feature * 10:feature * 10 + 10]

				print('\t\t\tcentroid velocity = ' + str([vel_16, vel_50, vel_84]) + ' km/sec')
				print('\t\t\tfwhm = ' + str([fwhm_16, fwhm_50, fwhm_84]) + ' km/sec')
				print('\n\t\t\t1612MHz peak tau = ' + str([tau_1612_16, tau_1612_50, tau_1612_84]))
				print('\t\t\t1665MHz peak tau = ' + str([tau_1665_16, tau_1665_50, tau_1665_84]))
				print('\t\t\t1667MHz peak tau = ' + str([tau_1667_16, tau_1667_50, tau_1667_84]))
				print('\t\t\t1720MHz peak tau = ' + str([tau_1720_16, tau_1720_50, tau_1720_84]))
				print('\n\t\t\t1612MHz Texp = ' + str([Texp_1612_16, Texp_1612_50, Texp_1612_84]) + ' K')
				print('\t\t\t1665MHz Texp = ' + str([Texp_1665_16, Texp_1665_50, Texp_1665_84]) + ' K')
				print('\t\t\t1667MHz Texp = ' + str([Texp_1667_16, Texp_1667_50, Texp_1667_84]) + ' K')
				print('\t\t\t1720MHz Texp = ' + str([Texp_1720_16, Texp_1720_50, Texp_1720_84]) + ' K')

		else:
			for feature in range(int(len(final_parameters) / 6)):
				print('\t\tfeature number ' + str(feature + 1) + ':')
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = final_parameters[feature * 6:feature * 6 + 6]

				print('\t\t\tcentroid velocity = ' + str([vel_16, vel_50, vel_84]) + ' km/sec')
				print('\t\t\tfwhm = ' + str([fwhm_16, fwhm_50, fwhm_84]) + ' km/sec')
				print('\n\t\t\t1612MHz peak tau = ' + str([tau_1612_16, tau_1612_50, tau_1612_84]))
				print('\t\t\t1665MHz peak tau = ' + str([tau_1665_16, tau_1665_50, tau_1665_84]))
				print('\t\t\t1667MHz peak tau = ' + str([tau_1667_16, tau_1667_50, tau_1667_84]))
				print('\t\t\t1720MHz peak tau = ' + str([tau_1720_16, tau_1720_50, tau_1720_84]))
def ResultsTable(final_parameters = None, final_median_parameters = None, data = None):
	'''
	make a latex table
	'''

	if len(final_parameters) > 5:

		final_parameters = final_parameters
		
		if data['Texp_spectrum']['1665'] != []:
			for feature in range(int(len(final_parameters) / 10)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84], [Texp_1612_16, Texp_1612_50, Texp_1612_84], [Texp_1665_16, Texp_1665_50, Texp_1665_84], [Texp_1667_16, Texp_1667_50, Texp_1667_84], [Texp_1720_16, Texp_1720_50, Texp_1720_84]] = final_parameters[feature * 10:feature * 10 + 10]
				[vel_16, vel_50, vel_84] = [round(x, 1) for x in [vel_16, vel_50, vel_84]]
				[fwhm_16, fwhm_50, fwhm_84] = [round(x, 2) for x in [fwhm_16, fwhm_50, fwhm_84]]
				[tau_1612_16, tau_1612_50, tau_1612_84, tau_1665_16, tau_1665_50, tau_1665_84, tau_1667_16, tau_1667_50, tau_1667_84, tau_1720_16, tau_1720_50, tau_1720_84, Texp_1612_16, Texp_1612_50, Texp_1612_84, Texp_1665_16, Texp_1665_50, Texp_1665_84, Texp_1667_16, Texp_1667_50, Texp_1667_84, Texp_1720_16, Texp_1720_50, Texp_1720_84] = [round(x, 3) for x in [tau_1612_16, tau_1612_50, tau_1612_84, tau_1665_16, tau_1665_50, tau_1665_84, tau_1667_16, tau_1667_50, tau_1667_84, tau_1720_16, tau_1720_50, tau_1720_84, Texp_1612_16, Texp_1612_50, Texp_1612_84, Texp_1665_16, Texp_1665_50, Texp_1665_84, Texp_1667_16, Texp_1667_50, Texp_1667_84, Texp_1720_16, Texp_1720_50, Texp_1720_84]]
				print(data['source_name'] + '&' + str(vel_50) + '$^{+' + str(vel_84 - vel_50) + '}_{-' + str(vel_50 - vel_16) + '}$' + '&' + str(fwhm_50) + '$^{+' + str(fwhm_84 - fwhm_50) + '}_{-' + str(fwhm_50 - fwhm_16) + '}$' + '&' + str(tau_1612_50) + '$^{+' + str(tau_1612_84 - tau_1612_50) + '}_{-' + str(tau_1612_50 - tau_1612_16) + '}$' + '&' + str(tau_1665_50) + '$^{+' + str(tau_1665_84 - tau_1665_50) + '}_{-' + str(tau_1665_50 - tau_1665_16) + '}$' + '&' + str(tau_1667_50) + '$^{+' + str(tau_1667_84 - tau_1667_50) + '}_{-' + str(tau_1667_50 - tau_1667_16) + '}$' + '&' + str(tau_1720_50) + '$^{+' + str(tau_1720_84 - tau_1720_50) + '}_{-' + str(tau_1720_50 - tau_1720_16) + '}$' + '&' + str(Texp_1612_50) + '$^{+' + str(Texp_1612_84 - Texp_1612_50) + '}_{-' + str(Texp_1612_50 - Texp_1612_16) + '}$' + '&' + str(Texp_1665_50) + '$^{+' + str(Texp_1665_84 - Texp_1665_50) + '}_{-' + str(Texp_1665_50 - Texp_1665_16) + '}$' + '&' + str(Texp_1667_50) + '$^{+' + str(Texp_1667_84 - Texp_1667_50) + '}_{-' + str(Texp_1667_50 - Texp_1667_16) + '}$' + '&' + str(Texp_1720_50) + '$^{+' + str(Texp_1720_84 - Texp_1720_50) + '}_{-' + str(Texp_1720_50 - Texp_1720_16) + '}$' + '\\\\')

		else:
			for feature in range(int(len(final_parameters) / 6)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = final_parameters[feature * 6:feature * 6 + 6]
				[vel_16, vel_50, vel_84] = [round(x, 1) for x in [vel_16, vel_50, vel_84]]
				[fwhm_16, fwhm_50, fwhm_84] = [round(x, 2) for x in [fwhm_16, fwhm_50, fwhm_84]]
				[tau_1612_16, tau_1612_50, tau_1612_84, tau_1665_16, tau_1665_50, tau_1665_84, tau_1667_16, tau_1667_50, tau_1667_84, tau_1720_16, tau_1720_50, tau_1720_84] = [round(x, 3) for x in [tau_1612_16, tau_1612_50, tau_1612_84, tau_1665_16, tau_1665_50, tau_1665_84, tau_1667_16, tau_1667_50, tau_1667_84, tau_1720_16, tau_1720_50, tau_1720_84]]
				print(data['source_name'] + '&' + str(vel_50) + '$^{+' + str(vel_84 - vel_50) + '}_{-' + str(vel_50 - vel_16) + '}$' + '&' + str(fwhm_50) + '$^{+' + str(fwhm_84 - fwhm_50) + '}_{-' + str(fwhm_50 - fwhm_16) + '}$' + '&' + str(tau_1612_50) + '$^{+' + str(tau_1612_84 - tau_1612_50) + '}_{-' + str(tau_1612_50 - tau_1612_16) + '}$' + '&' + str(tau_1665_50) + '$^{+' + str(tau_1665_84 - tau_1665_50) + '}_{-' + str(tau_1665_50 - tau_1665_16) + '}$' + '&' + str(tau_1667_50) + '$^{+' + str(tau_1667_84 - tau_1667_50) + '}_{-' + str(tau_1667_50 - tau_1667_16) + '}$' + '&' + str(tau_1720_50) + '$^{+' + str(tau_1720_84 - tau_1720_50) + '}_{-' + str(tau_1720_50 - tau_1720_16) + '}$' + '\\\\')
def ResultsTableExcel(final_parameters = None, final_median_parameters = None, data = None):
	'''
	make an excel table
	'''

	if len(final_parameters) > 5:

		final_parameters = final_parameters
		
		if data['Texp_spectrum']['1665'] != []:
			for feature in range(int(len(final_parameters) / 10)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84], [Texp_1612_16, Texp_1612_50, Texp_1612_84], [Texp_1665_16, Texp_1665_50, Texp_1665_84], [Texp_1667_16, Texp_1667_50, Texp_1667_84], [Texp_1720_16, Texp_1720_50, Texp_1720_84]] = final_parameters[feature * 10:feature * 10 + 10]
				print(data['source_name'] + ' \t ' + str(vel_50) + ' \t ' + str(vel_84 - vel_50) + ' \t ' + str(vel_50 - vel_16) + ' \t ' + str(fwhm_50) + ' \t ' + str(fwhm_84 - fwhm_50) + ' \t ' + str(fwhm_50 - fwhm_16) + ' \t ' + str(tau_1612_50) + ' \t ' + str(tau_1612_84 - tau_1612_50) + ' \t ' + str(tau_1612_50 - tau_1612_16) + ' \t ' + str(tau_1665_50) + ' \t ' + str(tau_1665_84 - tau_1665_50) + ' \t ' + str(tau_1665_50 - tau_1665_16) + ' \t ' + str(tau_1667_50) + ' \t ' + str(tau_1667_84 - tau_1667_50) + ' \t ' + str(tau_1667_50 - tau_1667_16) + ' \t ' + str(tau_1720_50) + ' \t ' + str(tau_1720_84 - tau_1720_50) + ' \t ' + str(tau_1720_50 - tau_1720_16) + ' \t ' + str(Texp_1612_50) + ' \t ' + str(Texp_1612_84 - Texp_1612_50) + ' \t ' + str(Texp_1612_50 - Texp_1612_16) + ' \t ' + str(Texp_1665_50) + ' \t ' + str(Texp_1665_84 - Texp_1665_50) + ' \t ' + str(Texp_1665_50 - Texp_1665_16) + ' \t ' + str(Texp_1667_50) + ' \t ' + str(Texp_1667_84 - Texp_1667_50) + ' \t ' + str(Texp_1667_50 - Texp_1667_16) + ' \t ' + str(Texp_1720_50) + ' \t ' + str(Texp_1720_84 - Texp_1720_50) + ' \t ' + str(Texp_1720_50 - Texp_1720_16))

		else:
			for feature in range(int(len(final_parameters) / 6)):
				[[vel_16, vel_50, vel_84], [fwhm_16, fwhm_50, fwhm_84], [tau_1612_16, tau_1612_50, tau_1612_84], [tau_1665_16, tau_1665_50, tau_1665_84], [tau_1667_16, tau_1667_50, tau_1667_84], [tau_1720_16, tau_1720_50, tau_1720_84]] = final_parameters[feature * 6:feature * 6 + 6]
				print(data['source_name'] + ' \t ' + str(vel_50) + ' \t ' + str(vel_84 - vel_50) + ' \t ' + str(vel_50 - vel_16) + ' \t ' + str(fwhm_50) + ' \t ' + str(fwhm_84 - fwhm_50) + ' \t ' + str(fwhm_50 - fwhm_16) + ' \t ' + str(tau_1612_50) + ' \t ' + str(tau_1612_84 - tau_1612_50) + ' \t ' + str(tau_1612_50 - tau_1612_16) + ' \t ' + str(tau_1665_50) + ' \t ' + str(tau_1665_84 - tau_1665_50) + ' \t ' + str(tau_1665_50 - tau_1665_16) + ' \t ' + str(tau_1667_50) + ' \t ' + str(tau_1667_84 - tau_1667_50) + ' \t ' + str(tau_1667_50 - tau_1667_16) + ' \t ' + str(tau_1720_50) + ' \t ' + str(tau_1720_84 - tau_1720_50) + ' \t ' + str(tau_1720_50 - tau_1720_16))

#############################
#                           #
#   M   M        i          #
#   MM MM   aa      n nn    #
#   M M M   aaa  i  nn  n   #
#   M   M  a  a  i  n   n   #
#   M   M   aaa  i  n   n   #
#                           #
#############################

def Main(source_name = None, vel_axes = None, tau_spectra = None, tau_rms = None, Texp_spectra = None, Texp_rms = None, Tbg = None, quiet = True, Bayes_threshold = 10., con_test_limit = 15, tau_tol = 5, max_cores = 10, test = False):
	'''
	Performs Bayesian Gaussian Decomposition on velocity spectra of the 2 Pi 3/2 J = 3/2 ground state 
	transitions of OH.

	Parameters:
	source_name - unique identifier for sightline, used in plots, dictionaries etc.
	vel_axes - list of velocity axes: [vel_axis_1612, vel_axis_1665, vel_axis_1667, vel_axis_1720]
	spectra - list of spectra (brightness temperature or tau): [spectrum_1612, spectrum_1665, spectrum_1667, 
			spectrum_1720]
	rms - list of estimates of rms error in spectra: [rms_1612, rms_1665, rms_1667, rms_1720]. Used by AGD
	expected_min_fwhm - estimate of the minimum full width at half-maximum of features expected in the data 
			in km/sec. Used when categorising features as isolated or blended.

	Returns parameters of Gaussian component(s): [vel_1, fwhm_1, height_1612_1, height_1665_1, height_1667_1, 
			height_1720_1, ..., _N] for N components
	'''
	data = {'source_name': source_name, 'vel_axis': {'1612': [], '1665': [], '1667': [], '1720': []}, 'tau_spectrum': {'1612': [], '1665': [], '1667': [], '1720': []}, 'tau_rms': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Texp_spectrum': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Texp_rms': {'1612': [], '1665': [], '1667': [], '1720': []}, 'Tbg': {'1612': [], '1665': [], '1667': [], '1720': []}}

		###############################################
		#                                             #
		#   Load Data into 'data' dictionary object   #
		#                                             #	
		###############################################

	data['vel_axis']['1612']		= vel_axes[0]
	data['tau_spectrum']['1612']	= tau_spectra[0]
	data['tau_rms']['1612']			= tau_rms[0]

	data['vel_axis']['1665']		= vel_axes[1]
	data['tau_spectrum']['1665']	= tau_spectra[1]
	data['tau_rms']['1665']			= tau_rms[1]

	data['vel_axis']['1667']		= vel_axes[2]
	data['tau_spectrum']['1667']	= tau_spectra[2]
	data['tau_rms']['1667']			= tau_rms[2]

	data['vel_axis']['1720']		= vel_axes[3]
	data['tau_spectrum']['1720']	= tau_spectra[3]
	data['tau_rms']['1720']			= tau_rms[3]

	dv = abs(((data['vel_axis']['1665'][1] - data['vel_axis']['1665'][0]) + (data['vel_axis']['1667'][1] - data['vel_axis']['1667'][0])) / 2)

	plot_num = 0
	
	if Texp_spectra != None: #absorption and emission spectra are available (i.e. Arecibo observations)
		
		data['Tbg']['1612']				= Tbg[0]
		data['Tbg']['1665']				= Tbg[1]
		data['Tbg']['1667']				= Tbg[2]
		data['Tbg']['1720']				= Tbg[3]

		data['Texp_rms']['1612']		= Texp_rms[0]
		data['Texp_rms']['1665']		= Texp_rms[1]
		data['Texp_rms']['1667']		= Texp_rms[2]
		data['Texp_rms']['1720']		= Texp_rms[3]

		data['Texp_spectrum']['1612']	= Texp_spectra[0]
		data['Texp_spectrum']['1665']	= Texp_spectra[1]
		data['Texp_spectrum']['1667']	= Texp_spectra[2]
		data['Texp_spectrum']['1720']	= Texp_spectra[3]


	###############################################
	#                                             #
	#              Identify Gaussians             #
	#                                             #
	###############################################

	sig_vel_ranges = FindGaussians(data, dv, tau_tol = tau_tol)

	###############################################
	#                                             #
	#                Fit Gaussians                #
	#                                             #
	###############################################


	if len(sig_vel_ranges) != 0:
		(final_parameters, final_median_parameters) = FitGaussians(data = data, sig_ranges = sig_vel_ranges, plot_num = plot_num, quiet = quiet, Bayes_threshold = Bayes_threshold, dv = dv, tau_tol = tau_tol, con_test_limit = con_test_limit)
		if test == False:
			try:
				ResultsReport(final_parameters = final_parameters, final_median_parameters = final_median_parameters, data = data)
			except:
				print(str(source_name) + '\t' + str(final_parameters))
		else:
			return final_parameters
	else:
		if test == False:
			try:
				ResultsReport(final_parameters = [None], final_median_parameters = [None], data = data)
			except:
				print(str(source_name) + '\tNo features identified')
		else:
			return None

	# short_source_name_dict = {'g003.74+0.64.':'g003', 
	# 		'g006.32+1.97.':'g006', 
	# 		'g007.47+0.06.':'g007', 
	# 		'g334.72-0.65.':'g334', 
	# 		'g336.49-1.48.':'g336', 
	# 		'g340.79-1.02a.':'g340a', 
	# 		'g340.79-1.02b.':'g340b', 
	# 		'g344.43+0.05.':'g344', 
	# 		'g346.52+0.08.':'g346', 
	# 		'g347.75-1.14.':'g347', 
	# 		'g348.44+2.08.':'g348', 
	# 		'g349.73+1.67.':'g349', 
	# 		'g350.50+0.96.':'g350', 
	# 		'g351.61+0.17a.':'g351a', 
	# 		'g351.61+0.17b.':'g351b', 
	# 		'g353.411-0.3.':'g353', 
	# 		'g356.91+0.08.':'g356'}

	
	# final_parameters = pickle.load(open('pickles/RESULTS_' + str(tau_tol) + '_' + str(con_test_limit) + '_' + short_source_name_dict[data['source_name']] + '.pickle', 'r'))
	# final_median_parameters = [x[1] for x in final_parameters]

	# ResultsTable(final_parameters = final_parameters, final_median_parameters = final_median_parameters, data = data)

	# ResultsTableExcel(final_parameters = final_parameters, final_median_parameters = final_median_parameters, data = data)
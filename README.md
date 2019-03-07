# BGD
Bayesian Gaussian decomposition algorithm

Written by Anita Petzler
Last modified 8 March 2019

Bayesian Gaussian Decomposer (BGD) uses a Bayesian approach to model evaluation to fit Gaussian curves 
to sets of spectra of the four 2 PI 3/2 J = 3/2 ground state transitions of OH. The current version does not 
allow for incomplete input spectra, so they must have the same velocity ranges. 
The code that eventually became BGD was begun as part of my Master's of Research degree at Macquarie University 
but is unpublished in its much more complete current form. If used, please cite my Master's thesis for now: 
  
  Petzler, A. (2018). Probing the Galactic ISM in OH Absorption (Master's thesis). Macquarie University, 
  Australia.


General advice:

See 'BGD_sample.py' for a sample implementation of BGD.

If you get the error 'OSError: [Errno 24] Too many open files', reset the number of 
allowed open files in the command line via:

$ ulimit -a

$ ulimit -Sn 10000


	


import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum

# import multiprocessing
# from multiprocessing import Pool#, Queue, Process
# from pathos.multiprocessing import ProcessingPool
# from memory_profiler import memory_usage
from abcBMCUtil import *

PATH = "../"
DEBUG = True
DEBUG = False
OPT = True
SC = 2
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
DIFF = 2 # function of depth, time, memory
# DIFF = 3 # function number of clauses/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
PDR = False
MAX_FRAME = 1e4
MAX_CLAUSE = 1e9
MAX_TIME = 3600
TIMEOUT = 3600
POOL = True

TIMEOUT = 3600
ST1 = 120
ST2 = 300

k =  4 # arms

Actions = ['bmc2', 'bmc3g', 'bmc3', 'bmc3s','bmc3r', 'bmcu', 'bmc3j','pdr']

def runseq(fn, seq):

	fname = os.path.join(PATH, fn)
	#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
	#ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	seq_list = ['({0}, {1})'.format(Actions[t[0]], t[1]) for t in seq]
	sequence = ';'.join(seq_list)
	print('##################################')
	print('---- Running sequence {0} ----'.format(sequence), fn, fname)
	# ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))	
	ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	#if DEBUG or self.n == 0:
	print(fname, ofname)

	# if self.n == 0:
	simplify(fname, ofname)
	print('Simplified model', ofname)

	#starting state
	sd = 0
	asrt = -1
	sm = None
	tt = 0
	pt = 0
	f = 0
	ar_tab = {}
	for item in seq:
		a, t = item
		if a == 0:    #ABC bmc2
			asrt, sm, ar_tab1, tt1 = bmc2(ofname, sd, t=t, f=f)
		elif a == 1: #ABC bmc3g
			asrt, sm, ar_tab1, tt1 = bmc3rg(ofname, sd,  t=t, f=f)
		elif a == 2: #ABC bmc3
			asrt, sm, ar_tab1, tt1 = bmc3(ofname, sd,  t=t, f=f)
		elif a == 3: #ABC bmc3rs
			asrt, sm, ar_tab1, tt1 = bmc3rs(ofname, sd,  t=t, f=f)
		elif a == 4: #ABC bmc3j
			asrt, sm, ar_tab1, tt1 = bmc3j(ofname, sd,  t=t, f=f)
		elif a == 5: #ABC bmc3u
			asrt, sm, ar_tab1, tt1 = bmc3ru(ofname, sd,  t=t, f=f)
		elif a == 6: #ABC bmc3r
			asrt, sm, ar_tab1, tt1 = bmc3r(ofname, sd,  t=t, f=f)
		elif a == 7: #ABC pdr
			asrt, sm, ar_tab1, tt1 = pdr(ofname, t)

		for ky in ar_tab1:
			ar_tab.update({ky: ar_tab1[ky]})

		if sm:
			sd = sm.frame+1 if sm.frame > 0 else sm.frame
			# if a == 0:
			# 	sd = sm.frame #if sm.frame > 0 else sm.frame
		else:
			sm =  abc_result(frame=max(0, sd-1), conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = tt1, ld = max(0, sd-1))
		print(sm)
		sys.stdout.flush()
		
		
		tt += sm.tt
		
		if asrt > 0:
			pt += sm.tt
			print('Output asserted at frame ', asrt, 'time', pt)
			print('Stopping iteration')
			
			break
		else:
			pt += t
		
	print('############### {0}: {1}, {2} ###################'.format( 'SAT' if asrt> 0 else 'TIMEOUT', sm.frame, pt if asrt > 0 else tt))
	return (asrt, 'SAT' if asrt> 0 else 'TIMEOUT', sm.frame, pt if asrt > 0 else tt, ar_tab)

def run(inputs):

	ofname, a, sd, t = inputs

	asrt = 0
	sm =  abc_result(frame=max(0, sd-1), conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = 0, ld = max(0, sd-1))
	ar_tab1 = {}
	f = 0

	if a == 0:    #ABC bmc2
		asrt, sm, ar_tab1, tt1 = bmc2(ofname, sd, t=t, f=f)
	elif a == 1: #ABC bmc3g
		asrt, sm, ar_tab1, tt1 = bmc3rg(ofname, sd,  t=t, f=f)
	elif a == 2: #ABC bmc3
		asrt, sm, ar_tab1, tt1 = bmc3(ofname, sd,  t=t, f=f)
	elif a == 3: #ABC bmc3rs
		asrt, sm, ar_tab1, tt1 = bmc3rs(ofname, sd,  t=t, f=f)
	elif a == 4: #ABC bmc3j
		asrt, sm, ar_tab1, tt1 = bmc3j(ofname, sd,  t=t, f=f)
	elif a == 5: #ABC bmc3u
		asrt, sm, ar_tab1, tt1 = bmc3ru(ofname, sd,  t=t, f=f)
	elif a == 6: #ABC bmc3r
		asrt, sm, ar_tab1, tt1 = bmc3r(ofname, sd,  t=t, f=f)
	elif a == 7: #ABC pdr
		asrt, sm, ar_tab1, tt1 = pdr(ofname, t)

	if sm is None:
		sm =  abc_result(frame=max(0, sd-1), conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = tt1, ld =max(0, sd-1))

	return asrt, sm, ar_tab1, a

def run_parallel(fn, seq, k):

	fname = os.path.join(PATH, fn)
	#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
	ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	# seq_list = ['({0}'.format(Actions[t[0]], t[1]) for t in seq]
	# sequence = ';'.join(seq_list)
	# print('##################################')
	# ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))
	#if DEBUG or self.n == 0:
	print(fname, ofname)

	# if self.n == 0:
	simplify(fname, ofname)
	print('Simplified model', ofname)
	print('---- Running sequence {0} ----'.format(seq), fn, fname)
	
	#starting state
	sd = 0
	asrt = -1
	sm = None
	tt = 0
	pt = 0

	# mcpu = multiprocessing.cpu_count()
	# np1 = int(mcpu)-2 # if mcpu > 30 else int(mcpu)-2
	num_proc = k #min(k, np1)
	ar_tab = {}
	sequence = []
	ind = 1
	for t in seq:
		# t = item
		# if POOL and num_proc > 1:
		# 	pool = ProcessingPool(num_proc)
		# 	inputs = [[ofname, a, sd, t] for a in range(num_proc)]
		# 	#print(inputs)
		# 	results = pool.map(run, inputs)
		# else:
		results = [run([ofname, a, sd, t]) for a in range(num_proc)]
		
		# if POOL:
		# 	pool.close()
		# 	pool.join()
		# 	pool.clear()

		max_act = results[0]
		for j in range(1, num_proc):   
			asrt, sm, ar_tab1, a = results[j]
			if sm and sm is not None:
				print('max: ', Actions[max_act[-1]], max_act[1].frame, 'current', Actions[a], sm.frame)
				if asrt > 0:
					if max_act[1].tt > sm.tt:
						max_act = asrt, sm, ar_tab1, a
				elif max_act[1].frame < sm.frame:
					max_act = asrt, sm, ar_tab1, a

		asrt, sm, ar_tab1, a = max_act
		for ky in ar_tab1:
			ar_tab.update({ky: ar_tab1[ky]})

		if sm:
			sd = sm.frame+1 if sm.frame > 0 else sm.frame
			# if a == 0:
			# 	sd = sm.frame #if sm.frame > 0 else sm.frame
		else:
			sm =  abc_result(frame=max(0, sd-1), conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = 0, ld = max(0, sd-1))
		print(ind, 'max_act', Actions[a], sm)
		print()
		sequence.append((Actions[a], t))
		sys.stdout.flush()
		ind += 1
		
		tt += sm.tt
		
		if asrt > 0:
			pt += sm.tt
			print('Output asserted at frame ', asrt, 'time', pt)
			print('Stopping iteration')
			
			break
		else:
			pt += t
		
	print('############### {0}: {1}, {2} ###################'.format( 'SAT' if asrt> 0 else 'TIMEOUT', sm.frame, pt if asrt > 0 else tt))
	return (sequence, asrt, 'SAT' if asrt> 0 else 'TIMEOUT', sm.frame, pt, TIMEOUT, tt, ar_tab)

def get_reward(asrt, frames, clauses, mems, times ):
	rewards = []
	for j in range(len(frames)):
		frame = frames[j]
		cla = clauses[j]
		mem = mems[j]
		to = times[j]
		reward =  np.exp(-1*to/(1+frame)) #np.exp(0.3*frame/MAX_FRAME + 0.2*cla/MAX_CLAUSE - 0.5*to/MAX_TIME)
		# if asrt > 0:
		# 	reward =  np.exp(1 - 0.2*frame/MAX_FRAME - 0.3*to/MAX_TIME)
		rewards.append(reward)
	return rewards

def main(argv):
	
	inputfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:", ["ifile="])
	except getopt.GetoptError:
			print("partition_run.py  -i ifile")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("partition_run.py -i ifile")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
	print("Input file is :" , inputfile)

	fname = (inputfile.split('/')[-1]).split('.')[0]
	print(fname)
	pp = PdfPages("plots/plot_Parallel_1800_{0}.pdf".format(fname))

	k =  4 # arms
	count1 = int(TIMEOUT/ST1)
	count2 = int(TIMEOUT/ST2)

	To1 = [ST1 for i in range(count1)]
	To2 = [ST2 for i in range(count2)]
	st = 60
	To3 = [st]
	tot = st
	while tot < TIMEOUT:
		To3.append(st)
		tot += st
		st = min(st*2,  TIMEOUT - tot)
	print(To1, To2, To3)
	 #[60, 60, 60, 60, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120]
	#print('iters', iters, T)
	

	figs = []
	string = 'Engine frames time wasted frames time wasted frames  time wasted \n'
	# seq = [(3, TIMEOUT)]
	# as0, cond0, sd0, tt0, ar_tab0 = runseq(inputfile, seq)
	# w0 = max(0, TIMEOUT - tt0)
	# tot0 = tt0 if as0 > 0 else TIMEOUT

	# seq = []
	# for t in To1:
	# 	seq.append((3, t))
	# as11, cond11, sd11, tt11, ar_tab11 = runseq(inputfile, seq)
	# w11 = max(0, TIMEOUT - tt11)
	# tot11 = tt11 if as11 > 0 else TIMEOUT

	# seq = []
	# for t in To2:
	# 	seq.append((3, t))
	# as21, cond21, sd21, tt21, ar_tab21 = runseq(inputfile, seq)
	# w21 = max(0, TIMEOUT - tt21)
	# tot21 = tt21 if as21 > 0 else TIMEOUT

	# seq = []
	# for t in To3:
	# 	seq.append((3, t))
	# as31, cond31, sd31, tt31, ar_tab31 = runseq(inputfile, seq)
	# w31 = max(0, TIMEOUT - tt31)
	# tot31 = tt31 if as31 > 0 else TIMEOUT
	
	# string += Actions[3]+'\t {0}  {1:0.2f}  {2:0.2f}  {3}  {4:0.2f}  {5:0.2f} {6}  {7:0.2f}  {8:0.2f} {9}  {10:0.2f}  {11:0.2f}\n'.format(sd0, tot0, w0, sd11, tot11, w11, sd21, tot21, w21, sd31, tot31, w31)

	# print(string)

	seq = []
	for t in To1:
		seq.append(t)
	seq1, as1, cond1, sd1, pt1, timeout1, tt1, ar_tab1 = run_parallel(inputfile, seq, k)
	w1 = max(0, pt1 - tt1) if as1 > 0 else max(0, timeout1 - tt1)
	tot1 = pt1 if as1 > 0 else TIMEOUT
		
	seq = []
	for t in To2:
		seq.append(t)
	seq2, as2, cond2, sd2, pt2, timeout2, tt2, ar_tab2 = run_parallel(inputfile, seq, k)
	w2 = max(0, pt2 - tt2) if as2 > 0 else max(0, timeout2 - tt2)
	tot2 = pt2 if as2 > 0 else TIMEOUT
		
	seq = []
	for t in To3:
		seq.append(t)
	seq3, as3, cond3, sd3, pt3, timeout3, tt3, ar_tab3 = run_parallel(inputfile, seq, k)
	w3 = max(0, pt3 - tt3) if as3 > 0 else max(0, timeout3 - tt3)
	tot3 = pt3 if as3 > 0 else TIMEOUT
		
	print('Sequences')
	print('Partition-1', seq1)
	print('Partition-2', seq2)
	print('Partition-3', seq3)



	string += '{0}  {1:0.2f}  {2:0.2f}  {3}  {4:0.2f}  {5:0.2f} {6}  {7:0.2f}  {8:0.2f}\n'.format(sd1, tot1, w1, sd2, tot2, w2, sd3, tot3, w3)	
	

	print('@@@@@@@@@@@@@@@@ RESULTS @@@@@@@@@@@@@@@@@')
	print(string)


	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')

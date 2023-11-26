import numpy as np 
import matplotlib.pyplot as plt 

plt.rcParams.update({'font.size': 12})

import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math, csv
from collections import namedtuple
from enum import Enum
from scipy import interpolate
from sklearn.neural_network import MLPRegressor
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
MAX_FRAME = 1e6
MAX_CLAUSE = 1e9
MAX_TIME = 3600
MAX_MEM = 4000
TIMEOUT = 3600

ST1 = 120
ST2 = 300

Actions = {0:'bmc2', 1:'bmc3g', 2:'bmc3', 3:'bmc3s', 4:'bmc3j', 5:'bmcu', 6:'bmc3r'}

k = 4 # arms

def get_fname(fn):
	fname = os.path.join(PATH, fn)
	# ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))	
	ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	#if DEBUG or self.n == 0:
	print(fname, ofname)

	# if self.n == 0:
	simplify(fname, ofname)
	print('Simplified model', ofname)
	return ofname

def runseq(fname, seq):

	#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
	#ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	seq_list = ['({0}, {1})'.format(Actions[t[0]], t[1]) for t in seq]
	sequence = ';'.join(seq_list)
	print('##################################')
	print('---- Running sequence {0} ----'.format(sequence), fname)
	ofname = fname

	#starting state
	sd = 0
	asrt = -1
	sm = None
	tt = 0
	pt = 0
	f = 0
	ar_tab = {}
	ar_tab1 = {}
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
			sd = sm.ld+1 if sm.ld > 0 else sm.ld
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
	
	frame = asrt if asrt > 0  else sm.frame
	print('############### {0}: {1}, {2} ###################'.format( 'SAT' if asrt> 0 else 'TIMEOUT', asrt if asrt > 0  else sm.frame, pt if asrt > 0 else tt))
	return (asrt, 'SAT' if asrt> 0 else 'TIMEOUT', frame, pt if asrt > 0 else tt, ar_tab)

def get_next_time(frames, clauses, confs, ttimes, r_flag = 0, flag = 0):
	#frames, clauses, mems, times, ttimes = [ar_tab[ky].frame for ky in ar_tab.keys()], [ar_tab[ky].conf for ky in ar_tab.keys()], \
	#[ar_tab[ky].mem for ky in ar_tab.keys()], [ar_tab[ky].to for ky in ar_tab.keys()], [ar_tab[ky].tt for ky in ar_tab.keys()]
	if len(frames)  == 0:
		return -1, -1
	sd = frames[0]
	ld = frames[-1]
	nd = max(5, (ld - sd) ) #/2)
	#ar_tab = self.engine_res[a]
	#frm = self.states
	frm  = -1
	ftrain, ctrain, conftrain, ttrain = [], [], [], []
	#frm = self.states
	frm  = -1
	prev = 0, 0, 0, 0
	pre_dif = 1.0
	partition_flag = 0
	for i in range(len(frames)): #ar_tab.keys():
		frm = frames[i]
		cla = clauses[i]
		conf = confs[i]
		tt = ttimes[i]
		# sm1 = ar_tab[frm]
		if cla > 0 and (cla not in ctrain): # and (sm1.tt - prev[3] > 0):
			diff = tt-prev[-1]
			if prev[0] <= frm-1 and prev[-1] > tt:
				partition_flag = 0
				# print('current ', sm1.frame, sm1.cla, sm1.conf, sm1.tt)
				# print('prev', prev)	
				ftrain, ctrain, conftrain, ttrain = [], [], [], []
			elif prev[0] == frm-1 and (prev[-1]> 5.0 and (tt/prev[-1] > 10.0 or (diff/pre_dif > 10.0))) and \
				(prev[-1]> 60.0 and (tt/prev[-1] > 1.1 or (diff/pre_dif > 1.2))):
				partition_flag = 1
			if flag and prev[-1]> 60.0: #and r_flag == 0 
				print(Actions[a], 'current (', frm, cla, conf, tt,  ') prev', prev, tt/prev[-1], partition_flag)	
			ftrain.append(frm)
			ctrain.append(cla)# - prev[1])
			conftrain.append(conf)# - prev[2])
			ttrain.append(tt)# - prev[3])
			prev = frm, cla, conf, tt


	next_tm = -1 #self.timeout[i]*SC
	ndt = -1
	if frm < 0:
		return next_tm, ndt

	last_frm = frm
	last_cla = cla #ar_tab[frm].cla
	last_tm = tt

	# if len(frames) > 10:
	#     ftrain, ttrain = frames[-11:], time_outs[-11:]
	if flag:
		print(Actions[a],'Training for action', a,  nd, len(ftrain), len(ctrain), len(conftrain), len(ttrain))
		print(Actions[a],'Last frame', prev, partition_flag)
	# print('Training data', (ftrain), (ttrain), (ttrain1))
	
	if len(ftrain) > 0:

		# f_test = np.arange(last_frm+1, last_frm+int(nd)+1, 1)

		# regr1 = MLPRegressor(random_state=1, max_iter=500).fit(np.array(ftrain).reshape(-1, 1), np.array(ctrain))
		# regr2 = MLPRegressor(random_state=1, max_iter=500).fit(np.array(ctrain).reshape(-1, 1), np.array(conftrain))
		# regr3 = MLPRegressor(random_state=1, max_iter=500).fit(np.array(conftrain).reshape(-1, 1), np.array(ttrain))
		# next_tm = max(regr3.predict(regr2.predict(regr1.predict(np.array(f_test).reshape(-1, 1)).reshape(-1, 1)).reshape(-1, 1)))

		# if flag:
		# 	print('Neural network prediction', f_test, next_tm)

		# next_tm1 = max(regr3.predict(regr2.predict(regr1.predict(np.array([last_frm+1]).reshape(1, -1)).reshape(-1, 1)).reshape(-1, 1)))

		# if flag:
		# 	print('Neural network prediction 1 frame', last_frm+1, next_tm1)

		# ndt = int(nd)+1

		# fpt1 = (ndt)/(next_tm) if next_tm > 0 else next_tm1
		# if flag:
		# 	print('Neural network prediction frame per time', fpt1)

	

		# ## inverse pred
		# i_next_to = last_tm*2.0 #if partition_flag == 0 else 
		# # i_cla = ifcls(ifconf(i_next_to))
		# # i_frame = iffrm(i_cla)
		# i_frame = int(fpt1*i_next_to)
		# #---
		# if r_flag:
		# 	# next_tm = np.max(new_to) #ttrain[-1]+  np.sum(new_to)
		# 	# ndt = int(nd)+1
		# 	if flag:
		# 		print(r_flag, 'NN Prediction for action {0}, for time {1}, frames {2}'.format((a,Actions[a]), next_tm, ndt), f_test)
		# else:
		# 	#next_tm = ttrain[-1] + i_next_to #np.sum(new_to)
		# 	ndt = max(nd, i_frame-last_frm)+ 1 #- ftrain[-1]+1
		# 	new_frames = np.arange(last_frm+1, last_frm+int(ndt), 1)
		# 	next_tm = max(regr3.predict(regr2.predict(regr1.predict(np.array(new_frames).reshape(-1, 1)).reshape(-1, 1)).reshape(-1, 1))) #last_tm + 
		# 	if partition_flag:
		# 		next_tm, ndt = -1, -1 
		# 	if flag:
		# 		print(r_flag, 'NN Prediction for action {0}, for time {1}, frames {2}'.format((a,Actions[a]), next_tm, ndt), new_frames, i_frame)
		

		# predict number of new clauses then predict timeout for next frame
		fcla = interpolate.interp1d(ftrain, ctrain, fill_value = 'extrapolate')
		# changed on 10/08
		#fto = interpolate.interp1d(ctrain, ttrain, fill_value = 'extrapolate')
		# predict number of new conflict clauses from new cluases then predict time to be taken for next frame
		fconf = interpolate.interp1d(ctrain, conftrain, fill_value = 'extrapolate')
		fto = interpolate.interp1d(conftrain, ttrain, fill_value = 'extrapolate')

		## inverse prediction
		# predict number of frames solved in a given timeout
		# ifconf = interpolate.interp1d(ttrain, conftrain, fill_value = 'extrapolate')
		# ifcls = interpolate.interp1d(conftrain, ctrain, fill_value = 'extrapolate')
		# iffrm = interpolate.interp1d(ctrain, ftrain, fill_value = 'extrapolate')
		next_frm = last_frm+1
		next_to = fto(fconf(fcla(next_frm)))
		if flag:
			print('Next frame', next_frm, 'Next time out', next_to)

		new_frames = np.arange(last_frm+1, last_frm+int(nd)+1, 2) #if int(1 + nd/2) > 2 else [last_frm+int(nd)+1]
		# new_cla = fcla(new_frames)
		# new_conf = fconf(new_cla)
		# changed on 10/08
		#new_to = fto(new_cla)
		new_to = fto(fconf(fcla(new_frames)))

		next_tm, ndt = max(new_to), int(nd)+1

		#---
		if r_flag:
			# next_tm = np.max(new_to) #ttrain[-1]+  np.sum(new_to)
			# ndt = int(nd)+1
			if flag:
				print(r_flag, 'Prediction for action {0}, for time {1}, frames {2}'.format((a,Actions[a]), next_tm, ndt), new_frames[0], new_frames[-1])
		else:
			#next_tm = ttrain[-1] + i_next_to #np.sum(new_to)
			fpt = (ndt)/(next_tm) if next_tm > 0 else ftrain[-1]/ttrain[-1]
			# ## inverse pred
			i_next_to = last_tm*2.0
			# # i_cla = ifcls(ifconf(i_next_to))
			#i_frame = iffrm(i_cla)
			i_frame = int(fpt*i_next_to)
			# ndt = max(nd, i_frame-last_frm)+ 1 #- ftrain[-1]+1
			# new_frames = np.arange(last_frm+1, last_frm+int(ndt), 1)
			# next_tm = np.max(fto(fconf(fcla(new_frames)))) #last_tm + 
			if partition_flag:
				next_tm, ndt = -1, -1 
			if flag:
				print(r_flag, 'Prediction for action {0}, for time {1}, frames {2}'.format((a,Actions[a]), next_tm, ndt), new_frames[-1], i_frame)
			
			# -----
			# new_cla = np.interp(new_frames, ftrain, ttrain)
			# new_to = np.interp(new_cla, ftrain, ttrain1)
			new_cla = fcla(new_frames)
			#next_tm = np.max(new_to) #np.sum(new_to)
			if flag:
				print(r_flag, 'Prediction ', new_cla[-1], ctrain[-1], new_cla[-1]/ctrain[-1] )
			ndt = int(nd)+1
			# if flag:
			while (ttrain[-1] >= next_tm and new_cla[-1] < 1.05*ctrain[-1] ): # atleast 5% increment in clauses #next_tm < self.timeout[self.n]: #*SC:
				new_frames = np.arange(last_frm+1, last_frm+int(ndt), 2)
				new_cla = fcla(new_frames)
				new_conf = fconf(new_cla)
				new_to = fto(new_conf)
				next_tm = np.max(new_to) #np.sum(new_to)
				ndt += 5
				if DEBUG:
					print('Prediction for {0} frames'.format(ndt), new_frames, new_cla, new_to)
	return next_tm, ndt

def get_reward(asrt, ar_tab): #frames, clauses, mems, times, ttimes ):
	rewards = [[],[],[]]
	cu_rewards = [[],[],[]]
	cu_re1 = 0
	cu_re2 = 0
	cu_re3 = 0
	c1 = 1.0/MAX_MEM
	c2 = 1.0/MAX_TIME

	frames, clauses, confs, mems, times, ttimes = [ar_tab[ky].frame for ky in ar_tab.keys()], [ar_tab[ky].cla for ky in ar_tab.keys()], \
	[ar_tab[ky].conf for ky in ar_tab.keys()], \
	[ar_tab[ky].mem for ky in ar_tab.keys()], [ar_tab[ky].to for ky in ar_tab.keys()], [ar_tab[ky].tt for ky in ar_tab.keys()]
	sd = frames[0]
	for j in range(len(frames)):
		frame = frames[j]
		cla = clauses[j]
		conf = confs[j]
		mem = mems[j]
		to = times[j]
		# reward =  np.exp(0.3*frame/MAX_FRAME + 0.2*cla/MAX_CLAUSE - 0.5*to/MAX_TIME)
		# if asrt > 0:
		# 	reward =  np.exp(1 - 0.2*frame/MAX_FRAME - 0.3*to/MAX_TIME)
		#reward = 2*np.exp(-1*to/(1+frame)) 

		reward1 = 2*np.exp(-to/(1+frame)) 
		cu_re1 += reward1
		rewards[0].append(reward1)
		cu_rewards[0].append(cu_re1/(j+1))

		# if j > 10:
		# 	frames1, clauses1, times1 = frames[0:j], clauses[0:j], ttimes[0:j] 
		# 	next_tm, ndt = get_next_time(frames1, clauses1, times1)
		# else:
		# 	ndt = 0
		# 	next_tm = -1

		# cu_re1 += to/(1+frame) 
		# rewards[0].append(cu_re1)
		# cu_avg = cu_re1/(j+1)
		# reward = np.exp(-0.4*cu_avg)  # + nd/nt)#(reward + np.exp(-pen/MAX_TIME))/cn
		# reward += np.exp(0.2*(frame-frames[0])/(1+frames[0])) # total number of frames explored --> more frames more reward
		# reward += np.exp(-0.2*next_tm/ndt) if (next_tm > -1 and nd > 0 and not math.innan(next_tm)) else 0 # reward based on future prediction
		# cu_rewards[0].append(np.exp(-1*cu_avg + ndt/next_tm))

		#reward = sm.cla/(10000 * sm.to) if sm.to > 0 else sm.to

		#reward 2
		# nt, nd = get_next_time(frames[0:j], clauses[0:j], confs[0:j], ttimes[0:j])
		cu_re2 += to/(1+frame)
		rewards[1].append(cu_re2)
		cu_avg1 = -0.5*cu_re2/(j+1)
		cu_avg2 = 0.5*((1)/(MAX_FRAME))
		# cu_avg3 = -0.2*nt/nd if (nt > -1 and nd > 0 and not math.isnan(nt)) else 0
		cu_rewards[1].append(np.exp(cu_avg1) + np.exp(cu_avg2))# + cu_avg3))


		# reward3 = to/(1+frame) 
		cu_re3 += to/(1+frame) 
		rewards[2].append(cu_re3)
		cu_avg = cu_re3/(j+1)
		cu_rewards[2].append(np.exp(-1*cu_avg))

	return rewards, cu_rewards

def part_res(a, fname, ofname, TO, name, part):
	seq = []
	for to in TO:
		seq.append((a, to))

	as1, cond1, sd1, tt1, ar_tab1 = runseq(ofname, seq)
	w1 = max(0, TIMEOUT - tt1)
	tot1 = tt1 if as1 > 0 else TIMEOUT

	string = '\t {0}  {1:0.2f}  {2:0.2f}\t'.format( sd1, tot1, w1) 
	frames1, clauses1, mems1, times1, ttimes1 = [ar_tab1[ky].frame for ky in ar_tab1.keys()], [ar_tab1[ky].conf for ky in ar_tab1.keys()], \
	[ar_tab1[ky].mem for ky in ar_tab1.keys()], [ar_tab1[ky].to for ky in ar_tab1.keys()], [ar_tab1[ky].tt for ky in ar_tab1.keys()]
	rewards1, cu_rewards1 = get_reward(as1, ar_tab1) #frames1, clauses1, mems1, times1, ttimes1)
	part1 = frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1

	rows1 = []
	# rows1.append(header)
	for kk in range(len(frames1)):
		row = []
		row.append(Actions[a])
		row.append(fname+str(name))
		row.append(frames1[kk])
		row.append(clauses1[kk])
		row.append(mems1[kk])
		row.append(times1[kk])
		row.append(rewards1[0][kk])
		row.append(cu_rewards1[0][kk])
		row.append(rewards1[1][kk])
		row.append(cu_rewards1[1][kk])
		row.append(rewards1[2][kk])
		row.append(cu_rewards1[2][kk])
		rows1.append(row)

	return string, part1, rows1

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
	# print(fname)

	ofname = get_fname(inputfile)

	# Parts = [2] 
	Parts = [1, 2, 3, 4]

	Part_names = ['Un-partitioned', 'Equal partition (small)', 'Equal partition (larger)', 'Increasing partition (fixed)']

	filename = "plots/BMC_records_single_2608_{0}_{1}_{2}.csv".format(TIMEOUT, fname, len(Parts))
	# header = ['Design', 'Frame', 'Clauses', 'Mem', 'time']
	# writing to csv file 
	with open(filename, 'w+') as csvfile: 
		print('filename', fname)

	k_action = range(0, k)
	pdfname = "plots/plot_Partition_2608_all_{0}_{1}_{2}.pdf".format(fname, TIMEOUT, len(Parts))

	count1 = int(TIMEOUT/ST1)
	count2 = int(TIMEOUT/ST2)
	#To1 = [300,300,300,300,300,300]
	#[150,150,150,150,150,150, 150,150, 150, 150, 150, 150]
	To = [TIMEOUT]
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
	
	print(To1, To2, To3)
	#To3 = [60, 60, 60, 60, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120]
	#print('iters', iters, T)
	Action_partitions = {}
	Action_partitions.update({1:[]})
	Action_partitions.update({2:[]})
	Action_partitions.update({3:[]})
	Action_partitions.update({4:[]})

	header = ['Action', 'Design', 'Frame', 'Clauses', 'Mem', 'time', 'reward']
	figs = []
	string = 'Engine frames time wasted frames time wasted frames  time wasted  frames  time wasted\n'
	for i in k_action: #range(k):
		# seq = []
		# for t in To3:
		# 	seq.append((i, t))
		# as4, cond4, sd4, tt4, ar_tab4 = runseq(inputfile, seq)
		# w4 = max(0, TIMEOUT - tt4)
		# tot4 = tt4 if as4 > 0 else TIMEOUT
		
		#string += '{0}  {1}  {2:0.2f}  {3:0.2f}  {4}  {5:0.2f}  {6:0.2f}  {7}  {8:0.2f}  {9:0.2f} {10} {11:0.2f} {12:0.2f}\n'.\
					 # format(Actions[i], sd1, tot1, w1, sd2, tot2, w2, sd3, tot3, w3, sd4, tot4, w4)	
		# string += '{0}  {1}  {2:0.2f}  {3:0.2f}  {4}  {5:0.2f}  {6:0.2f}  {7}  {8:0.2f}  {9:0.2f}\n'.format(Actions[i], sd1, tot1, w1, sd2, tot2, w2, sd3, tot3, w3) 

		string += '\n{0}'.format(Actions[i]) 

		if 1 in Parts:
			tr1, part1, rows1 = part_res(i, fname, ofname, To, Part_names[0], 1)
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part1
			string += tr1
			res1 = Action_partitions[1]
			res1.append(part1)
			Action_partitions.update({1:res1})

		if 2 in Parts:
			tr2, part2, rows2 = part_res(i, fname, ofname, To1, Part_names[1], 2)
			frames2, clauses2, mems2, times2, rewards2, cu_rewards2, ttimes2 = part2
			string += tr2

			res1 = Action_partitions[2]
			res1.append(part2)
			Action_partitions.update({2:res1})

		if 3 in Parts:
			tr3, part3, rows3 = part_res(i, fname, ofname, To2, Part_names[2], 3)
			frames3, clauses3, mems3, times3, rewards3, cu_rewards3, ttimes3 = part3
			string += tr3
			res1 = Action_partitions[3]
			res1.append(part3)
			Action_partitions.update({3:res1})

		if 4 in Parts:
			tr4, part4, rows4 = part_res(i, fname, ofname, To3, Part_names[3], 4)
			frames4, clauses4, mems4, times4, rewards4, cu_rewards4, ttimes4 = part4
			string += tr4	
			res1 = Action_partitions[4]
			res1.append(part4)
			Action_partitions.update({4:res1})

		filename = "plots/BMC_records_single_{0}_{1}_{2}.csv".format(TIMEOUT, fname, len(Parts))
		# header = ['Design', 'Frame', 'Clauses', 'Mem', 'time']
		# writing to csv file 
		with open(filename, 'a+') as csvfile: 
		    # creating a csv writer object 
		    csvwriter = csv.writer(csvfile) 
		        
		    # writing the fields 
		    csvwriter.writerow(header) 
		        
		    # writing the data rows 
		    if 1 in Parts:
		    	csvwriter.writerows(rows1)
		    if 2 in Parts:
		    	csvwriter.writerows(rows2)
		    if 3 in Parts:
		    	csvwriter.writerows(rows3)
		    if 4 in Parts:
		    	csvwriter.writerows(rows4)

		fig1 , ax = plt.subplots()
		# plt.subplot(2, 2, 1)
		if 1 in Parts:
		    ax.plot(frames1, clauses1, 'b', label=Part_names[0])
		if 2 in Parts:
			ax.plot(frames2, clauses2, 'r', label=Part_names[1])
		if 3 in Parts:
			ax.plot(frames3, clauses3, 'g', label=Part_names[2])
		if 4 in Parts:
			ax.plot(frames4, clauses4, 'c', label=Part_names[3])
		# plt.plot(frames4, clauses4, 'c', label='Part-3')
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		ax.set_xlabel("Depth")
		ax.set_ylabel("Clauses")

		# ax2 = ax.twinx()
		# if 1 in Parts:
		# 	ax2.plot(frames1, rewards1[1], 'c', label='total')
		# if 2 in Parts:
		# 	ax2.plot(frames2, rewards2[1], 'y', label='Part-1')
		# if 3 in Parts:
		# 	ax2.plot(frames3, rewards3[1], 'k', label='Part-2')
		# # plt.plot(frames4, clauses4, 'c', label='Part-3')
		# # plt.legend(bbox_to_anchor=(1.3, 0.5))
		# ax2.set_xlabel("Depth")
		# ax2.set_ylabel("Rewards")

		plt.title(Actions[i]+"-- Depth vs clauses")
		plt.legend(fontsize="12")

		fig2 , ax = plt.subplots()
		# plt.subplot(2, 2, 1)
		if 1 in Parts:
			ax.plot(frames1, mems1, 'b', label=Part_names[0])
		if 2 in Parts:
			ax.plot(frames2, mems2, 'r', label=Part_names[1])
		if 3 in Parts:
			ax.plot(frames3, mems3, 'g', label=Part_names[2])
		if 4 in Parts:
			ax.plot(frames4, mems4, 'c', label=Part_names[3])
		# plt.plot(frames4, clauses4, 'c', label='Part-3')
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		ax.set_xlabel("Depth")
		ax.set_ylabel("Memory")

		# ax2 = ax.twinx()
		# if 1 in Parts:
		# 	ax2.plot(frames1, rewards1[1], 'c', label='total')
		# if 2 in Parts:
		# 	ax2.plot(frames2, rewards2[1], 'y', label='Part-1')
		# if 3 in Parts:
		# 	ax2.plot(frames3, rewards3[1], 'k', label='Part-2')
		# # plt.plot(frames4, clauses4, 'c', label='Part-3')
		# # plt.legend(bbox_to_anchor=(1.3, 0.5))
		# ax2.set_xlabel("Depth")
		# ax2.set_ylabel("Rewards")

		plt.title(Actions[i]+"-- Depth vs memory")
		plt.legend(fontsize="12")

		def ttms_ffms(frames, times):
			ttms = []
			ffms = []
			pre_tm = 0.0
			jk = 0
			for tm in times:
				if tm < pre_tm:
					continue
				ttms.append(tm)
				ffms.append(frames[jk])
				pre_tm = tm
				jk += 1
			return ffms, ttms
			#return frames, times

		fig3 , ax = plt.subplots()
		# plt.subplot(2, 2, 1)
		if 1 in Parts:
			ffms, ttms = ttms_ffms(frames1, times1)
			ax.plot(ffms, ttms, 'b', label=Part_names[0])
		if 2 in Parts:
			ffms, ttms = ttms_ffms(frames2, times2)
			ax.plot(ffms, ttms, 'r', label=Part_names[1])
		if 3 in Parts:
			ffms, ttms = ttms_ffms(frames3, times3)
			ax.plot(ffms, ttms, 'g', label=Part_names[2])
		if 4 in Parts:
			ffms, ttms = ttms_ffms(frames4, times4)
			ax.plot(ffms, ttms, 'c', label=Part_names[3])
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		ax.set_xlabel("Depth")
		ax.set_ylabel("times")

		fig3 , ax = plt.subplots()
		# plt.subplot(2, 2, 1)
		if 1 in Parts:
			# ffms, ttms = ttms_ffms(frames1, times1)
			ax.plot(frames1, ttimes1, 'b', label=Part_names[0])
		if 2 in Parts:
			# ffms, ttms = ttms_ffms(frames2, times2)
			ax.plot(frames2, ttimes2,'r', label=Part_names[1])
		if 3 in Parts:
			# ffms, ttms = ttms_ffms(frames3, times3)
			ax.plot(frames3, ttimes3, 'g', label=Part_names[2])
		if 4 in Parts:
			# ffms, ttms = ttms_ffms(frames4, times4)
			ax.plot(frames4, ttimes4, 'c', label=Part_names[3])
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		ax.set_xlabel("Depth")
		ax.set_ylabel("times")

		# ax2 = ax.twinx()
		# if 1 in Parts:
		# 	ax2.plot(frames1, rewards1[1], 'c', label='total')
		# if 2 in Parts:
		# 	ax2.plot(frames2, rewards2[1], 'y', label='Part-1')
		# if 3 in Parts:
		# 	ax2.plot(frames3, rewards3[1], 'k', label='Part-2')
		# # plt.legend(bbox_to_anchor=(1.3, 0.5))
		# ax2.set_xlabel("Depth")
		# ax2.set_ylabel("Rewards")

		plt.title(Actions[i]+"-- Times vs Depth")
		plt.legend(fontsize="12")
		
		fig4 = plt.figure()
		plt.subplot(2, 2, 1)
		if 1 in Parts:
			plt.plot(frames1, rewards1[0], 'b', label=Part_names[0])
		if 2 in Parts:
			plt.plot(frames2, rewards2[0], 'r', label=Part_names[1])
		if 3 in Parts:
			plt.plot(frames3, rewards3[0], 'g', label=Part_names[2])
		if 4 in Parts:
			plt.plot(frames4, rewards4[0], 'c', label=Part_names[3])
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		plt.xlabel("Depth")
		plt.ylabel("Reward1")
		plt.title(Actions[i]+" -- Depth vs Reward1")
		plt.legend(fontsize="12")


		plt.subplot(2, 2, 2)
		if 1 in Parts:
			plt.plot(frames1, rewards1[1], 'b', label=Part_names[0])
		if 2 in Parts:
			plt.plot(frames2, rewards2[1], 'r', label=Part_names[1])
		if 3 in Parts:
			plt.plot(frames3, rewards3[1], 'g', label=Part_names[2])
		if 4 in Parts:
			plt.plot(frames4, rewards4[1], 'c', label=Part_names[3])
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		plt.xlabel("Depth")
		plt.ylabel("Reward2")
		plt.title(Actions[i]+" -- Depth vs Reward2")
		plt.legend(fontsize="12")


		plt.subplot(2, 2, 3)
		if 1 in Parts:
			plt.plot(frames1, rewards1[2], 'b', label=Part_names[0])
		if 2 in Parts:
			plt.plot(frames2, rewards2[2], 'r', label=Part_names[1])
		if 3 in Parts:
			plt.plot(frames3, rewards3[2], 'g', label=Part_names[2])
		if 4 in Parts:
			plt.plot(frames4, rewards4[2], 'c', label=Part_names[3])
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		plt.xlabel("Depth")
		plt.ylabel("Reward2")
		plt.title(Actions[i]+" -- Depth vs Reward3")
		plt.legend(fontsize="12")

		plt.subplot(2, 2, 4)
		if 1 in Parts:
			plt.plot(frames1, clauses1, 'b', label=Part_names[0])
		if 2 in Parts:
			plt.plot(frames2, clauses2, 'r', label=Part_names[1])
		if 3 in Parts:
			plt.plot(frames3, clauses3, 'g', label=Part_names[2])
		if 4 in Parts:
			plt.plot(frames4, clauses4, 'c', label=Part_names[3])
		# plt.legend(bbox_to_anchor=(1.3, 0.5))
		plt.xlabel("Depth")
		plt.ylabel("Reward2")
		plt.title(Actions[i]+" -- Depth vs clauses")
		plt.legend(fontsize="12")

		figs.append(fig1)
		figs.append(fig2)
		figs.append(fig3)
		figs.append(fig4)

		pp = PdfPages(pdfname)
		for fig in figs:
			pp.savefig(fig)   
		pp.close()

	print('@@@@@@@@@@@@@@@@ RESULTS @@@@@@@@@@@@@@@@@')
	print(string)
	print('@@@@@@@@@@@@@@@@ RESULTS end @@@@@@@@@@@@@@@@@')
	sys.stdout.flush()

	colors = cm.rainbow(np.linspace(0, 1, k))
	for p in Action_partitions.keys():
		all_acts = Action_partitions[p]

		fig = plt.figure(figsize=(12,8))
		i = 0
		for part in all_acts:
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part
			# frames2, clauses2, mems2, times2, rewards2, cu_rewards2 = part2
			# frames3, clauses3, mems3, times3, rewards3, cu_rewards3 = part3

			# axes[0][0].plot(frames1, rewards1, colors[i], label=Actions[i])

			if 1 in Parts:
				plt.plot(ttimes1, frames1, color=colors[i], linestyle='solid', label=Actions[i])

			# axes[1][0].plot(frames1, clauses1, colors[i], label=Actions[i])

			# axes[1][1].plot(frames1, mems1, colors[i], label=Actions[i])
		
			i += 1

		# axes[0][0].set_xlabel("Depth")
		# axes[0][0].set_ylabel("Reward")
		# axes[0][0].legend()

		plt.xlabel("Times")
		plt.ylabel("Depth")
		plt.legend(fontsize="12")
		figs.append(fig)

		fig = plt.figure(figsize=(12,8))
		i = 0
		for part in all_acts:
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part
			# frames2, clauses2, mems2, times2, rewards2, cu_rewards2 = part2
			# frames3, clauses3, mems3, times3, rewards3, cu_rewards3 = part3

			# axes[0][0].plot(frames1, rewards1, colors[i], label=Actions[i])

			if 1 in Parts:
				plt.plot(frames1, clauses1, color=colors[i], linestyle='solid', label=Actions[i])

			# axes[1][0].plot(frames1, clauses1, colors[i], label=Actions[i])

			# axes[1][1].plot(frames1, mems1, colors[i], label=Actions[i])
		
			i += 1

		# axes[0][0].set_xlabel("Depth")
		# axes[0][0].set_ylabel("Reward")
		# axes[0][0].legend()

		plt.xlabel("Depth")
		plt.ylabel("Clauses")
		plt.legend(fontsize="12")
		figs.append(fig)

		fig = plt.figure(figsize=(12,8))
		i = 0
		for part in all_acts:
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part
			# frames2, clauses2, mems2, times2, rewards2, cu_rewards2 = part2
			# frames3, clauses3, mems3, times3, rewards3, cu_rewards3 = part3

			# axes[0][0].plot(frames1, rewards1, colors[i], label=Actions[i])

			if 1 in Parts:
				plt.plot(frames1, mems1, color=colors[i], linestyle='solid', label=Actions[i])

			# axes[1][0].plot(frames1, clauses1, colors[i], label=Actions[i])

			# axes[1][1].plot(frames1, mems1, colors[i], label=Actions[i])
		
			i += 1

		# axes[0][0].set_xlabel("Depth")
		# axes[0][0].set_ylabel("Reward")
		# axes[0][0].legend()

		plt.xlabel("Depth")
		plt.ylabel("Memory")
		plt.legend(fontsize="12")
		figs.append(fig)

		fig = plt.figure(figsize=(12,8))
		i = 0
		for part in all_acts:
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part
			# frames2, clauses2, mems2, times2, rewards2, cu_rewards2 = part2
			# frames3, clauses3, mems3, times3, rewards3, cu_rewards3 = part3

			# axes[0][0].plot(frames1, rewards1, colors[i], label=Actions[i])

			if 1 in Parts:
				plt.plot(frames1, cu_rewards1[0], color=colors[i], linestyle='solid', label=Actions[i])

			# axes[1][0].plot(frames1, clauses1, colors[i], label=Actions[i])

			# axes[1][1].plot(frames1, mems1, colors[i], label=Actions[i])
		
			i += 1

		# axes[0][0].set_xlabel("Depth")
		# axes[0][0].set_ylabel("Reward")
		# axes[0][0].legend()

		plt.xlabel("Depth")
		plt.ylabel("Cumulative Reward- 0")
		plt.legend(fontsize="12")
		figs.append(fig)

		fig = plt.figure(figsize=(12,8))
		i = 0
		for part in all_acts:
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part
			# frames2, clauses2, mems2, times2, rewards2, cu_rewards2 = part2
			# frames3, clauses3, mems3, times3, rewards3, cu_rewards3 = part3

			# axes[0][0].plot(frames1, rewards1, colors[i], label=Actions[i])

			if 1 in Parts:
				plt.plot(frames1, cu_rewards1[1], color=colors[i], linestyle='solid', label=Actions[i])

			# axes[1][0].plot(frames1, clauses1, colors[i], label=Actions[i])

			# axes[1][1].plot(frames1, mems1, colors[i], label=Actions[i])
		
			i += 1

		# axes[0][0].set_xlabel("Depth")
		# axes[0][0].set_ylabel("Reward")
		# axes[0][0].legend()

		plt.xlabel("Depth")
		plt.ylabel("Reward function")
		plt.legend(fontsize="15")
		figs.append(fig)
		
		fig = plt.figure(figsize=(12,8))
		i = 0
		for part in all_acts:
			frames1, clauses1, mems1, times1, rewards1, cu_rewards1, ttimes1 = part
			# frames2, clauses2, mems2, times2, rewards2, cu_rewards2 = part2
			# frames3, clauses3, mems3, times3, rewards3, cu_rewards3 = part3

			# axes[0][0].plot(frames1, rewards1, colors[i], label=Actions[i])

			if 1 in Parts:
				plt.plot(frames1, cu_rewards1[2], color=colors[i], linestyle='solid', label=Actions[i])

			# axes[1][0].plot(frames1, clauses1, colors[i], label=Actions[i])

			# axes[1][1].plot(frames1, mems1, colors[i], label=Actions[i])
		
			i += 1

		# axes[0][0].set_xlabel("Depth")
		# axes[0][0].set_ylabel("Reward")
		# axes[0][0].legend()

		plt.xlabel("Depth")
		plt.ylabel("Reward function 2")
		plt.legend(fontsize="12")
		# axes[1][0].set_xlabel("Depth")
		# axes[1][0].set_ylabel("Clauses")
		# # axes[1][0].legend()

		# axes[1][1].set_xlabel("Depth")
		# axes[1][1].set_ylabel("Memory")
		# axes[1][1].legend()

		figs.append(fig)

		pp = PdfPages(pdfname)
		for fig in figs:
			pp.savefig(fig)   
		pp.close()

	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')

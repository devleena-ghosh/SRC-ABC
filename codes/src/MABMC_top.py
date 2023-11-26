import numpy as np 
PLOT = True
if PLOT:
	import matplotlib.pyplot as plt 
	from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math, csv
from collections import namedtuple
from enum import Enum
from scipy import interpolate
import tracemalloc
import random
from sklearn.neural_network import MLPRegressor
# from memory_profiler import memory_usage
from abcBMCUtil import *

DEBUG = True
DEBUG = False
OPT = True
T = 60 
TIMEOUT = 3600 #/2.0
SC = 2
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
# DIFF = 2 # function of depth, time, memory
DIFF = 3 # function number of frames/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
FIXED = False
FIXED = True
PDR = False
MAX_FRAME = 1e5
MAX_CLAUSE = 1e9
MAX_TIME = TIMEOUT
MAX_MEM = 2000
MAX_TIMEOUT = 4*MAX_TIME
F = 3

time_outs = {}
Actions = ['bmc2', 'bmc3', 'bmc3s', 'bmc3g', 'bmcu', 'bmc3r', 'bmc3j2'] #, 'bmc3j3', 'bmc3j4'] # 'pdr']
To1 = []
st = 30
tot = 0
Iters = 0
while tot < TIMEOUT:
	To1.append(st)
	tot += st
	st = min(st*SC,  TIMEOUT - tot)
	Iters += 1

M = int(len(Actions))

def run_engine(ofname, a, sd, t=0, f=0):
	print('run_engine --', a, sd, t, f)
	sys.stdout.flush()
	if a == 0:    #ABC bmc2
		asrt, sm, ar_tab, tt1 = bmc2(ofname, sd, t=t, f=f)
	elif a == 1: #ABC bmc3
		asrt, sm, ar_tab, tt1 = bmc3(ofname, sd,  t=t, f=f)
	elif a == 2: #ABC bmc3rs
		asrt, sm, ar_tab, tt1 = bmc3rs(ofname, sd,  t=t, f=f)
	elif a == 3: #ABC bmc3g
		asrt, sm, ar_tab, tt1 = bmc3rg(ofname, sd,  t=t, f=f)
	elif a == 4: #ABC bmc3u
		asrt, sm, ar_tab, tt1 = bmc3ru(ofname, sd,  t=t, f=f)
	elif a == 5: #ABC bmc3r
		asrt, sm, ar_tab, tt1 = bmc3r(ofname, sd,  t=t, f=f)
	elif a == 6: #ABC bmc3j=2
		asrt, sm, ar_tab, tt1 = bmc3j(ofname, sd, j = 2,  t=t, f=f)
	elif a == 7: #ABC bmc3j=3
		asrt, sm, ar_tab, tt1 = bmc3j(ofname, sd, j = 3, t=t, f=f)
	elif a == 8: #ABC bmc3j=4
		asrt, sm, ar_tab, tt1 = bmc3j(ofname, sd, j = 4, t=t, f=f)

	sys.stdout.flush()
	return asrt, sm, ar_tab, tt1

class bandit:

	def __init__(self, k, iters, alpha, reward=0, fname = ''):
		# Number of arms
		self.k = k
		# Number of iterations
		self.iters = iters
		# Step count
		self.n = 0
		# Step count for each arm
		self.k_n = np.zeros(k)
		# Total mean reward
		self.mean_reward = 0
		self.reward = np.zeros(iters)
		# Mean reward for each arm
		self.k_reward = np.zeros(k)
		self.eps = 0.2

		for i in range(self.k):
			self.k_reward[i] = reward

		# recency constant
		self.alpha = alpha
		##
		self.fname =  fname
		self.timeout = np.zeros(iters)
		#for i in range(iters):
		self.timeout[0] = To1[0]
		self.frameout = np.zeros(iters)
		self.partition_flag = [1 for i in range(k)]

		fn = (fname.split('/')[-1]).split('.')[0]
		self.csvfilename = 'todaes_results/new_dist_MABMC_tm_pred_{0}.csv'.format(fn)
		
		self.engine_res = [{} for i in range(k)]

		# current state of the run; eg - bmc depth
		self.states = 0

	def get_next_time(self, a, ld, r_flag=1, flag = 0):
		# expected_f = self.frameout[n]
		# cal_f = ld

		nd = max(5, (ld - self.states) ) #/2)
		ar_tab = self.engine_res[a]

		# expected_t = self.timeout[n]
		# cal_t = ar_tab[ld].tt

		ftrain, ctrain, conftrain, ttrain = [], [], [], []
		#frm = self.states
		frm  = -1
		prev = 0, 0, 0, 0
		pre_dif = 1.0
		diff = 1.0
		# partition_flag = 1
		for frm in ar_tab.keys():
			sm1 = ar_tab[frm]
			if sm1.cla > 0 and (sm1.cla not in ctrain): # and (sm1.tt - prev[3] > 0):
				pre_dif= diff
				diff = sm1.tt-prev[-1]
				if prev[0] <= sm1.frame-1 and (prev[-1] > sm1.tt or sm1.cla < 0.8*prev[1]): #or prev[2] < sm1.conf:
					''' Either taking less time after partition or with less clauses'''
					# partition_flag = 1
					# print('current ', sm1.frame, sm1.cla, sm1.conf, sm1.tt)
					# print('prev', prev)	
					ftrain, ctrain, conftrain, ttrain = [], [], [], []
					self.partition_flag[a] = 1

				if prev[0] == sm1.frame-1 and ((prev[-1]> 5.0 and prev[2] < sm1.conf) or
				(prev[-1]> 5.0 and (sm1.tt/prev[-1] > 10.0 or (diff/pre_dif > 10.0))) or \
					(prev[-1]> 60.0 and (sm1.tt/prev[-1] > 1.1 or (diff/pre_dif > 1.2)))):
					self.partition_flag[a] = 0 # no partition

				# if prev[0] <= sm1.frame-1 and prev[-1] > sm1.tt: #or prev[2] < sm1.conf:
				# 	partition_flag = 0 # does not apply partition
				# 	# print('current ', sm1.frame, sm1.cla, sm1.conf, sm1.tt)
				# 	# print('prev', prev)	
				# 	ftrain, ctrain, conftrain, ttrain = [], [], [], []

				# #if prev[0] == sm1.frame-1 and prev[1] > 0 and sm1.cla < 0.8*prev[1]: 
				# if prev[0] == sm1.frame-1 and ((prev[-1]> 5.0 and prev[2] < sm1.conf) or
				# (prev[-1]> 5.0 and (sm1.tt/prev[-1] > 10.0 or (diff/pre_dif > 10.0))) or \
				# 	(prev[-1]> 60.0 and (sm1.tt/prev[-1] > 1.1 or (diff/pre_dif > 1.2)))):
				# 	partition_flag = 1 # apply partition

				# if flag and prev[-1]> 60.0 and partition_flag: #and r_flag == 0 
				# 	print(Actions[a], 'current (', sm1.frame, sm1.cla, sm1.conf, sm1.tt,  ') prev', prev, sm1.cla/prev[1], partition_flag)	

				ftrain.append(sm1.frame)
				ctrain.append(sm1.cla)# - prev[1])
				conftrain.append(sm1.conf)# - prev[2])
				ttrain.append(sm1.tt)# - prev[3])
				prev = sm1.frame, sm1.cla, sm1.conf, sm1.tt

				sys.stdout.flush()

		next_tm = -1 #self.timeout[i]*SC
		ndt = -1
		if frm < 0:
			return next_tm, ndt

		last_frm = frm
		last_cla = sm1.cla #ar_tab[frm].cla
		last_tm = sm1.tt

		# if len(frames) > 10:
		#     ftrain, ttrain = frames[-11:], time_outs[-11:]
		if flag:
			print(Actions[a],'Training for action', a,  nd, len(ftrain), len(ctrain), len(conftrain), len(ttrain))
			print(Actions[a],'Last frame', prev, self.partition_flag[a])
		# print('Training data', (ftrain), (ttrain), (ttrain1))
		
		if len(ftrain) > 0:

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
			next_cla = fcla(next_frm) #next_vc[1] if next_vc[1] > 0 else fcla(next_frm)
			next_to = fto(fconf(next_cla))
			if flag:
				print('Next frame', next_frm, 'Next time out', next_to)

			new_frames = np.arange(last_frm+1, last_frm+int(nd)+1, 1) #if int(1 + nd/2) > 2 else [last_frm+int(nd)+1]
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
				i_next_to = last_tm*SC
				# # i_cla = ifcls(ifconf(i_next_to))
				#i_frame = iffrm(i_cla)
				i_frame = int(fpt*i_next_to)
				# ndt = max(nd, i_frame-last_frm)+ 1 #- ftrain[-1]+1
				# new_frames = np.arange(last_frm+1, last_frm+int(ndt), 1)
				# next_tm = np.max(fto(fconf(fcla(new_frames)))) #last_tm + 
				new_frames = np.arange(last_frm+1, last_frm+int(nd)+1, 1) #if int(1 + nd/2) > 2 else [last_frm+int(nd)+1]
				# new_cla = fcla(new_frames)
				# new_conf = fconf(new_cla)
				# changed on 10/08
				#new_to = fto(new_cla)
				new_to = fto(fconf(fcla(new_frames)))
				if self.partition_flag[a] == 0:
					# next_tm, ndt = -1, -1
					print("@@@@ PARTITION NOT NEEDED FOR ENGINE {0}!!".format(Actions[a])) 
				
				# else:
				# -----
				# new_cla = np.interp(new_frames, ftrain, ttrain)
				# new_to = np.interp(new_cla, ftrain, ttrain1)
				new_cla = fcla(new_frames)
				new_conf = fconf(new_cla)
				#next_tm = np.max(new_to) #np.sum(new_to)
				if DEBUG and flag:
					print('new frames', np.arange(last_frm+1, last_frm+int(nd)+1, 1))
					print(r_flag, 'Prediction ', new_frames, new_cla, new_conf, conftrain[-1], new_conf/conftrain[-1])
				ndt = int(nd)+1
				# if flag:
				iiflg = 0
				while (SC*ttrain[-1] >= next_tm):# and new_conf[-1] < 2*conftrain[-1] ): # atleast 5% increment in clauses #next_tm < self.timeout[self.n]: #*SC:
					new_frames = np.arange(last_frm+1, last_frm+int(ndt), 1)
					new_cla = fcla(new_frames)
					new_conf = fconf(new_cla)
					new_to = fto(new_conf)
					next_tm = np.max(new_to) #np.sum(new_to)
					ndt += max(5, int(nd)/2)+1
					iiflg += 1
					if iiflg >= 5:
						print(iiflg, 'Prediction for {0} frames'.format(ndt), new_frames, new_cla, new_to, 'next_tm', next_tm, 'ndt', ndt)
						break
					if DEBUG:
						print(iiflg, 'Prediction for {0} frames'.format(ndt), new_frames, new_cla, new_to)

				if flag:
					print(r_flag, iiflg, 'Prediction for action {0}, for time {1}, frames {2}'.format((a,Actions[a]), next_tm, ndt), new_frames[-1], i_frame)
				
		# if r_flag:

		#print('get_next_time: Prediction for action {0}, for time {1}, frames {2}'.format((a,Actions[a]), next_tm, ndt))
		sys.stdout.flush()
		return next_tm, ndt

	def cal_reward(self, a, sm, t, asrt, tt1, sd, ed = -1):
		ar_tab_old = self.engine_res[a]
		MAX_mem = 0
		if sm is not None:
			# print(sm)
			reward = 0
			if DIFF == 0 :
				reward = (sm.frame - sd)
			elif DIFF == 1:
				reward = sm.frame
			elif DIFF == 2:
				pen = t - sm.tt
				reward =  np.exp(2*sm.frame/MAX_FRAME - 0.5*sm.mem/MAX_MEM - 0.6*sm.tt/MAX_TIME)
				# + 0.1*sm.cla/MAX_CLAUSE 
				#np.exp(0.5*sm.frame/MAX_FRAME + 0.1*sm.cla/MAX_CLAUSE - 0.3*sm.to/MAX_TIME) #-  0.1*pen/MAX_TIME)
				if asrt > 0:
					reward =  np.exp(1 - 0.4*sm.frame/MAX_FRAME - 0.6*sm.tt/MAX_TIME)
			else:
				#reward = sm.cla/(10000 * sm.to) if sm.to > 0 else sm.to
				c1 = 1.0/MAX_MEM
				c2 = 1.0/MAX_TIME
				pen = t - sm.tt
				reward = 0
				ky = sd
				cn = 0
				md = MAX_FRAME
				for ky in ar_tab_old.keys():
					if (ed > -1 and sd <= ky <= ed) or sd <= ky:
						tm = ar_tab_old[ky].to
						mem = ar_tab_old[ky].mem
						#reward += 2*np.exp(-c1*mem/(1+ky) - c2*tm/(1+ky))    
						reward += (1*tm/(1+ky))
						cn += 1
					md = max(md, ky)
				wa = (reward/cn) if cn > 0 else 0

				# nt, nd = self.get_next_time(a, sm.ld, r_flag = 1, flag = 0)
				# print('In reward', sd, sm.frame, nt, nd)

				reward = np.exp(-0.5*wa)  # + nd/nt)#(reward + np.exp(-pen/MAX_TIME))/cn
				if sd > 0:
					reward += np.exp(0.5*(ky-sd)/(md))
					#np.exp(0.5*(ky-sd)/(1+sd)) # total number of frames explored --> more frames more reward
					# reward += np.exp(-0.2*nt/nd) if (nt > -1 and nd > 0 and not math.isnan(nt)) else 0 # reward based on future prediction
					# reward += np.exp(-0.2*pen/t) if (pen > 0 and t > 0) else 0 # reward based on future prediction
			
				if sd > sm.frame:
					reward = -0.5 * np.exp(t/MAX_TIME) # no exploration --> penalty
		else:
			sm =  abc_result(frame=sd, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = tt1,ld= sd)
			if asrt > 0:
				reward = asrt
			else:
				reward = -1 * np.exp(t/MAX_TIME) #np.log(t)

		print('cal_reward', sd, sm.frame, reward, sm)

		sys.stdout.flush()
		return reward, sm

	def get_reward(self, a, t1 = -1):

		# get starting depth
		# sd = 0
		# if self.n  == 0:
		sd = int(self.states)+1 if int(self.states) > 0 else int(self.states) 

		fname = os.path.join(PATH, self.fname)
		#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
		ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))
		if DEBUG or self.n == 0:
			print(fname, ofname)

		st = ''

		f = int(self.frameout[self.n])

		if t1  == -1:
			t = T
		else:
			t = int(t1)

		if self.n == 0:
			simplify(fname, ofname)
			print('Simplified model', ofname)

		sm = None
		asrt = -1
		ar_tab = {}
		asrt, sm, ar_tab, tt1 = run_engine(ofname, a, sd, t, f)
			#asrt, sm, ar_tab, tt1 = pdr(ofname, t)
		
		# min_t = MAX_TIME
		# if len(ar_tab.keys()) > 0:
        # 	min_k = sorted(ar_tab.keys())[0]
        # 	min_t =  ar_tab[min_k].to

		ar_tab_old = self.engine_res[a]
		for ky in ar_tab.keys():
			sm1 = ar_tab[ky]
			if DEBUG:
				print('get_reward: explored frame', sm1)
			if sm1 and  sm1.frame > sd:
				sm = sm1
				ar_tab_old.update({ky:sm})
				if DEBUG:
					print('get_reward: added frame', sm1)

		self.engine_res[a] = ar_tab_old
		reward, sm = self.cal_reward(a, sm, t, asrt, tt1, sd) #, next_vc)
		
		#todo: clean up the directory and store it in a file
		# for k in list(ar_tab_old):
		#     v = ar_tab_old[k]
		#     if not v:
		#         del ar_tab_old[k]

		# ft_vs_pt = {}/
		sys.stdout.flush()
		return reward, sm

	def write_log(self, a, sm, reward):
		rows = []
		with open(self.csvfilename, 'a+') as csvfile: 
			# creating a csv writer object 
			csvwriter = csv.writer(csvfile) 
			# for a in range(k):
			ar_tab_old = self.engine_res[a]

			#frame=sd, conf=0, var=0, cla=0, mem = -1, to
			for ky in ar_tab_old.keys():

				row = []
				frame = ar_tab_old[ky].frame
				cla = ar_tab_old[ky].cla
				conf = ar_tab_old[ky].conf
				var = ar_tab_old[ky].var
				tm = ar_tab_old[ky].to
				mem = ar_tab_old[ky].mem

				nt, nd =  self.get_next_time(a, sm.ld, r_flag=1, flag = 0)
				if self.states <= ky: 
					row.append(Actions[a])
					# row.append(fname)
					row.append(frame)
					row.append(cla)
					row.append(conf)
					row.append(var)
					row.append(tm)
					row.append(mem)
					row.append(nt)
					row.append(nd)
					pt = 0
					if nd == -1:
						pt = -1
						# row.append(-1)
					else:
						pt = nt/nd
					row.append(pt)
					row.append(reward)

					# ft_vs_pt.update({frame:(tm, pt)})
				if len(row) > 0:
					rows.append(row)
			if len(rows) > 0:
				csvwriter.writerows(rows)

		
	def run(self):
		totalTime = 0
		all_time = 0
		seq = []
		sm = None
		next_to = -1
		next_time = {}
		# for i in self.iters:
		#     next_time[i] = -1
		for a in range(self.k):
			time_outs.update({a:next_time})
		frames = {}
		asrt = -1
		explore = True
		ending = 0
		ocount = 0
		pcount = 0
		all_ending = False

		best_sd = 0
		best = ()
		MAX_mem = 0
		max_conf = 0
		r_exp = 0

		bc = 0
		nflag = 1
		repeated_blocker = 0
		partition_flag = True

		av = []
		repeat_count= 2*self.k
		M = int(2*self.k/3.0)
		expected_f = 0
		cal_f = 0

		def blocker(sm, i):
			if not sm and i > 0:
				return True
			elif sm and sm.to == -1:
				return True
			else:
				return False

		def ending_explore(i, r_exp=0):
			if explore:
				if (i == repeat_count-1):# == repeat_count -1):
					return True
				if r_exp > 0:
					return False
				if (i > repeat_count and i%M == M -1):
					return True
			if all_ending:
				return True
			return False

		for i in range(4*self.iters):

			print('------Iteration {0} start ------'.format(i))

			print('Total time till now -- starting: ', totalTime, 'repeated_blocker', repeated_blocker)

			''' ----- select engine for iteration i '''
			if i < (repeat_count):
				a = i%(self.k)
				print(i, self.k, 'Initial exploring select', a, Actions[a])
				
				ocount = 0
				pcount = 0
				# explore = False

			else:
				if explore:
					a = self.pull(av, count=r_exp)
					print(i, self.k, M,'Transient exploring select', a, Actions[a], 'already explored', av)
					if ocount == 0:
						av = [a]
					else:
						av.append(a)
					ocount += 1
					pcount = 0
					self.frameout[i] = 0
				else:
					a = self.pull(av, count=-1)
					print(i, self.k, M,'Transient exploiting select', a, Actions[a])
					av = [a]
					ocount = 0
					pcount += 1
			# --- completing engine selection -----
			
			''' --- select time slot for iteration i ---- '''
			if explore: # for exploration
				if i == 0:
					next_timeout = T
						
				else:
					next_timeout = self.timeout[i-1] #* SC
					
					if (i <= repeat_count and i%self.k == 0) or (i > repeat_count and i%M == 0):
						next_timeout = self.timeout[i-1] * SC

					if (blocker(sm, i) and repeated_blocker > 2 and nflag): #or ending_explore(i):
						next_timeout = self.timeout[i-1] * SC
						nflag = 0

					self.frameout[i] = 0 #-1

				self.timeout[i] = min(next_timeout, TIMEOUT - totalTime)	
				# if self.timeout[i] > TIMEOUT - (totalTime + self.timeout[i]):
				# 	ending = 1

				print(i, M, i%M, 'Calculating time out explore', self.timeout[i], next_timeout, 'total till now', totalTime)

			else: # for exploitation
				nflag = 1
				# max_next_to = -1
				next_to, next_fo = -1, -1
				if a in next_time.keys():
					next_to, next_fo = next_time[a]
				# 	if max_next_to < next_to:
				# 		max_next_to = next_to
				# if i > 0:
				next_timeout = self.timeout[i-1] * SC # default

				if next_to > 0: # predicted time 
					self.frameout[i] = self.states + next_fo
					next_timeout = max(next_to, self.timeout[i-1] * SC)

					if blocker(sm,i):
						next_timeout = self.timeout[i-1] * SC
						self.frameout[i] = 0

				# elif not partition_flag and next_to < 0: # no more partition
				# 	next_timeout = (TIMEOUT - totalTime)	
				# 	self.frameout[i] = 0
				# 	all_ending = True
				# 	ending = True

				if expected_f*cal_f > 0 and cal_f/expected_f < 0.6: # no more partition
					next_timeout = self.timeout[i-1]* SC

				# self.timeout[i] = 

				# blocker but time-slots > 900s... no more exploration
				# if blocker(sm, i) and self.timeout[i-1] > 0.25*TIMEOUT: 
					# next_timeout = TIMEOUT - totalTime
					# self.frameout[i] = 0
					# all_ending = True
					# ending = True
				
				self.timeout[i] = min(next_timeout, TIMEOUT - totalTime)	
				# if self.timeout[i] > TIMEOUT - (totalTime + self.timeout[i]):
				# 	ending = 1
				# else:

				print('Calculating time out exploit', next_timeout, 'predicted', next_to, next_fo, 'previous', self.timeout[i-1],'total till now', totalTime)


			# --- completing engine selection ----- #
				
			if int(MAX_TIMEOUT - all_time) <= 0:		
				a = self.pull(av, count=-1)
				self.timeout[i] = (TIMEOUT - totalTime)
				self.frameout[i] = 0
				print('More than {0} hrs spent in learning --- closing iterations now'.format(MAX_TIMEOUT/TIMEOUT))
				all_ending = True
				ending = True
				# break
				# if self.timeout[i] > 3000:
				# 	self.stt

			print('Next time out', self.timeout[i], 'frame_out', self.frameout[i], 'for chosen action', a, Actions[a], 'ocount', ocount, \
				'explore', explore, 'ending', ending)

			if (blocker(sm, i) and repeated_blocker > 5 and self.timeout[i] > TIMEOUT/2.0):
				ending = True
				all_ending = True

			if (self.timeout[i-1]> 0 and self.timeout[i]/self.timeout[i-1] < (100/600.0) and (totalTime + self.timeout[i] - TIMEOUT) < 10.0) \
			     or (self.timeout[i]< 10.0 and repeated_blocker > 5):
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('BMC-depth reached ', self.states, 'totalTime', totalTime)
				print('Stopping iteration - condition with timeout < ', 1.0)
				ending = True
				all_ending = True
				break

			# if self.frameout[i] > 0:
			# 	a, reward, sm = self.update_policy(a, (TIMEOUT - totalTime))
			# else:
			a, reward, sm = self.update_policy(a, self.timeout[i])
			self.write_log(a, sm, reward)

			print('------- ## --------- Starting with depth', str(self.states), ' Ending depth', str(sm.ld))

			if sm and MAX_mem < sm.mem :
				MAX_mem = sm.mem

			# fragmentation
			tt = math.ceil(sm.tt) #sm.tt if sm.asrt > 0 else self.timeout[i] #sm.tt if sm.asrt > 0  else math.ceil(sm.tt) # self.timeout[i]

			tp = self.timeout[i] if (ending and all_ending) else tt #self.timeout[i]
			if self.timeout[i] < sm.tt or self.frameout[i]-1 == sm.ld or self.frameout[i] == sm.ld or sm.asrt > 0:
				tp = math.ceil(sm.tt)

			all_time += self.timeout[i] #sm.tt if sm.asrt > 0 else tp

			sd = self.states
			pre_state = self.states

			expected_f = self.frameout[i]
			cal_f = sm.ld if sm else sd

			if not blocker(sm, i) and reward > 0: # frames unrolled > 0
				# count = 0
				# print(i, 'sm', 'conf', sm.conf, 'cla', sm.cla, max(F*conf_begin_phase, 1e5), 'conf_begin_phase', conf_begin_phase, 'ocount', ocount, 'enter_critical', enter_critical, 'exit_critical', exit_critical, 'critical', critical, 'iter', (i+1)%self.k, 'repeat_count', repeat_count, 'M', M)
				# if not explore: # and sm and reward > 0:
				if (ending_explore(i) or (i < repeat_count and i%self.k == self.k-1)):
					next_time = {}
					self.partition_flag = [1 for i in range(self.k)]
					for a2 in range(self.k):
						next_tm, ndt = self.get_next_time(a2, sm.ld, r_flag =0, flag = 1)
							# next_time.update({new_frames[0]: new_to[0]})
							# print('predicted time out for next frame', new_frames, new_to)
						# else:
						#     next_to = -1
						#if ndt > 0:
						next_time.update({a2:(next_tm, ndt)})
					partition_flag = not(all(ele == 0 for ele in self.partition_flag))
					print('Next time out', next_time)
					print(partition_flag, 'Partitions for engines', self.partition_flag)


				if explore:
					ss = (a, tp, reward, totalTime, self.timeout[i], sd)
					if len(best) == 0:
						best = ss
						best_sd = best[-1]

					sd = sm.ld #Next(sm.ld, a)
					if best_sd < sd:
						best_sd = sd
						best = ss

					elif best_sd == sd:
						if best[1] > sm.tt:
							best_sd = sd
							best = ss
					#best_sd = max(best_sd, sd)
					# if (i < repeat_count):
					max_conf = max(max_conf, sm.cla)

					print('# exploring --', i, 'explore', explore, 'best_sd', best_sd, 'max_conf', max_conf)
					if (ending_explore(i) or (i < repeat_count and i%self.k == self.k-1) or (int(MAX_TIMEOUT - all_time) <= 0 and all_ending) or sm.asrt > 0): #enter_critical and ocount >= M-1) or (i < repeat_count and sm.asrt > 0) or (enter_critical and sm.asrt > 0):
						# end of exploration --- pick the best one
						print('#  at the end of exploration')
						# sd = sm.frame+1 if sm.frame > 0 else sm.frame
						# best_sd = max(best_sd, sd)
						if self.states < best_sd:		
							self.states = best_sd

							totalTime += best[1]
							ss = (best[0], best[1], best[2], totalTime, best[3], best[4])
							#ss = best #(Actions[a], tt, reward, totalTime, self.timeout[i], self.states)
							#print('Total time till now: ', totalTime)
							print('Adding ss -- ending_explore', ss, 'Current state', self.states)
							seq.append(ss)
						# if (i < repeat_count and i%self.k == self.k-1):
						conf_begin_phase = max_conf

				else: #if not explore: # exploitation 
					if expected_f > 0 and cal_f/expected_f >= 0.6:
						self.states = sm.ld
						totalTime += tp
						ss = (a, tp, reward, totalTime, self.timeout[i], sd)
						#print('Total time till now: ', totalTime)
						print('Adding ss -- exploitation', ss)
						seq.append(ss)
						repeated_blocker = 0

			else:
				ss = (a, -1, reward, -1, self.timeout[i], sd)
				repeated_blocker += 1
				if all_ending:
					totalTime += tp
					#ss = (Actions[a], tp, reward, totalTime, self.timeout[i], sd)
					#print('Total time till now: ', totalTime)

			self.reward[i] = self.mean_reward

			print('Total time till now: ', totalTime, 'Current time out ', self.timeout[i])
			# --- state of next run ----
			p = np.random.rand()
			if not ending and not explore and self.timeout[i] < 0.25*TIMEOUT and (partition_flag):
				# if p < self.eps:
				# 	print('random exploration phase')
				# 	r_exp += 1
				# 	explore = True
				# 	ocount = 0
				# 	print('#  Starting exploring --', i, ocount, pcount, r_exp)
				if (blocker(sm,i)) or reward < 0:
					print('blocker -- exploration phase', reward, sm)
					r_exp = 0
					explore = True
					ocount = 0
					print('#  Starting exploring --', i, ocount, pcount, r_exp)

				elif ((sm and next_to > 0 and sm.tt > 0  and (self.frameout[i] - sm.ld) >= 2  and abs(next_to - sm.tt)/sm.tt > 0.75)):
					print('Incorrect prediction of next time -- exploration phase', abs(sm.tt - next_to)/sm.tt)
					r_exp += 1
					explore = True
					ocount = 0
					print('#  Starting exploring --', i, ocount, pcount, r_exp)

				elif (( sm  and (sm.ld - pre_state) < 2)):
					print('current slowing down -- exploration phase',sm.ld, pre_state, self.states)
					r_exp += 1
					explore = True
					ocount = 0
					print('#  Starting exploring --', i, ocount, pcount, r_exp)
				
			# elif not ending and explore and self.timeout[i-1] > 0.25*TIMEOUT: 
			# 	explore = False
			# 	print('#  Ending exploring --', i, ocount, pcount)

			elif ending_explore(i, r_exp):
				explore = False
				print('#  Ending exploring --', i, ocount, pcount)

			if blocker(sm, i) and self.timeout[i-1] > 0.25*TIMEOUT: 
				explore = False
				print('#  Ending exploring --', i, ocount, pcount)
			print()
			print('#### iter ', i, a, Actions[a], 'current state', self.states,'time taken', tt, self.timeout[i], 'totalTime', totalTime, 'ss', ss, sm)

			print('--------- Iteration {0} end ------'.format(i))

			# self.timeout[i] = T
			if sm and sm.asrt > 0:
				asrt = sm.asrt
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('Output asserted at frame ', sm.asrt, 'tt', tt, 'totalTime', totalTime)
				print('Stopping iteration')
				break

			# if ending or int(1.5*MAX_TIMEOUT - all_time) <= 0:		
			# 	end_frame = self.states, asrt, totalTime, seq, MAX_mem
			# 	print('BMC-depth reached ', self.states, 'totalTime', totalTime, 'all_time', all_time)
			# 	print('Stopping iteration -- all timeout')
			# 	all_ending = True
			# 	break
				
			if int(TIMEOUT - totalTime) <= 0:
				end_frame = self.states, asrt, totalTime, seq, MAX_mem
				print('BMC-depth reached ', self.states, 'totalTime', totalTime)
				print('Stopping iteration -- seq timeout')
				all_ending = True
				break

		# while i < self.iters:
		# 	self.reward[i] = self.mean_reward
		# 	i += 1
		
		print(i, 'BMC-depth reached ', self.states, 'totalTime', totalTime)
		print(i, 'Sequence of actions and BMC depth:', seq)

		if not partition_flag:
			print("PARTITION NOT NEEDED FOR DESIGN")
			a = self.pull(av, count=-1)
			self.states = 0
			a, reward, sm = self.update_policy(a, TIMEOUT)
			print("Reached state without Partition : ", sm.ld if sm else self.states)
			# totalTime = sm.tt
			# MAX_mem = sm.mem
			# ss = (a, TIMEOUT, reward, totalTime, self.timeout[i], 0)
			# seq = [ss]
			# end_frame = self.states, sm.asrt, totalTime, seq, MAX_mem

		return end_frame #(sm.frame, sm.asrt, totalTime, seq)
			
	def reset(self, reward):
		# Resets results while keeping settings
		self.n = 0
		self.k_n = np.zeros(self.k)
		self.mean_reward = 0
		self.reward = np.zeros(self.iters)
		self.k_reward = np.zeros(self.k)
		for i in range(self.k):
			self.k_reward[i] = reward
		self.timeout = np.zeros(self.iters)
   
class ucb1_bandit(bandit):
	'''
	Upper Confidence Bound k-bandit problem
	 
	Inputs 
	============================================
	k: number of arms (int)
	c:
	iters: number of steps (int)
	mu: set the average rewards for each of the k-arms.
		Set to "random" for the rewards to be selected from
		a normal distribution with mean = 0. 
		Set to "sequence" for the means to be ordered from 
		0 to k-1.
		Pass a list or array of length = k for user-defined
		values.
	'''
	def __init__(self, k, c, iters, alpha = 1, reward = 0, fname = ''):
		bandit.__init__(self, k, iters, alpha, reward, fname)
		# Exploration parameter
		self.c = c
		self.k_ucb_reward = np.zeros(self.k)	
		self.eps = 0.2
		 
	def pull(self, a1, count = 0):
		# Select action according to UCB Criteria
		c = [self.c for i in range(self.k)]
		
		for i in range(self.k):
			if i in a1:
				c[i] = -1

		self.k_ucb_reward = [self.k_reward[i] + c[i] * np.sqrt((np.log(self.n)) / self.k_n[i]) for i in range(self.k)]
		if count < 0:
			#c = 0
			self.k_ucb_reward = self.k_reward #+ c * np.sqrt((np.log(self.n)) / self.k_n)
		# elif count > 0:
		# 	c = self.c * np.exp(-0.01* count)
		# 	self.k_ucb_reward = self.k_reward + c * np.sqrt((np.log(self.n)) / self.k_n)
		
		a = np.argmax(self.k_ucb_reward)

		print('Action {0} All reward {1}'.format(a, self.k_ucb_reward))
		return a

	def update_policy(self, a, t):
		# Execute the action and calculate the reward
		reward, sm = self.get_reward(a, t) #np.random.normal(self.mu[a], 1)
		
		# Update counts
		self.n += 1
		self.k_n[a] += 1

		# Update total
		self.mean_reward = self.mean_reward + (reward - self.mean_reward) / self.n
		 
		# Update results for a_k
		if self.alpha == 1:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) / self.k_n[a]
		else:
			self.k_reward[a] = self.k_reward[a] + (reward - self.k_reward[a]) * self.alpha

		# self.k_reward[a] = reward

		self.k_ucb_reward = self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n)

		print('Action {0} reward {1} All reward {2}'.format(a, reward, self.k_reward))

		return a, reward, sm

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
	# ar_tab = {}
	# ar_tab1 = {}
	for item in seq:
		a, t = item[0], item[1]
		asrt, sm, ar_tab, tt1 = run_engine(ofname, a, sd, t, f)

		# for ky in ar_tab1:
		# 	ar_tab.update({ky: ar_tab1[ky]})

		if sm:
			sd = int(sm.ld+1) if sm.ld > 0 else int(sm.ld)
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
	return (asrt, 'SAT' if asrt> 0 else 'TIMEOUT', frame, pt if asrt > 0 else tt)#, ar_tab)

def main(argv):
	
	inputfile = ''
	
	try:
		opts, args = getopt.getopt(argv,"hi:", ["ifile="])
	except getopt.GetoptError:
			print("MAB_BMC.py  -i ifile")
			sys.exit(2)
			
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("MAB_BMC.py -i ifile")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
	print("Input file is :" , inputfile)

	fname = (inputfile.split('/')[-1]).split('.')[0]
	print(fname)
	filename = "todaes_results/new_results_MABMC_tm_pred_{0}_{1}.csv".format(TIMEOUT, fname)
	ofname = os.path.join(PATH, (inputfile.split('.')[0])+'_n.'+( inputfile.split('.')[1]))
	# # header = ['Design', 'Frame', 'Clauses', 'Mem', 'time']
	# # writing to csv file 
	# with open(filename, 'w+') as csvfile: 
	# 	print('filename', inputfile)

	k = M # arms
	iters = 10000 #00 #int((TIMEOUT/T)) 
	#iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
	episodes = 1 #episodes
	print('iters', iters)
	alpha = 0.6
	reward = 0
	c = 2.0
	# Initialize bandits

	ucb1 = ucb1_bandit(k, c, iters, 1,  reward, inputfile)

	options = [ucb1]
	labels = [r'ucb1']

	if PLOT:
		pp = PdfPages("todaes_results/new_results_MABMC_tm_pred_{0}.pdf".format(fname)) #, DIFF, '_FIX' if DIFF else ''))
	j = 0
	all_rewards = []
	all_selection = []
	all_results = []
	all_times = []
	for opt in options:

		# starting the monitoring
		tracemalloc.start()
		stime = time.time()
		print('---------------- Running bandit {0} ------------------'.format(labels[j]))
		rewards = np.zeros(iters)
		selection = np.zeros(k)
	
		# Run experiments
		for i in range(episodes):
			
			print('---- episodes ', i)
			# Run experiments
			res = opt.run()

			# Update long-term averages
			rewards = rewards + (opt.reward - rewards) / (i + 1)

			# Average actions per episode
			selection = selection + (opt.k_n - selection) / (i + 1)
		
		all_results.append(res)
		all_rewards.append(rewards)
		all_selection.append(selection)

		if PLOT:
			fig1 = plt.figure(figsize=(12,8))
			plt.plot(rewards, label=labels[j])
			plt.legend(bbox_to_anchor=(1.3, 0.5))
			plt.xlabel("Iterations")
			plt.ylabel("Average Reward")
			plt.title("Average Rewards after " + str(episodes)  + " Episodes")
			plt.legend()
			pp.savefig(fig1)   

		j += 1
		# plt.show() 
		etime = time.time()
		# displaying the memory
		current, peak = tracemalloc.get_traced_memory()
		# stopping the library
		tracemalloc.stop()
		print('time by bandit ', labels[j-1], (etime-stime))
		print('Memory by bandit ', labels[j-1], (current/(1024*1024), peak/(1024*1024)))
		 
		all_times.append((etime-stime, current/(1024*1024), peak/(1024*1024)))
	print('-------------------------------------------')
	print()
	print('Bandit policy: \t BMC depth \t time \t sequence')

	rows  = []
	
	rows.append(['Design', 'Bandit-policy','BMC-depth', 'status', 'time(s)', 'total(s)', 'memory-current(MB)', 'memory-peak MB)', 'MAX-mem(BMC)(MB)', 'sequence'])
	print('Bandit policy: \t BMC depth \t time \t sequence')
	
	all_plots = []
	if PLOT:
		fig2 = plt.figure()
	for j in range(len(options)):
		d, a, t, seq, mem = all_results[j]
		print('{0}: \t {1} ({4}) \t time: {2:0.2f} s, real: {5:0.2f}s, Memory: {6:0.2f}MB,{7:0.2f}MB {8}MB \t {3} '.format(labels[j], a if a > 0 else d, t, seq, 'assert' if a>0 else '', all_times[j][0], all_times[j][1], all_times[j][2], mem))
		rows.append([fname, labels[j], a if a > 0 else d, 'sat' if a>0 else 'timeout', '{0:0.2f}'.format(t), '{0:0.2f}'.format(all_times[j][0]), '{0:0.2f}'.format(all_times[j][1]), mem, '{0:0.2f}'.format(all_times[j][2]), seq])

		if PLOT:
			plt.plot(all_rewards[j], label=labels[j])
			plt.legend(bbox_to_anchor=(1.3, 0.5))

		to_plot = [[],[]]
		res_seq = []
		total_t = 0
		ac = 0
		for ss in seq:
			ac, tp, rw, tt, t, frame =  ss
			to_plot[0].append(frame)
			to_plot[1].append(rw)
			ftt = min(TIMEOUT - total_t, math.ceil(tp))
			res_seq.append((int(ac), math.ceil(ftt)))
			total_t += ftt

		if total_t < TIMEOUT:
			ftt = (TIMEOUT - total_t)
			res_seq.append((int(ac), math.ceil(ftt)))
			total_t += ftt

		all_plots.append(to_plot)

		seq_list = ['({0}, {1})'.format(Actions[t[0]], t[1]) for t in seq]
		sequence = ';'.join(seq_list)
		print('Optimal sequence:', sequence)
		r_as1, r_cond1, r_sd1, r_tt1 = runseq(ofname, res_seq)
		r_w1 = max(0, TIMEOUT - r_tt1)
		r_tot1 = r_tt1 if r_as1 > 0 else TIMEOUT

		res_str = 'Sequence execution result: {0}  {1:0.2f}  {2:0.2f}\t'.format(r_sd1, r_tot1, r_w1) 
		print(res_str)

	if PLOT:
		plt.xlabel("Iterations")
		plt.ylabel("Average Reward")
		plt.title("Average Rewards after " + str(episodes)  + " Episodes")
		plt.legend()
		pp.savefig(fig2)

		fig3 = plt.figure()
		for j in range(len(options)):
			to_plot = all_plots[j]
			plt.plot(to_plot[0], to_plot[1], label=labels[j])

		plt.xlabel("Frames")
		plt.ylabel("Reward")
		# plt.title("Average Rewards after " + str(episodes)  + " Episodes")
		plt.legend()
		pp.savefig(fig3)

		pp.close()

	opt_per = np.array(all_selection)/ iters * 100
	df = pd.DataFrame(opt_per, index=labels, columns=[Actions[x] for x in range(0, k)])
	print("Percentage of actions selected:")
	print(df)
	

	with open(filename, 'a+') as csvfile: 
		csvwriter = csv.writer(csvfile) 
		csvwriter.writerows(rows)

	
if __name__ == "__main__":
	start_time = time.time()
	main(sys.argv[1:])
	end_time = time.time()
	print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')

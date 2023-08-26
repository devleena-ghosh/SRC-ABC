import numpy as np 
PLOT = True
if PLOT:
	import matplotlib.pyplot as plt 
	from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math, csv
from collections import namedtuple
from enum import Enum
from scipy.interpolate import CubicSpline, interp1d
import tracemalloc
import random
from sklearn.neural_network import MLPRegressor
from abcBMCUtil import *

flag = 1
r_flag = 1
X = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
y = [1553, 7649, 16555, 32582, 55350, 77868, 103984, 142791, 191351, 267887, 312036, 363544, 430200, 495049, 562708, 630413, 734670, 825952, 968899, 1085004, 1179526, 1305827, 1429115, 1571103, 1681698, 1797865, 1901039, 2035218, 2164742]

X1 = X[0:-2]
y1 = y[0:-2]

fcla = interp1d(X1, y1, fill_value = 'extrapolate')
cscla = CubicSpline(X1, y1)

next_frm = np.arange(X[-1], X[-1]+5, 1)

ncla1 = fcla(next_frm)
ncla2 = cscla(next_frm)

print('original :', y[-1], 'predicted', ncla1, ncla2)
fig = plt.figure()
plt.plot(X1, y1, 'b-')
plt.plot(X, y, 'g--')
plt.plot(next_frm, ncla1, 'ro')
plt.plot(next_frm, ncla2, 'k*')
plt.show()

output = \
' 12 + : Var =   13697. Cla =    52853. Conf =      0. Learn =      0.    1 MB   0 MB     0.09 sec \
  13 + : Var =   22316. Cla =    87847. Conf =      0. Learn =      0.    1 MB   0 MB     0.11 sec \
  14 + : Var =   40584. Cla =   166090. Conf =   1553. Learn =   1534.    1 MB   0 MB     0.30 sec \
  15 + : Var =   54699. Cla =   228025. Conf =   7649. Learn =   3847.    2 MB   0 MB     1.02 sec \
  16 + : Var =   69015. Cla =   291542. Conf =  16555. Learn =   7753.    2 MB   0 MB     2.48 sec \
  17 + : Var =   83829. Cla =   357755. Conf =  32582. Learn =  16333.    2 MB   0 MB     5.32 sec \
  18 + : Var =   99935. Cla =   429795. Conf =  55350. Learn =  16767.    2 MB   0 MB     9.71 sec \
  19 + : Var =  115640. Cla =   500488. Conf =  77868. Learn =  24289.    2 MB   0 MB    14.17 sec \
  20 + : Var =  131373. Cla =   571382. Conf = 103984. Learn =  32816.    2 MB   0 MB    19.71 sec \
  21 + : Var =  147108. Cla =   642335. Conf = 142791. Learn =  28633.    2 MB   0 MB    27.83 sec \
  22 + : Var =  162851. Cla =   713320. Conf = 191351. Learn =  51728.    2 MB   0 MB    38.45 sec \
  23 + : Var =  178632. Cla =   784458. Conf = 250424. Learn =  52266.    2 MB   0 MB    51.37 sec \
  24 + : Var =  194415. Cla =   856559. Conf = 284690. Learn =  53307.    2 MB   0 MB    53.57 sec \
  25 + : Var =  210268. Cla =   927967. Conf = 338865. Learn =  71702.    2 MB   0 MB    61.77 sec \
  26 + : Var =  226134. Cla =   999442. Conf = 418371. Learn =  71805.    3 MB   0 MB    74.38 sec \
  27 + : Var =  242004. Cla =  1070951. Conf = 482873. Learn =  46698.    3 MB   0 MB    84.52 sec \
  28 + : Var =  257881. Cla =  1142516. Conf = 563219. Learn =  78137.    3 MB   0 MB    96.30 sec \
  29 + : Var =  273758. Cla =  1214117. Conf = 642900. Learn = 106903.    3 MB   0 MB   107.50 sec \
  30 + : Var =  289635. Cla =  1285724. Conf = 744877. Learn =  99554.    3 MB   0 MB   122.38 sec \
  31 + : Var =  305512. Cla =  1357331. Conf = 857171. Learn =  93495.    3 MB   0 MB   140.39 sec \
  32 + : Var =  321389. Cla =  1428938. Conf =1008267. Learn = 117148.    3 MB   0 MB   169.98 sec \
  33 + : Var =  337266. Cla =  1500545. Conf =1147375. Learn = 119040.    3 MB   0 MB   196.44 sec \
  34 + : Var =  353143. Cla =  1572152. Conf =1313299. Learn = 120590.    3 MB   0 MB   227.44 sec \
  35 + : Var =  369020. Cla =  1643759. Conf =1456326. Learn = 176492.    3 MB   0 MB   253.11 sec \
  36 + : Var =  384897. Cla =  1715366. Conf =1629737. Learn = 155912.    4 MB   0 MB   289.13 sec \
  37 + : Var =  400774. Cla =  1786973. Conf =1791065. Learn = 208315.    4 MB   0 MB   318.87 sec \
  38 + : Var =  416597. Cla =  1858582. Conf =1681698. Learn = 246738.    4 MB   0 MB   534.95 sec \
  39 + : Var =  432480. Cla =  1930213. Conf =1797865. Learn = 264155.    4 MB   0 MB   562.56 sec \
  40 + : Var =  448363. Cla =  2001844. Conf =1901039. Learn = 258989.    4 MB   0 MB   585.71 sec \
  41 + : Var =  464240. Cla =  2073451. Conf =2035218. Learn = 269733.    4 MB   0 MB   622.00 sec \
  42 + : Var =  480117. Cla =  2145058. Conf =2164742. Learn = 272528.    4 MB   0 MB   652.77 sec \
  43 + : Var =  495994. Cla =  2216665. Conf =2302554. Learn = 278942.    4 MB   0 MB   687.66 sec \
  44 + : Var =  511871. Cla =  2288272. Conf =2444789. Learn = 285458.    4 MB   0 MB   719.05 sec \
  45 + : Var =  527748. Cla =  2359879. Conf =2618681. Learn = 317200.    4 MB   0 MB   760.06 sec \
  46 + : Var =  543625. Cla =  2431486. Conf =2788313. Learn = 345231.    4 MB   0 MB   798.33 sec \
  47 + : Var =  559502. Cla =  2503093. Conf =2981311. Learn = 385400.    5 MB   0 MB   846.21 sec \
  48 + : Var =  575379. Cla =  2574700. Conf =3186613. Learn = 276058.    5 MB   0 MB   898.41 sec \
  49 + : Var =  591256. Cla =  2646307. Conf =3389946. Learn = 316618.    5 MB   0 MB   951.96 sec \
  50 + : Var =  607133. Cla =  2717914. Conf =3627904. Learn = 384684.    5 MB   0 MB  1012.25 sec \
  51 + : Var =  623010. Cla =  2789521. Conf =3930492. Learn = 341258.    5 MB   0 MB  1090.13 sec \
  52 + : Var =  638887. Cla =  2861128. Conf =4232765. Learn = 455140.    5 MB   0 MB  1167.23 sec \
  53 + : Var =  654764. Cla =  2932735. Conf =4561905. Learn = 409345.    5 MB   0 MB  1257.73 sec \
  54 + : Var =  670641. Cla =  3004342. Conf =4933835. Learn = 385392.    5 MB   0 MB  1365.16 sec \
  55 + : Var =  686518. Cla =  3075949. Conf =5268079. Learn = 506036.    5 MB   0 MB  1459.90 sec \
  56 + : Var =  702395. Cla =  3147556. Conf =5665342. Learn = 479696.    5 MB   0 MB  1579.28 sec \
  57 + : Var =  718272. Cla =  3219163. Conf =6117702. Learn = 492444.    6 MB   0 MB  1725.00 sec \
  58 + : Var =  734149. Cla =  3290770. Conf =6521238. Learn = 448179.    6 MB   0 MB  1851.02 sec \
  59 + : Var =  750026. Cla =  3362377. Conf =6891187. Learn = 567711.    6 MB   0 MB  1971.24 sec \
  60 + : Var =  765903. Cla =  3433984. Conf =7361647. Learn = 556632.    6 MB   0 MB  2132.25 sec \
  61 + : Var =  781780. Cla =  3505591. Conf =7763397. Learn = 710886.    6 MB   0 MB  2269.26 sec \
  62 + : Var =  797657. Cla =  3577198. Conf =8222745. Learn = 646785.    6 MB   0 MB  2425.21 sec \
  63 + : Var =  813534. Cla =  3648805. Conf =8708906. Learn = 611663.    6 MB   0 MB  2586.53 sec \
  64 + : Var =  829411. Cla =  3720412. Conf =9230450. Learn = 590710.    6 MB   0 MB  2764.01 sec \
  65 + : Var =  845288. Cla =  3792019. Conf =9875990. Learn = 663158.    6 MB   0 MB  2997.66 sec \
  66 + : Var =  861165. Cla =  3863626. Conf =10413312. Learn = 623618.    6 MB   0 MB  3188.55 sec\
Runtime:  CNF = 2.4 sec (0.1 %)  UNSAT = 3186.1 sec (99.6 %)  SAT = 0.0 sec (0.0 %)  UNDEC = 9.0 sec (0.3 %)\
LStart(P) = 0  LDelta(Q) = 0  LRatio(R) = 0  ReduceDB = 0  Vars = 877042  Used = 0 (0.00 %)\
Buffs = 4446. Dups = 0.   Hash hits = 1829.  Hash misses = 865019.  UniProps = 0.\
No output asserted in 0 frames. Resource limit reached (timeout 3197 sec). Time =  3197.79 sec\
[1;37m../benchmark/HWMCC_15_17/6s343b31_n:[0m i/o =  205/    1  lat = 4603  and =  41858  lev = 47'

asrt, sm, ar_tab, tt1 = parse_bmc3(output,t=3600)
print(ar_tab)
nd = 5
ftrain, ctrain, conftrain, ttrain = [], [], [], []
#frm = self.states
frm  = -1
prev = 0, 0, 0, 0
pre_dif = 1.0
partition_flag = 0
for frm in ar_tab.keys():
	sm1 = ar_tab[frm]
	if sm1.cla > 0 and (sm1.cla not in ctrain): # and (sm1.tt - prev[3] > 0):
		diff = sm1.tt-prev[-1]
		# if prev[0] <= sm1.frame-1 and prev[-1] > sm1.tt:
		# 	partition_flag = 0
		# 	# print('current ', sm1.frame, sm1.cla, sm1.conf, sm1.tt)
		# 	# print('prev', prev)	
		# 	ftrain, ctrain, conftrain, ttrain = [], [], [], []
		# if prev[0] == sm1.frame-1 and (prev[-1]> 5.0 and (sm1.tt/prev[-1] > 10.0 or (diff/pre_dif > 10.0))) and \
		# 	(prev[-1]> 60.0 and (sm1.tt/prev[-1] > 1.1 or (diff/pre_dif > 1.2))):
		# 	partition_flag = 1
		# if flag and prev[-1]> 60.0: #and r_flag == 0 
		# 	print(Actions[a], 'current (', sm1.frame, sm1.cla, sm1.conf, sm1.tt,  ') prev', prev, sm1.tt/prev[-1], partition_flag)	
		ftrain.append(sm1.frame)
		ctrain.append(sm1.cla)# - prev[1])
		conftrain.append(sm1.conf)# - prev[2])
		ttrain.append(sm1.tt)# - prev[3])
		prev = sm1.frame, sm1.cla, sm1.conf, sm1.tt

		sys.stdout.flush()
sm1 = ar_tab[frm]
last_frm = frm
last_cla = sm1.cla #ar_tab[frm].cla
last_tm = sm1.tt

fcla = interp1d(ftrain, ctrain, fill_value = 'extrapolate')
# predict number of new conflict clauses from new cluases then predict time to be taken for next frame
fconf = interp1d(ctrain, conftrain, fill_value = 'extrapolate')
fto = interp1d(conftrain, ttrain, fill_value = 'extrapolate')

## inverse prediction
# predict number of frames solved in a given timeout
# ifconf = interpolate.interp1d(ttrain, conftrain, fill_value = 'extrapolate')
# ifcls = interpolate.interp1d(conftrain, ctrain, fill_value = 'extrapolate')
# iffrm = interpolate.interp1d(ctrain, ftrain, fill_value = 'extrapolate')
new_frames = np.arange(last_frm+1, last_frm+int(nd)+1, 2) #if int(1 + nd/2) > 2 else [last_frm+int(nd)+1]
new_cla = fcla(new_frames)
new_conf = fconf(new_cla)
# changed on 10/08
#new_to = fto(new_cla)
new_to = fto(new_conf)

fig = plt.figure()
plt.subplot(2,2,1)
plt.plot(ftrain, ttrain, 'b-')
# plt.plot(X, y, 'g--')
# plt.plot(new_frames, new_to, 'ro')

plt.subplot(2,2,2)
plt.plot(ftrain, ctrain, 'b-')
# plt.plot(X, y, 'g--')
# plt.plot(new_frames, new_cla, 'ro')

plt.subplot(2,2,3)
plt.plot(ctrain, conftrain, 'b-')
# plt.plot(X, y, 'g--')
# plt.plot(new_cla, new_conf, 'ro')

plt.subplot(2,2,4)
plt.plot(conftrain, ttrain, 'b-')
# plt.plot(X, y, 'g--')
# plt.plot(new_conf, new_to, 'ro')
plt.show()
import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
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
MAX_CLAUSE = 1e10
MAX_TIME = 1800

def runseq(fn, seq):

	fname = os.path.join(PATH, fn)
	#ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
	ofname = os.path.join(PATH, (fn.split('.')[0])+'_n.'+(fn.split('.')[1]))
	print('----Running sequence {0} ----'.format(seq),fn, fname)
	
	#starting state
	sd = 0
	asrt = -1
	sm = None
	for item in seq:
		a, t = item
		if a == 0:    #ABC bmc2
			asrt, sm = bmc2(ofname, sd, t)
		elif a == 1: #ABC bmc3
			asrt, sm = bmc3(ofname, sd, t)
		elif a == 2: #ABC bmc3s
			asrt, sm = bmc3s(ofname, sd, t)
		elif a == 3: #ABC bmc3j
			asrt, sm = bmc3j(ofname, sd, t)
		elif a == 4: #ABC bmc3az
			asrt, sm = bmc3az(ofname, sd, t)
		elif a == 5: #ABC bmc3x
			asrt, sm = bmc3x(ofname, sd, t)
            
		if sm:
			sd = sm.frame+1 if sm.frame > 0 else sm.frame
		else:
			sm =  abc_result(frame=sd-1, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt)
		print(sm)
		sys.stdout.flush()
		if asrt > 0:
			print('Output asserted at frame ', asrt)
			print('Stopping iteration')
			break

def main(argv):
    
    inputfile = ''
    
    try:
        opts, args = getopt.getopt(argv,"hi:", ["ifile="])
    except getopt.GetoptError:
            print("All_seq.py  -i ifile")
            sys.exit(2)
            
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("All_seq.py -i ifile")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    print("Input file is :" , inputfile)

    k = 5 # arms
    iters = 4 #int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
    TIMEOUT = 3600
    T = int(TIMEOUT*(SC-1)/(SC**iters - 1))
    print('iters', iters, T)
    
    seq = [(), (), (), ()]
    for i1 in range(5):
        seq[0] = (i1, T)
        for i2 in range(5):
            seq[1] = (i2, T*SC)
            for i3 in range(5):
                seq[2] = (i3, T*SC*SC)
                for i4 in range(5):
                    seq[3] = (i4, T*SC*SC*SC)
                    runseq(inputfile, seq)
			
    


    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')

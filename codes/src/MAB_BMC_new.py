import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd 
import os, sys, subprocess, getopt, gc, time, re, math
from collections import namedtuple
from enum import Enum
from scipy import interpolate
# from memory_profiler import memory_usage
from abcBMCUtil import *

DEBUG = True
DEBUG = False
OPT = True
T = 30 
TIMEOUT = 3600
SC = 2
DIFF = 1 # BMC depth absolute
DIFF = 0 # BMC depth relative
DIFF = 2 # function of depth, time, memory
# DIFF = 3 # function number of clauses/time diff
#DIFF = Enum ('DIFF', ['F1', 'F2', 'F3'])
FIXED = False
FIXED = True
PDR = False
MAX_FRAME = 1e4
MAX_CLAUSE = 1e9
MAX_TIME = 3600

time_outs = {}
Actions = ['bmc2', 'bmc3', 'bmc3rs', 'bmc3j', 'bmc3rg', 'bmcru', 'bmc3r', 'pdr']

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

        for i in range(self.k):
            self.k_reward[i] = reward

        # recency constant
        self.alpha = alpha
        ##
        self.fname =  fname
        self.timeout = np.zeros(iters)


        # current state of the run; eg - bmc depth
        self.states = 0

    def get_reward(self, a, t1 = -1):

        # get starting depth
        # sd = 0
        # if self.n  == 0:
        sd = int(self.states) #[a])

        fname = os.path.join(PATH, self.fname)
        #ofname = os.path.join(PATH,(self.fname.split('/')[-1]),'new', ((self.fname.split('/')[-1]).split('.')[0])+'_n.'+( ((self.fname.split('/')[-1])).split('.')[1]))
        ofname = os.path.join(PATH, (self.fname.split('.')[0])+'_n.'+( self.fname.split('.')[1]))
        if DEBUG or self.n == 0:
            print(fname, ofname)

        st = ''

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
        if a == 0:    #ABC bmc2
            asrt, sm, ar_tab = bmc2(ofname, sd, t)
        elif a == 1: #ABC bmc3
            asrt, sm, ar_tab = bmc3(ofname, sd, t)
        elif a == 2: #ABC bmc3rs
            asrt, sm, ar_tab = bmc3rs(ofname, sd, t)
        elif a == 3: #ABC bmc3j
            asrt, sm, ar_tab = bmc3j(ofname, sd, t)
        elif a == 4: #ABC bmc3g
            asrt, sm, ar_tab = bmc3rg(ofname, sd, t)
        elif a == 5: #ABC bmc3u
            asrt, sm, ar_tab = bmc3ru(ofname, sd, t)
        elif a == 6: #ABC bmc3r
            asrt, sm, ar_tab = bmc3r(ofname, sd, t)
        elif a == 7: #ABC pdr
            asrt, sm, ar_tab = pdr(ofname, t)

        if sm is not None:
            # print(sm)
            reward = 0
            if DIFF == 0 :
                reward = (sm.frame - sd)
            elif DIFF == 1:
                reward = sm.frame
            elif DIFF == 2:
                pen = t - sm.tt
                reward =  np.exp(0.5*sm.frame/MAX_FRAME - 0.3*sm.tt/MAX_TIME)
                # + 0.1*sm.cla/MAX_CLAUSE 
                #np.exp(0.5*sm.frame/MAX_FRAME + 0.1*sm.cla/MAX_CLAUSE - 0.3*sm.to/MAX_TIME) #-  0.1*pen/MAX_TIME)
                if asrt > 0:
                    reward =  np.exp(1 - 0.2*sm.frame/MAX_FRAME - 0.3*sm.tt/MAX_TIME)
            else:
                reward = sm.cla/(10000 * sm.to) if sm.to > 0 else sm.to
        else:
            sm =  abc_result(frame=sd, conf=0, var=0, cla=0, mem = -1, to=-1, asrt = asrt, tt = t)
            if asrt > 0:
                reward = asrt
            else:
                reward = -0.5 * np.exp(-0.3*t/MAX_TIME) #np.log(t)
        print(reward, sm)

        return reward, sm, ar_tab


    def run(self):
        totalTime = 0
        seq = []
        sm = None
        next_to = -1
        next_time = {}
        # for i in self.iters:
        #     next_time[i] = -1
        for a in range(self.k):
            time_outs.update({a:next_time})
        frames = {}
        count = 0
        for i in range(self.iters):
            if totalTime >= TIMEOUT:
                print('BMC-depth reached ', self.states, 'totalTime', totalTime)
                print('Stopping iteration')
                break
            
            a = self.pull()
            if i < 2*(self.k):
                a = i%(self.k)

            # if a in time_outs:
            #     nextt = time_outs[a]
            #     if self.states in nextt:
            #         next_to = nextt[self.states]

            if i == 0:
                self.timeout[i] = T
            else:
                if FIXED:
                    if next_to == -1:
                        if sm and sm.to == -1 and count > 2:
                            self.timeout[i] = self.timeout[i-1] * SC 
                            count = 0
                        else:
                            self.timeout[i] = self.timeout[i-1]
                            count = count + 1
                    else:
                        try:
                            if sm and sm.to == -1 and count > 2:
                                self.timeout[i] = max(math.ceil(next_to), self.timeout[i-1]*SC, 480)
                                count = 0
                            else:
                                self.timeout[i] = max(math.ceil(next_to), self.timeout[i-1])
                                count = count + 1

                        except OverflowError:
                            self.timeout[i] = self.timeout[i-1]
                        except ValueError:
                            self.timeout[i] = self.timeout[i-1]
                else:
                    if next_to == -1:
                        self.timeout[i] = self.timeout[i-1] 
                    else:
                        try:
                            if sm and sm.to == -1:
                                self.timeout[i] = max(math.ceil(next_to), self.timeout[i-1]*SC, 480)
                            else:
                                self.timeout[i] = max(math.ceil(next_to), self.timeout[i-1])
                        except OverflowError:
                            self.timeout[i] = self.timeout[i-1]
                        except ValueError:
                            self.timeout[i] = self.timeout[i-1]
            self.timeout[i] = min(self.timeout[i], TIMEOUT - totalTime)

            print('Next time out', self.timeout[i], next_to)

            a, reward, sm, ar_tab = self.update_policy(a, self.timeout[i])

            tt = sm.tt if sm.asrt > 0  else self.timeout[i]

        
            if sm and reward > 0:
                ss = (Actions[a], tt, reward, totalTime, self.states)
                seq.append(ss)
                if i > 2*self.k or i%self.k == self.k-1:
                    self.states = sm.frame+1 if sm.frame > 0 else sm.frame
            else:
               ss = (a, -1, reward, -1, self.states)
            
            self.reward[i] = self.mean_reward

            if sm and sm.to > 0 and reward > 0:
                totalTime += tt
                
            print('#### iter ', i, Actions[a], 'time taken', tt, self.timeout[i], 'totalTime', totalTime, 'ss', ss, sm)

            # self.timeout[i] = T
            if sm and sm.asrt > 0:
                print('Output asserted at frame ', sm.asrt, 'tt', tt, 'totalTime', totalTime)
                print('Stopping iteration')
                break

            if a in frames:
                ftrain, ttrain, ttrain1 = frames[a]
            else:
                ftrain, ttrain,ttrain1 = [], [], []
            for frm in ar_tab.keys():
                sm1 = ar_tab[frm]
                if sm1.conf > 0:
                    ftrain.append(sm1.frame)
                    ttrain.append(sm1.cla)
                    ttrain1.append(sm1.to)
            frames.update({a:(ftrain, ttrain, ttrain1)})
            # if len(frames) > 10:
            #     ftrain, ttrain = frames[-11:], time_outs[-11:]
            # else:
            if sm.conf > 1e6:
                # for a in range(self.k):
                    # if a in frames:
                    #     ftrain, ttrain = frames[a]
                    # else:
                    #     ftrain, ttrain = [], []
                print('Training for action', a, len(ftrain), len(ttrain))
                if len(ftrain) > 0:
                    # predict number of clauses then predict timeout for next frame
                    fcla = interpolate.interp1d(ftrain, ttrain, fill_value = 'extrapolate')
                    fto = interpolate.interp1d(ttrain, ttrain1, fill_value = 'extrapolate')
                    new_frames = np.arange(ftrain[-1]+1, ftrain[-1]+2, 1)
                    new_cla = fcla(new_frames)
                    new_to = fto(new_cla)
                    next_to = np.sum(new_to)
                    next_time.update({new_frames[0]: new_to[0]})
                    # print('predicted time out for next frame', new_frames, new_to)
                # else:
                #     next_to = -1
                time_outs.update({a:next_time})
                # print('next time out', next_to)

        while i < self.iters:
            self.reward[i] = self.mean_reward
            i += 1
        print('BMC-depth reached ', sm.frame, 'totalTime', totalTime)
        print('Sequence of actions and BMC depth:', seq)
        return (sm.frame, sm.asrt, totalTime, seq)
            
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


class eps_bandit(bandit):
    '''
    epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, iters, alpha = 1, reward = 0,fname = ''):
        bandit.__init__(self, k, iters, alpha, reward, fname)
        # Search probability
        self.eps = eps
        
    def pull(self):
        # Generate random number
        # np.random.seed(time.time())
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            ap = [1,2,4]
            a = np.random.choice(ap)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
        return a

    def update_policy(self, a, t):
        # Execute the action and calculate the reward
        # mem_use, tm = memory_usage(self.get_reward(a))
        reward, sm, ar_tab = self.get_reward(a, t)
        
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

        # if a >0:
        #print('For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
        return a, reward, sm, ar_tab
        
   
class eps_decay_bandit(bandit):
    '''
    epsilon-decay k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, iters, alpha = 1, reward = 0, fname = ''):
        bandit.__init__(self, k, iters, alpha, reward, fname)
       
    def pull(self):
        # Generate random number
        # np.random.seed(self.n)
        p = np.random.rand()
        if self.n == 0:
            ap = [1,2,4]
            a = np.random.choice(ap)
        elif p < 1 / (1 + self.n / self.k):
            # Randomly select an action
            a = np.random.choice(self.k)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
        return a

    def update_policy(self, a, t):
        # Execute the action and calculate the reward
        # mem_use, tm = memory_usage(self.get_reward(a))
        reward, sm, ar_tab = self.get_reward(a, t)
        
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

        # if a >0:
        # print('For action {0} reward {1}, updated reward {2}'.format(a, reward, self.k_reward))
        return a, reward, sm, ar_tab
       
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
         
    def pull(self):
        # Select action according to UCB Criteria
        a = np.argmax(self.k_ucb_reward)
        return a

    def update_policy(self, a, t):
        # Execute the action and calculate the reward
        reward, sm, ar_tab = self.get_reward(a, t) #np.random.normal(self.mu[a], 1)
        
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
        
        self.k_ucb_reward = self.k_reward + self.c * np.sqrt((np.log(self.n)) / self.k_n)
        return a, reward, sm, ar_tab
    
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

    k = 7 # arms
    iters = 1000 #int((TIMEOUT/T)) 
    #iters = int(np.log((TIMEOUT/T)*(SC-1) +1)/(np.log(SC))) + 1 # time-steps
    episodes = 1 #episodes
    print('iters', iters)
    alpha = 0.05
    reward = 0
    # Initialize bandits
    eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile)
    eps_01 = eps_bandit(k, 0.4, iters, 1, reward,  inputfile)
    eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile)
    eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile)
    eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile)
    ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile)

    reward = 6000
    o_eps_0 = eps_bandit(k, 0.0, iters,1, reward, inputfile)
    o_eps_01 = eps_bandit(k, 0.4, iters, 1, reward,  inputfile)
    o_eps_1 = eps_bandit(k, 0.1, iters, 1,  reward, inputfile)
    o_eps_1_alpha = eps_bandit(k, 0.1, iters, alpha,  reward, inputfile)
    o_eps_decay = eps_decay_bandit(k, iters,1, reward, inputfile)
    o_ucb1 = ucb1_bandit(k, 2, iters,1,  reward, inputfile)

    options = [eps_0, eps_01, eps_1, eps_1_alpha, eps_decay, ucb1, o_eps_0, o_eps_01, o_eps_1, o_eps_1_alpha, o_eps_decay, o_ucb1]
    labels = [r'$\epsilon=0$', r'$\epsilon=0.4$', r'$\epsilon=0.1$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha), r'$\epsilon-decay$', 'ucb1', \
    r'opt $\epsilon=0$', r'opt $\epsilon=0.01$', r'opt $\epsilon=0.1$', r'opt $\epsilon=0.1, \alpha = {0}$'.format(alpha), \
        r'opt $\epsilon-decay$', r'opt ucb1']

    options = [ eps_01, eps_1_alpha, eps_decay, ucb1, o_eps_01]
    labels = [ r'$\epsilon=0.4$', r'$\epsilon=0.1, \alpha = {0}$'.format(alpha), r'$\epsilon-decay$', 'ucb1', r'opt $\epsilon=0.01$' ]

    fname = (inputfile.split('/')[-1]).split('.')[0]
    print(fname)
    pp = PdfPages("plots/plot_MAB_BMC_all_{0}_{1}{2}.pdf".format(fname, DIFF, '_FIX' if DIFF else ''))
    j = 0
    all_rewards = []
    all_selection = []
    all_results = []
    for opt in options:

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

        fig1 = plt.figure(figsize=(12,8))
        plt.plot(rewards, label=labels[j])
        plt.legend(bbox_to_anchor=(1.3, 0.5))
        plt.xlabel("Iterations")
        plt.ylabel("Average Reward")
        plt.title("Average Rewards after " + str(episodes)  + " Episodes")
        plt.legend()

        j += 1
        # plt.show()

        pp.savefig(fig1)   
        etime = time.time()
        print('time by bandit ', labels[j], (etime-stime)/3600.0)
    print('-------------------------------------------')
    print()
    print('Bandit policy: \t BMC depth \t time \t sequence')
    fig2 = plt.figure(figsize=(12,8))
    for j in range(len(options)):
        d, a, t, s = all_results[j]
        print('{0}: \t {1} ({4}) \t {2} s \t {3}'.format(labels[j], a if a > 0 else d, t, s, 'assert' if a>0 else ''))
        plt.plot(all_rewards[j], label=labels[j])
        plt.legend(bbox_to_anchor=(1.3, 0.5))
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards after " + str(episodes)  + " Episodes")
    plt.legend()
    pp.savefig(fig2)

    opt_per = np.array(all_selection)/ iters * 100
    df = pd.DataFrame(opt_per, index=labels, columns=[Actions[x] for x in range(0, k)])
    print("Percentage of actions selected:")
    print(df)
    
    pp.close()


    
if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv[1:])
    end_time = time.time()
    print('Code Running duration: ', (end_time - start_time)/3600, ' hrs')

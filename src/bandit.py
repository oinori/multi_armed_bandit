# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

class MultiarmedBandit:

    def __init__(self, arm_probs):
        self.arm_probs = arm_probs
        self.num_arms = len(arm_probs)
        self.num_trials = np.array([0]*self.num_arms)
        self.num_hits = np.array([0]*self.num_arms)
        self.hit_ratios  = np.array([0.0]*self.num_arms)
        self.total_trial = 0

    def draw(self, arm):

        if arm > self.num_arms-1:
            print "no arm!!"
            return

        res = np.random.binomial(n=1, p=self.arm_probs[arm])
        if res == 1:
            self.num_hits[arm] += 1
        self.num_trials[arm] += 1
        self.hit_ratios[arm] = 1.0*self.num_hits[arm]/self.num_trials[arm]
        self.total_trial += 1
        return res

    def print_num_trials(self):
        print "number of trials: %s" %self.num_trials

    def print_num_hits(self):
        print "number of hits: %s" %self.num_hits

    def print_regret(self):
        total_trial = self.total_trial
        total_hits = sum(self.num_hits)
        print "regret: %s" %(max(self.arm_probs)*total_trial - total_hits)

    def get_regret(self):
        total_trial = self.total_trial
        total_hits = sum(self.num_hits)
        return max(self.arm_probs)*total_trial - total_hits

    def reset_trial(self):
        self.num_trials = [0]*self.num_arms
        self.num_hits = [0]*self.num_arms
        self.total_trial = 0

def execute_eps_greedy(eps, MB, T):

    for arm in range(MB.num_arms):
        MB.draw(arm)

    for t in range(T-MB.num_arms):
        coin_flip = np.random.rand()
        if coin_flip >= eps:
            arm = np.argmax(MB.hit_ratios)
        else:
            arm = np.random.choice(range(MB.num_arms))
        MB.draw(arm)

def execute_UCB1(MB, T):

    for arm in range(MB.num_arms):
        MB.draw(arm)

    for t in range(T-MB.num_arms):
        UCB1_index = MB.hit_ratios + np.sqrt((2*np.log(t+MB.num_arms))/MB.num_trials)
        arm = np.argmax(UCB1_index)
        MB.draw(arm)

def execute_Thompson_sampling(MB, T):
    alphas = np.ones(MB.num_arms)
    betas = np.ones(MB.num_arms)

    for arm in range(MB.num_arms):
        res = MB.draw(arm)
        if res == 1:
            alphas[arm] += 1
        else:
            betas[arm] += 1

    for t in range(T-MB.num_arms):
        thetas = [np.random.beta(a=alphas[i], b=betas[i]) for i in range(MB.num_arms)]
        arm = np.argmax(thetas)
        res = MB.draw(arm)
        if res == 1:
            alphas[arm] += 1
        else:
            betas[arm] += 1


def compare_policy(MB, num_sample):
    Ts = [np.power(10, i) for i in range(2, 6)]

    res_eps_greedy = []
    eps = 0.1
    res_UCB1 = []
    res_Thompson_sampling = []

    print "eps-Greedy running..."
    for T in Ts:
        print T
        tmp = []
        for sample in range(num_sample):
            print sample
            execute_eps_greedy(eps, MB, T)
            tmp.append(MB.get_regret())
            MB.reset_trial()
        res_eps_greedy.append(1.0*sum(tmp)/num_sample)

    print "UCB1 running..."
    for T in Ts:
        print T
        tmp = []
        for sample in range(num_sample):
            print sample
            execute_UCB1(MB, T)
            tmp.append(MB.get_regret())
            MB.reset_trial()
        res_UCB1.append(1.0*sum(tmp)/num_sample)

    print "Thompson_sampling running..."
    for T in Ts:
        print T
        tmp = []
        for sample in range(num_sample):
            print sample
            execute_Thompson_sampling(MB, T)
            tmp.append(MB.get_regret())
            MB.reset_trial()
        res_Thompson_sampling.append(1.0*sum(tmp)/num_sample)

    return Ts, res_eps_greedy, res_UCB1, res_Thompson_sampling

def main():
    MB = MultiarmedBandit(np.random.random(20))

    #T = 100000
    #execute_eps_greedy(0.1, MB, T)
    #execute_Thompson_sampling(MB, T)
    #execute_UCB1(MB, T)
    #MB.print_num_trials()
    #MB.print_num_hits()

    num_sample = 10
    Ts, res_eps_greedy, res_UCB1, res_Thompson_sampling = compare_policy(MB, num_sample)

    plt.xscale("log")
    plt.plot(Ts, res_eps_greedy, color='b', label="eps-Greedy", marker = 'o')
    plt.plot(Ts, res_UCB1, color='r', label="UCB1", marker = 'o')
    plt.plot(Ts, res_Thompson_sampling, color='g', label="Thompson_sampling", marker = 'o')
    plt.ylabel("regret")
    plt.xlabel("# of trial")
    plt.legend()
    plt.savefig("fig/regret.png", dpi=300)
    plt.show()
    print Ts, res_eps_greedy, res_UCB1, res_Thompson_sampling
    print MB.arm_probs

if __name__ == '__main__':
    main()
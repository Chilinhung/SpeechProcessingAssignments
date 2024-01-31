'''
name = 洪綺臨
stID = 61021027L
title = HMM HW1 (HW2) - likelihood & decoding
'''
import numpy as np
import math 


def forward(o, time, states, pie, a, b, alpha):
	for t in range(time):
		for state in range(states):
			if t == 0:
				alpha[t][state] = pie[state] * b[state][o[t]] #init.
			else:
				p = 0
				for i in range(states):
					p += alpha[t-1][i] * a[i][state]
				alpha[t][state] = p * np.float128(b[state][o[t]])
	p = 0
	for state in range(states):
		p += alpha[time - 1][state]
	return p

def decode(o, time, states, pie, a, b, alpha, psi, seq):
	for t in range(time):
		for state in range(states):
			if t == 0:
				alpha[t][state] = pie[state] * b[state][o[t]]
			else:
				p = -1e9
				for i in range(states):
					tmp = alpha[t-1][i] * a[i][state]
					if tmp > p:
						p = tmp
						psi[t][state] = i
				alpha[t][state] = p * np.float128(b[state][o[t]])
	p = -1e9
	for state in range(states):
		if alpha[time - 1][state] > p:
			p = alpha[time - 1][state]
			seq[time - 1] = state
	for t in range(time - 1,0,-1):
		seq[t-1] = psi[t][seq[t]]
	return seq, p

emission_type = {'up':0, 'down':1, 'unchanged':2}
a = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]]) # N states * N states
b = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]]) # N states * M emission types
o = [0, 0, 2, 1, 2, 1, 0]
pie = [0.5, 0.2, 0.3]
states = a.shape[0]
alpha = [[0] * states for i in range(len(o))] # N state * T timestamps
psi = [[0] * states for i in range(len(o))] # N state * T timestamps
seq = [0] * len(o)
likelihood = forward(o, len(o), states, pie, a, b, alpha)
path, p = decode(o, len(o), states, pie, a, b, alpha, psi, seq)
path = [x+1 for x in path]
print("P(up, up, unchanged, down, unchanged, down, up|λ):\nlikelihood = {}\noptimal state sequence = {}, probability = {}".format(likelihood, path,p))

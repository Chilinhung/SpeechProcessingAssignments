
import numpy as np
import math 
import random


trainStr1 = ["ABBCABCAABC","ABCABC","ABCAABC","BBABCAB","BCAABCCAB","CACCABCA","CABCABCA","CABCA","CABCA"]
trainStr2 = ["BBBCCBC","CCBABB","AACCBBB","BBABBAC","CCAABBAB","BBBCCBAA","ABBBBABA","CCCCC","BBAAA"]
recog_para = []

def rand_norm_init(n):
	num = [random.random() for iter in range(n)]
	num = [num[iter]/sum(num) for iter in range(len(num))]
	return num

def observation_trans(trainStr):
	transD = {"A":0, "B":1, "C":2}
	train = [" ".join(trainStr[iter]) for iter in range(len(trainStr))] # add space
	train = [train[i].split(" ") for i in range(len(train))] #seperate / string to list
	for i in range(len(train)): # chars to numbers
		for j in range(len(train[i])):
			train[i][j] = transD[train[i][j]]
	return train

def forward(o, time, states, pi, a, b, alpha):
	for t in range(time):
		for state in range(states):
			if t == 0:
				alpha[t][state] = pi[state] * b[state][o[t]] #init.
			else:
				p = 0
				for i in range(states):
					p += alpha[t-1][i] * a[i][state]
				alpha[t][state] = p * np.float128(b[state][o[t]]) #+0.000000001
	p = 0
	for state in range(states):
		p += alpha[time - 1][state]
	return p

def backward(o, time, states, pi, a, b, beta):
	for t in range(time - 1, -1, -1):
		for s in range(states):
			if t == time - 1:
				beta[t][s] = 1.0
			else:
				p = 0
				for s2 in range(states):
					p += a[s][s2] * np.float128(b[s2][o[t+1]]) * beta[t+1][s2]
				beta[t][s] = p #+ 0.000000001
	p = 0
	for s in range(states):
		p += pi[s] * b[s][o[0]] * beta[0][s]
	return p

def LogAdd(x, y):
	if x < y:
		temp = x
		x = y
		y = temp
	diff = y - x
	if diff < -1.0e10:
		return -1.0e10 if (x < -0.5e10) else x
	else:
		z = math.exp(diff)
		return x + math.log(1.0 + z)

def hmmLearning(obs, states, emissions, pi, a, b):
	f_alpha = []
	f_beta = []
	f_gamma = []
	f_xi = []
	f_time = []
	for epoch in range(101):
		if epoch == 0:
			for o in range(len(obs)):
				time = len(obs[o])
				alpha = [[0] * states for i in range(time)] # T timestamps * N states
				beta = [[0] * states for i in range(time)] # T timestamps * N states
				gamma = [[0] * states for i in range(time)] # T timestamps * N states
				xi = [[[0] * states for i in range(states)] for j in range(time)] # T timestamps * N states * N states
				f_alpha.append(alpha)
				f_beta.append(beta)
				f_gamma.append(gamma)
				f_xi.append(xi)
				f_time.append(time)
			#print(f_alpha)#, f_beta.shape(), f_gamma.shape(), f_xi.shape())
			print("1st: ",len(obs), time, states)

		else:				
			sum_pi_u = [0] * states #i
			sum_a_u = [[0] * states for i in range(states)] #i, j  #N * N
			#sum_a_u = [0] * states
			sum_a_d = [0] * states #i
			sum_b_u = [[0] * emissions for i in range(states)] # M * N
			sum_b_d = [0] * states #j
			for o in range(len(obs)): #o組值
				forward(obs[o], f_time[o], states, pi, a, b, f_alpha[o])
				backward(obs[o], f_time[o], states, pi, a, b, f_beta[o])					
				# E step
				for t in range(f_time[o]):
					p = 0
					for s in range(states):
						p += f_alpha[o][t][s] * f_beta[o][t][s]#np.logaddexp(alpha[t][s], beta[t][s])#
					assert p > 0
					for s in range(states):
						f_gamma[o][t][s] = f_alpha[o][t][s] * f_beta[o][t][s] / p
				for t in range(f_time[o] - 1):
					p = 0
					for s1 in range(states):
						for s2 in range(states):
							p += f_alpha[o][t][s1] * a[s1][s2] * b[s2][obs[o][t+1]] * f_beta[o][t+1][s2] #np.logaddexp(np.logaddexp(alpha[t][s], a[s1][s2]),beta[t+1][s2])#
					assert p > 0
					for s1 in range(states): #m
						for s2 in range(states): #n
							f_xi[o][t][s1][s2] = f_alpha[o][t][s1] * a[s1][s2] * b[s2][obs[o][t+1]] * f_beta[o][t+1][s2] / p
								
			# M step	
				#update pi
				for i in range(states): #每個 t0 的 state i 相加
					pi[i] = f_gamma[o][0][i]
					sum_pi_u[i] += pi[i]
				#update a
				for i in range(states):
					pg = 0
					for t in range(f_time[o] - 1):
						pg += f_gamma[o][t][i]
					assert pg != 0
					sum_a_d[i] += pg 
					for j in range(states):
						px = 0
						for t in range(f_time[o] - 1):
							px += f_xi[o][t][i][j]
						assert px != 0
						sum_a_u[i][j] += px
				#update b
				for n in range(states):
					pu = [0] * states
					pd = 0
					for m in range(emissions):
						for t in range(f_time[o]):
							if m == obs[o][t]:
								pu[m] += f_gamma[o][t][m]
					for t in range(f_time[o]):		
						pd += f_gamma[o][t][n]
					for m in range(emissions):
						sum_b_u[n][m] += pu[m]
					sum_b_d[n] += pd
			#update pi	
			for i in range(states):
				pi[i] /= len(obs)+1
			#update a
			for i in range(states):
				for j in range(states):
					a[i][j] = sum_a_u[i][j] / sum_a_d[i]
			#update b
			for i in range(states):
				for j in range(emissions):
					b[i][j] = sum_b_u[i][j] / sum_b_d[i]
						
		get = [1, 50]
		if epoch in get:
			print("\n=== epoch {} ===".format(epoch))
			print("-> pi\n",pi)
			print("-> a\n",a)
			print("-> b\n",b)
			tmp = {"a": a,"b": b, "pi":pi}
			recog_para.append(tmp)
			print("\n")

def trainModel(train):
	#a = np.array([rand_norm_init(3), rand_norm_init(3), rand_norm_init(3)])
	a = np.array([[0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34]])
	b = np.array([[0.34, 0.33, 0.33], [0.33, 0.34, 0.33], [0.33, 0.33, 0.34]]) # N states * M emission types
	states = a.shape[0]
	emissions = b.shape[1]
	pi = np.array([0.34,0.33, 0.33])#rand_norm_init(states)
	hmmLearning(train, states, emissions, pi, a, b)
	model = {"a": a,"b": b, "pi":pi}
	return model

#def recognition_test(recog_para, train, trainStr): #P2
	#for i in range(len(train)):
		#alpha = [[0] * len(model1["a"]) for i in range(len(train1[i]))] # T timestamps * N states
		#likelihood_1_1 = forward(train1[i], len(train1[i]), len(recog_para[0]["a"]), recog_para[0]["pi"], recog_para[0]["a"], recog_para[0]["b"],alpha)
		#likelihood_1_50 = forward(train1[i], len(train1[i]), len(recog_para[1]["a"]), recog_para[1]["pi"], recog_para[1]["a"], recog_para[1]["b"],alpha)
		#likelihood_2_1 = forward(train1[i], len(train1[i]), len(recog_para[2]["a"]), recog_para[2]["pi"], recog_para[2]["a"], recog_para[2]["b"],alpha)
		#likelihood_2_50 = forward(train1[i], len(train1[i]), len(recog_para[3]["a"]), recog_para[3]["pi"], recog_para[3]["a"], recog_para[3]["b"],alpha)
		##print(likelihood_1_1,"\t", likelihood_1_50,"\t", likelihood_2_1,"\t", likelihood_2_50)
		#print(trainStr[i], ": 1st iter ---> class{}".format(1 if likelihood_1_1 > likelihood_2_1 else 2),", 50th iter ---> class{}".format(1 if likelihood_1_50 > likelihood_2_50 else 2))

def testModel(testStr, model1, model2):
	test_ = observation_trans(testStr)
	test = []
	for i in range(len(test_)):
		test += test_[i]
	alpha = [[0] * len(model1["a"]) for i in range(len(test))] # T timestamps * N states
	likelihood1 = forward(test, len(test), len(model1["a"]), model1["pi"], model1["a"], model1["b"], alpha)
	likelihood2 = forward(test, len(test), len(model2["a"]), model2["pi"], model2["a"], model2["b"], alpha)
	#print(likelihood1, likelihood2)
	print(testStr, " ---> class {}".format(1 if likelihood1 > likelihood2 else 2))	

def omm_trainObs(train):
	trainObs = []
	for tl in enumerate(train):
		trainObs.extend(tl[1])
	return trainObs

def omm_transModel(o_train, states):
	prob_matrix = [[0] * states for i in range(states)] 
	for i in range(len(o_train) - 1):
		prob_matrix[o_train[i]][o_train[i+1]] += 1
	result_matrix = [[0] * states for i in range(states)] 
	for i in range(states):
		for j in range(states):
			result_matrix[i][j] = prob_matrix[i][j] / sum(prob_matrix[i])
	result_matrix = np.array(result_matrix)
	return result_matrix

def omm(testStr, model1, model2):
	pi = np.array([0.34, 0.33, 0.33])
	test = observation_trans(testStr)
	model1_prob = 1
	model2_prob = 1
	for i, o_curr in enumerate(test):
		if i == 0:
			model1_prob  *= pi[o_curr[0]]
			model2_prob  *= pi[o_curr[0]]
		else:
			model1_prob  *= model1[o_prev[0]][o_curr[0]]
			model2_prob  *= model2[o_prev[0]][o_curr[0]]
		o_prev = o_curr
	print(testStr, " ---> class {}".format(1 if model1_prob > model2_prob else 2))	
	
	
if __name__ == "__main__":
	#P1 : train
	train1 = observation_trans(trainStr1)
	train2 = observation_trans(trainStr2)
	model1 = trainModel(train1)
	model2 = trainModel(train2)
	print("=====================")
	print("model1:\n",model1)
	print("\nmodel2:\n",model2)
	print("=====================")
	#P2 : recognition test
	print("\n==== train set 1 ====")
	for i, string in enumerate(trainStr1):
		testModel(string, model1, model2)	
	print("\n==== train set 2 ====")
	for i, string in enumerate(trainStr2):
		testModel(string, model1, model2)	
	print("\n=====================")
	#P3 : test / predict
	test1Str = "ABCABCCAB"
	test2Str = "AABABCCCCBBB"
	testModel(test1Str, model1, model2)
	testModel(test2Str, model1, model2)	
	print("=====================")
	#P4
	o_train1 = omm_trainObs(train1)
	o_train2 = omm_trainObs(train2)
	o_model1 = omm_transModel(o_train1, 3)
	o_model2 = omm_transModel(o_train2, 3)
	print("P4:")
	print("<model 1>")
	print(o_model1)
	print("<model 2>")
	print(o_model2)
	print("\n=== train set 1 ===")
	for i, string in enumerate(trainStr1):
		omm(string, o_model1, o_model2)
	print("=== train set 2 ===")
	for i, string in enumerate(trainStr2):
		omm(string, o_model1, o_model2)	
	print("=== test ===")
	omm(test1Str, o_model1, o_model2)
	omm(test2Str, o_model1, o_model2)
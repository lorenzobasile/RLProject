import pickle
import matplotlib.pyplot as plt
import numpy as np

gamma=0.999

infile = open("data.pickle",'rb')
distances = pickle.load(infile)
times_thompson = pickle.load(infile)
times_greedy = pickle.load(infile)

regret_thompson = times_thompson**gamma-distances**gamma
regret_greedy = times_greedy**gamma-distances**gamma

avg_regret_thompson = np.average(regret_thompson, axis=1)
avg_regret_greedy = np.average(regret_greedy, axis=1)


plt.plot(avg_regret_thompson, label="thompson")
plt.plot(avg_regret_greedy, label="greedy")
plt.ylabel("regret")
plt.xlabel("tau")
plt.legend()

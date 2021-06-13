import pickle
import matplotlib.pyplot as plt
import numpy as np

infile = open("data.pickle",'rb')
distances = pickle.load(infile)
times_thompson = pickle.load(infile)
times_greedy = pickle.load(infile)

relative_times_thompson = times_thompson/distances
relative_times_greedy = times_greedy/distances

avg_relative_times_thompson = np.average(relative_times_thompson, axis=1)
avg_relative_times_greedy = np.average(relative_times_greedy, axis=1)


plt.plot(avg_relative_times_thompson)
plt.plot(avg_relative_times_greedy)
plt.show()

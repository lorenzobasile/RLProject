from gridworld import *
import pickle
import sys

#distance is chosen through command line argument

if sys.argv[1]=="man":
    metrics=manhattan_distance
elif sys.argv[1]=="man2":
    metrics=squared_manhattan_distance
elif sys.argv[1]=="euc":
    metrics=euclidean_distance
elif sys.argv[1]=="euc2":
    metrics=squared_euclidean_distance

nrows=40
ncols=40

#number of configurations (initial states and real targets), chosen uniformly at random
n_config=100

tau=np.arange(1,nrows+ncols+1)
n_tau=len(tau)
times_thompson=np.zeros((n_tau,n_config)) #number of steps taken by Thompson algorithm
times_greedy=np.zeros((n_tau,n_config)) #number of steps taken by greedy algorithm
distances=np.zeros(n_config) #distances between initial states and real targets


for j in range(n_config):
    init_state = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    real_target = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    distances[j]=manhattan_distance(init_state, real_target)

    for k in range(n_tau):
        grid=Gridworld(nrows, ncols, metrics, real_target, init_state, render=False)
        times_thompson[k,j]=gridworld_search(grid, tau[k])
        grid=Gridworld(nrows, ncols, metrics, real_target, init_state, render=False)
        times_greedy[k,j]=gridworld_search(grid, tau[k], greedy=True)

filename="data.pickle"
outfile = open(filename,'wb')
pickle.dump(distances, outfile)
pickle.dump(times_thompson, outfile)
pickle.dump(times_greedy, outfile)

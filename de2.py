from belief import *
import pickle
import sys

if sys.argv[1]=="man":
    metrics=manhattan_distance
elif sys.argv[1]=="man2":
    metrics=squared_manhattan_distance
elif sys.argv[1]=="euc":
    metrics=euclidean_distance
elif sys.argv[1]=="euc2":
    metrics=squared_euclidean_distance

nrows=4
ncols=4
gamma=1

n_config=3

steps=np.arange(1,nrows+ncols+1)
n_steps=len(steps)
times_thompson=np.zeros((n_steps,n_config))
times_greedy=np.zeros((n_steps,n_config))
distances=np.zeros(n_config)


for j in range(n_config):
    init_state = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    real_target = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    distance=manhattan_distance(init_state, real_target)
    distances[j]=distance

    for k in range(n_steps):
        grid=TSGridworld(nrows, ncols, gamma, metrics, real_target, init_state, render=False)
        times_thompson[k,j]=thompson_loop(grid, steps[k])
        grid=TSGridworld(nrows, ncols, gamma, metrics, real_target, init_state, render=False)
        times_greedy[k,j]=thompson_loop(grid, steps[k], greedy=True)

filename="data_"+sys.argv[1]+str(nrows)+"x"+str(ncols)+"_"+str(n_config)+"config.pickle"
outfile = open(filename,'wb')
pickle.dump(distances, outfile)
pickle.dump(times_thompson, outfile)
pickle.dump(times_greedy, outfile)

from belief import *
import pickle

nrows=10  #40
ncols=10  #40
gamma=1

n_config=3 # 100

steps=np.arange(1,nrows+ncols+1)
n_steps=len(steps)
times_thompson=np.zeros((n_steps,n_config))
times_greedy=np.zeros((n_steps,n_config))
distances=np.zeros(n_config)


for j in range(n_config):
    init_state = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    real_target = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    distance=squared_manhattan_distance(init_state, real_target)
    distances[j]=distance

    for k in range(n_steps):
        grid=TSGridworld(nrows, ncols, gamma, manhattan_distance, real_target, init_state, render=False)
        times_thompson[k,j]=thompson_loop(grid, steps[k])
        grid=TSGridworld(nrows, ncols, gamma, manhattan_distance, real_target, init_state, render=False)
        times_greedy[k,j]=thompson_loop(grid, steps[k], greedy=True)

outfile = open("data.pickle",'wb')
pickle.dump(distances, outfile)
pickle.dump(times_thompson, outfile)
pickle.dump(times_greedy, outfile)

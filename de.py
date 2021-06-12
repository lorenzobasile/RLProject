from belief import *

nrows=10
ncols=20
gamma=1

n_config=5

n_steps=np.arange(1,21)
np.append(n_steps, np.inf)
relative_times_thompson=np.zeros_like(n_steps, dtype=np.float)
relative_times_greedy=np.zeros_like(n_steps, dtype=np.float)


for j in range(n_config):
    init_state = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    real_target = np.random.choice(range(nrows)), np.random.choice(range(ncols))
    distance=manhattan_distance(init_state, real_target)
    
    for k in range(len(n_steps)):
        grid=TSGridworld(nrows, ncols, gamma, manhattan_distance, real_target, init_state, render=False)
        relative_times_thompson[k]+=(distance/thompson_loop(grid, n_steps[k]))
        grid=TSGridworld(nrows, ncols, gamma, manhattan_distance, real_target, init_state, render=False)
        relative_times_greedy[k]+=(distance/thompson_loop(grid, n_steps[k], greedy=True))
    
print(relative_times_greedy/n_config)
print(relative_times_thompson/n_config)
    
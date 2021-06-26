from belief import *

nrows=10
ncols=20
gamma=1
np.random.seed(3)
init_state = np.random.choice(range(nrows)), np.random.choice(range(ncols))
real_target = np.random.choice(range(nrows)), np.random.choice(range(ncols))
grid=TSGridworld(nrows, ncols, gamma, manhattan_distance, real_target, init_state, render=True)
print(thompson_loop(grid, 1))

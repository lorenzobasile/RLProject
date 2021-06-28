from gridworld import *

nrows=10
ncols=20
np.random.seed(3)    # for reproducibility
init_state = np.random.choice(range(nrows)), np.random.choice(range(ncols))
real_target = np.random.choice(range(nrows)), np.random.choice(range(ncols))
grid=Gridworld(nrows, ncols, manhattan_distance, real_target, init_state, render=True)
print(gridworld_search(grid, 1))

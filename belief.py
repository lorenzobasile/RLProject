import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.animation as animation


class TSGridworld():
    def __init__(self, nrows, ncols, gamma, distance):
        self.dimensions=(nrows,ncols)
        self.belief=np.ones(self.dimensions)/(nrows*ncols)
        self.belief_sequence=[self.belief]
        self.state=(0,0)
        self.distance=distance
    def render(self):
        plt.imshow(self.belief, cmap='gray', interpolation='nearest')
        plt.show()
    def init_animation(self):
        pass
    def updatefig(self, j):
        if j==0:
            self.im=plt.imshow(self.belief_sequence[0], cmap='autumn', vmin=0, vmax=1)
            plt.colorbar()
        self.im.set_array(self.belief_sequence[j])
        return [self.im]
    def animated_render(self):
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, self.updatefig, init_func=self.init_animation, frames=range(len(self.belief_sequence)), repeat=False)
        plt.show()
    def update(self, observation, state):
        lkl=np.vectorize(self.likelihood)
        nrows, ncols=self.dimensions
        likelihood_matrix=np.ones_like(self.belief)
        for target_pos in product(range(nrows), range(ncols)):
            likelihood_matrix[target_pos]=self.likelihood(observation, target_pos, state)
        marginal=np.sum(np.multiply(self.belief, likelihood_matrix))
        self.belief=np.multiply(self.belief, likelihood_matrix)/marginal
        self.belief_sequence.append(self.belief)
    def step(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            self.state=new_state
        return self.state
    def sample_model(self):
        index=np.random.choice(range(self.dimensions[0]*self.dimensions[1]), p=self.belief.flatten())
        r=index//self.dimensions[1]
        c=index%self.dimensions[1]
        return (r,c)
    def policy(self, est_target):
        var_r=est_target[0]-self.state[0]
        var_c=est_target[1]-self.state[1]
        action_r=np.sign(var_r)
        action_c=np.sign(var_c)
        if action_r==0 or action_c==0:
            return (action_r, action_c)
        elif np.random.uniform()>0.5:
            return (action_r, 0)
        else:
            return (0, action_c)
    def likelihood(self, y, est_target, state):
        p=1/(self.distance(state, est_target)+1)
        if y==1:
            return p
        else:
            return 1-p

nrows=10
ncols=20
gamma=1

def manhattan_distance(s1, s2):
    x1,y1=s1
    x2,y2=s2
    return abs(x1-x2)+abs(y1-y2)

grid=TSGridworld(nrows, ncols, gamma, manhattan_distance)
real_target=(9,19)
np.random.seed(0)
for t in range(1000):
    target_pos=grid.sample_model()
    action=grid.policy(target_pos)
    new_state=grid.step(action)
    obs=1 if np.random.uniform()<1/(manhattan_distance(new_state, real_target)+1) else 0
    print(new_state, " ", obs)
    grid.update(obs, new_state)
grid.animated_render()

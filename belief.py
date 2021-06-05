import numpy as np
import matplotlib.pyplot as plt


class TSGridworld():
    def __init__(self, nrows, ncols, gamma, distance):
        self.dimensions=(nrows,ncols)
        self.belief=np.ones(self.dimensions)/(nrows*ncols)
        self.state=(0,0)
        self.distance=distance
    def render(self):
        plt.imshow(self.belief, cmap='gray', interpolation='nearest')
        plt.show()
    def update(self, observation, state):
        nrows, ncols=self.dimensions
        likelihood_matrix=np.ones_like(self.belief)
        for r in range(nrows):
            for c in range(ncols):
                likelihood_matrix[r,c]=self.likelihood(observation, (r,c), state)
        marginal=np.sum(np.multiply(self.belief, likelihood_matrix))
        for r in range(nrows):
            for c in range(ncols):
                self.belief[r,c]=self.belief[r,c]*self.likelihood(observation, (r,c), state)/marginal
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
real_target=(3,6)
np.random.seed(0)
for t in range(10000):
    target_pos=grid.sample_model()
    action=grid.policy(target_pos)
    new_state=grid.step(action)
    obs=1 if np.random.uniform()>manhattan_distance(new_state, real_target) else 0
    grid.update(obs, new_state)

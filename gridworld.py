import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def manhattan_distance(s1, s2):
    return abs(s1[0]-s2[0])+abs(s1[1]-s2[1])

def squared_manhattan_distance(s1, s2):
    return manhattan_distance(s1, s2)**2

def squared_euclidean_distance(s1, s2):
    return (s1[0]-s2[0])**2+(s1[1]-s2[1])**2

def euclidean_distance(s1, s2):
    return np.sqrt(squared_euclidean_distance(s1, s2))


class Gridworld():
    def __init__(self, nrows, ncols, distance, real_target, init_state, render=True):
        self.dimensions=(nrows,ncols)
        self.belief=np.ones(self.dimensions)/(nrows*ncols)    # uniform prior
        self.state=init_state    # current state
        self.distance=distance
        self.done=False    # flag set to true when real target is reached (reward=1)
        self.real_target=real_target    # real target position
        self.estimated_target=(None,None)     # current estimate of target position
        self.render=render

        if self.render:
          self.fig = plt.figure()
          self.ax = self.fig.add_subplot(111)
          self.im = self.ax.imshow(self.belief, cmap='Greens')
          self.ax.set_xticks(np.arange(self.dimensions[1], dtype=np.int))
          self.ax.set_yticks(np.arange(self.dimensions[0], dtype=np.int))
          self.scat_real_target = self.ax.scatter(self.real_target[1], self.real_target[0], color='r', marker='*')
          self.scat_mel = self.ax.scatter(self.state[1], self.state[0], color='k', marker='o')
          self.scat_target = self.ax.scatter(self.estimated_target[1], self.estimated_target[0], color='b', marker='x')
          self.fig.colorbar(self.im, ax=self.ax, ticks=None)
          plt.show(block=False)

    def show(self):
        self.im.autoscale()
        self.im.set_array(self.belief)
        self.scat_me = self.ax.scatter(self.state[1], self.state[0], color='y', marker='o', s=8)
        self.scat_mel.set_offsets([self.state[1], self.state[0]])
        self.scat_target.set_offsets([self.estimated_target[1], self.estimated_target[0]])
        self.fig.canvas.draw()
        plt.pause(0.1)

    #update: posterior calculation given an observation

    def update(self, observation):
        nrows, ncols=self.dimensions
        likelihood_matrix=np.empty_like(self.belief)
        for pos in product(range(nrows), range(ncols)):
            likelihood_matrix[pos]=self.likelihood(observation, pos)
        self.belief=np.multiply(self.belief, likelihood_matrix)
        self.belief/=np.sum(self.belief)
        if self.render:
            self.show()

    #step: apply action and transition to the new state

    def step(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        #if the new position is outside the grid the state is not updated
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            self.state=new_state
        self.done = (self.state==self.real_target)

    #thompson: thompson sampling to choose a new estimate for the target position
    #greedy: greedy choice for the next estimated target position

    def thompson(self):
        index=np.random.choice(range(self.dimensions[0]*self.dimensions[1]), p=self.belief.flatten())
        self.estimated_target = np.unravel_index(index, self.dimensions)
    def greedy(self):
        index=np.argmax(self.belief)
        self.estimated_target = np.unravel_index(index, self.dimensions)

    #policy: pick action to get one step closer to the current estimate of the target

    def policy(self):
        var_r=self.estimated_target[0]-self.state[0]
        var_c=self.estimated_target[1]-self.state[1]
        action_r=np.sign(var_r) # direction of movement along horizontal axis
        action_c=np.sign(var_c) # direction of movement along vertical axis
        if action_r==0 or action_c==0: # stay still on at least one axis
            return (action_r, action_c)
        elif np.random.uniform()>0.5: # if two actions are equivalent, choose randomly
            return (action_r, 0)
        else:
            return (0, action_c)

    #likelihood: compute the likelihood of a given observation y assuming that the target is est_target

    def likelihood(self, y, est_target):
        p=1/(self.distance(self.state, est_target)+1) # p: parameter of the Bernoulli distribution
        return p if y==1 else 1-p

    #observe: get real (random) observation from the system

    def observe(self):
        return 1 if np.random.uniform()<1/(self.distance(self.state, self.real_target)+1) else 0


def gridworld_search(grid, n_steps, greedy=False):
    t=0
    while not grid.done:
        if greedy:
            grid.greedy()
        else:
            grid.thompson()
        i=0
        reached_est_target=False
        while not reached_est_target and i < n_steps and not grid.done:
            action=grid.policy()
            grid.step(action)
            obs=grid.observe()
            grid.update(obs)
            t+=1
            i+=1
            reached_est_target = grid.state == grid.estimated_target
    return t

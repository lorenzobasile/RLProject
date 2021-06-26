import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.animation as animation

def manhattan_distance(s1, s2):
    return abs(s1[0]-s2[0])+abs(s1[1]-s2[1])

def squared_manhattan_distance(s1, s2):
    return manhattan_distance(s1, s2)**2

def squared_euclidean_distance(s1, s2):
    return (s1[0]-s2[0])**2+(s1[1]-s2[1])**2

def euclidean_distance(s1, s2):
    return np.sqrt(squared_euclidean_distance(s1, s2))


class TSGridworld():
    def __init__(self, nrows, ncols, gamma, distance, real_target, init_state, render=True):
        self.dimensions=(nrows,ncols)
        self.belief=np.ones(self.dimensions)/(nrows*ncols)
        self.state=init_state
        self.distance=distance
        self.done=False
        self.real_target=real_target    # real target position
        self.estimated_target=(None,None)     # estimated target position
        self.state_list=np.array([(i,j) for i, j in product(range(nrows),range(ncols))], dtype="i,i")
        self.likelihood_matrix=np.empty_like(self.belief)
        self.likelihood_fast=np.vectorize(self.likelihood, excluded=['state'])

        if render:
          self.fig = plt.figure()
          self.ax = self.fig.add_subplot(111)
          self.im = self.ax.imshow(self.belief)
          self.ax.set_xticks(np.arange(self.dimensions[1], dtype=np.int))
          self.ax.set_yticks(np.arange(self.dimensions[0], dtype=np.int))
          self.scat_mel = self.ax.scatter(self.state[1], self.state[0], color='w', marker='o')
          self.scat_target = self.ax.scatter(self.estimated_target[1], self.estimated_target[0], color='b', marker='x')
          self.fig.colorbar(self.im, ax=self.ax, ticks=None)
          plt.show(block=False)

    def render(self):
        self.im.autoscale()
        self.im.set_array(self.belief)
        self.scat_me = self.ax.scatter(self.state[1], self.state[0], color='r', marker='o', s=10)
        self.scat_mel.set_offsets([self.state[1], self.state[0]])
        self.scat_target.set_offsets([self.estimated_target[1], self.estimated_target[0]])
        self.fig.canvas.draw()
        plt.pause(0.1)

    def update(self, observation, state):
        nrows, ncols=self.dimensions
        self.likelihood_matrix=self.likelihood_fast(y=observation, est_target=self.state_list, state=state).reshape((nrows,ncols))
        self.belief=np.multiply(self.belief, self.likelihood_matrix)
        self.belief/=np.sum(self.belief)

    def step(self, action):
        new_state=self.state[0]+action[0], self.state[1]+action[1]
        if new_state[0]<self.dimensions[0] and new_state[0]>=0 and new_state[1]<self.dimensions[1] and new_state[1]>=0:
            self.state=new_state
        self.done = (self.state==self.real_target)
        return self.state

    def thompson(self):
        index=np.random.choice(range(self.dimensions[0]*self.dimensions[1]), p=self.belief.flatten())
        self.estimated_target = np.unravel_index(index, self.dimensions)
        return self.estimated_target
    def greedy(self):
        index=np.argmax(self.belief)
        self.estimated_target = np.unravel_index(index, self.dimensions)
        return self.estimated_target

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
        return p if y==1 else 1-p

    def observe(self, state):
        return 1 if np.random.uniform()<1/(self.distance(state, self.real_target)+1) else 0


def thompson_loop(grid, n_steps, greedy=False):
    t=0
    while not grid.done:
        if greedy:
            target_pos=grid.greedy()
        else:
            target_pos=grid.thompson()
        i=0
        condition=True
        while condition and i < n_steps and not grid.done:
            action=grid.policy(target_pos)
            new_state=grid.step(action)
            obs=grid.observe(new_state)
            grid.update(obs, new_state)
            grid.render()
            t+=1
            i+=1
            condition= grid.state != target_pos
    return t

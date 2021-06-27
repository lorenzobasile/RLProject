# Thompson Sampling for Gridworld

Final project for Reinforcement Learning course

Authors: Lorenzo Basile, Irene Brugnara

Source files:

- `gridworld.py` contains the implementation of the class `Gridworld`, defining a two-dimensional grid environment in which an agent moves from an initial cell to a target cell. The position of the target cell is not known, but at each time step the agent receives a random binary signal from the target depending on its distance from the target. The method `gridworld_search` implements a search algorithm based on Thompson sampling (or a greedy algorithm if `greedy=True`);
- `animation.py` contains an example animated run of the search algorithm;
- `benchmark.py` and `analysis.py` respectively contain code to collect and process data on larger-scale runs of both Thompson algorithm and greedy algorithm.

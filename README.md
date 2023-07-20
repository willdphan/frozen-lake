# Frozen Lake

## What is Frozen Lake?
The Frozen Lake game is a simple grid-based environment where the player controls a character navigating a frozen lake. The objective is to reach the goal position without falling into holes in the ice. 


The lake is represented by a grid, with each cell being either frozen, indicating safe ground, or a hole, which is a dangerous spot. The player can move in four directions: up, down, left, and right. 


However, the ice is slippery, so once the player starts moving in a direction, they will continue sliding until they hit an obstacle or the edge of the grid. 
The game provides a challenging and strategic experience as the player must plan their moves carefully to avoid the treacherous holes and successfully reach the goal.

## How Frozen Lake Works
This grid is our environment where S is the agent's starting point, and it's safe. F represents the frozen surface and is also safe. H represents a hole, and if our agent steps in a hole in the middle of a frozen lake, well, that's not good. Finally, G represents the goal, which is the space on the grid where the prized frisbee is located.

The agent can navigate left, right, up, and down, and the episode ends when the agent reaches the goal or falls in a hole. It receives a reward of one if it reaches the goal, and zero otherwise.

[Code](Frozen_Lake_Q_Learning.ipynb)

## License

This script is open-source and licensed under the MIT License. For more details, check the [LICENSE](LICENSE) file.

# Simple_DDQN_Cartpole
Simple_DDQN_Cartpole: Applying Double Deep Q-Networks to train an agent for balancing a pole on a cart. Reinforcement learning project focuses on efficient learning and improved decision-making for mastering the Cartpole control problem.

## Files

- `agent.py`: Contains classes for the DDQN agent and its training algorithm.
- `train.py`: Initializes the Cartpole environment and trains the agent. It saves model weights and displays training curves.
- `test.py`: Uses saved model weights to test the agent's inference performance.

## Usage

1. Install the necessary dependencies: Python 3.x, PyTorch, NumPy.
2. Run `train.py` to train the DDQN agent on the Cartpole environment.
3. After training, model weights are saved for future use.
4. Run `test.py` to evaluate the trained agent's performance using the saved weights.
5. Training progress and inference results can be monitored through the displayed curves and outputs.


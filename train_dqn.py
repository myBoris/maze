import os
import sys

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt
from tqdm import tqdm

from agent.dqn import DQNAgent
from env.maze_env import MazeEnv, MainWindow

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    maze_array = np.array([
        [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0]
    ])

    env = MazeEnv(maze_array)
    state_size = maze_array.size
    action_size = 4
    agent = DQNAgent(state_size, action_size, batch_size=64, gamma=0.99, epsilon=1.0, eps_decay=0.995, eps_min=0.01,
                     target_update=10, memory_capacity=10000, device=device)

    num_episodes = 300000
    rewards = []

    for i_episode in tqdm(range(num_episodes), desc="Training Progress"):
        state = env.reset().flatten()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        total_reward = 0

        for t in range(100):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            reward = torch.tensor([reward], dtype=torch.float32).to(device)
            total_reward += reward.item()

            agent.memory.push(state, action, next_state, reward)

            state = next_state

            agent.optimize_model()

            if done:
                break

        rewards.append(total_reward)

        if i_episode % 1000 == 0:
            print(
                f"Episode {i_episode}/{num_episodes} complete - Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

            if rewards:
                plt.plot(rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Training Rewards Over Time')
                plt.savefig('training_rewards.png')
                # plt.show()

    print("Training complete")

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards Over Time')
    plt.savefig('training_rewards.png')
    plt.show()

    agent.save("dqn_model.pth")

    app = QApplication(sys.argv)
    main_window = MainWindow(env)
    sys.exit(app.exec_())

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    maze_array = np.array([
        [0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0]
    ])

    env = MazeEnv(maze_array)
    state_size = maze_array.size
    action_size = 4
    agent = DQNAgent(state_size, action_size, batch_size=64, gamma=0.99, epsilon=1.0, eps_decay=0.995, eps_min=0.01,
                     target_update=10, memory_capacity=10000, device=device)
    agent.load("dqn_model.pth")

    state = env.reset().flatten()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0
    steps = 0

    while True:
        env.render()
        action = agent.policy_net(state).max(1)[1].view(1, 1)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        total_reward += reward
        steps += 1

        state = next_state

        if done:
            print(f"Goal reached in {steps} steps with total reward {total_reward}")
            break

if __name__ == '__main__':
    main()
    # test_model()

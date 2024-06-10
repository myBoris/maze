import random
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer


class MazeEnv:
    def __init__(self, maze_array):
        self.maze = maze_array
        self.start_pos = tuple(np.argwhere(self.maze == 8)[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 9)[0])
        self.current_pos = self.start_pos

    def reset(self):
        self.current_pos = self.start_pos
        return self._get_obs()

    def step(self, action):
        x, y = self.current_pos
        if action == 0:  # 上
            x = max(0, x - 1)
        elif action == 1:  # 下
            x = min(self.maze.shape[0] - 1, x + 1)
        elif action == 2:  # 左
            y = max(0, y - 1)
        elif action == 3:  # 右
            y = min(self.maze.shape[1] - 1, y + 1)

        if self.maze[x, y] in [1, 8, 9]:
            self.current_pos = (x, y)

        done = self.current_pos == self.goal_pos
        reward = 1 if done else -0.1
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = np.zeros_like(self.maze)
        obs[self.current_pos] = 1
        return obs

    def render(self):
        print(self.maze)
        print(f'当前坐标: {self.current_pos}')


class MazeWidget(QWidget):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 500, 500)
        self.setWindowTitle('Maze Environment')
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def paintEvent(self, event):
        painter = QPainter(self)
        self.drawMaze(painter)

    def drawMaze(self, painter):
        cell_size = 50
        for i in range(self.env.maze.shape[0]):
            for j in range(self.env.maze.shape[1]):
                rect = (j * cell_size, i * cell_size, cell_size, cell_size)
                if self.env.maze[i, j] == 0:
                    painter.setBrush(QBrush(Qt.black, Qt.SolidPattern))
                elif self.env.maze[i, j] == 1:
                    painter.setBrush(QBrush(Qt.white, Qt.SolidPattern))
                elif self.env.maze[i, j] == 8:
                    painter.setBrush(QBrush(Qt.green, Qt.SolidPattern))
                elif self.env.maze[i, j] == 9:
                    painter.setBrush(QBrush(Qt.blue, Qt.SolidPattern))
                painter.drawRect(*rect)

        x, y = self.env.current_pos
        painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        painter.drawRect(y * cell_size, x * cell_size, cell_size, cell_size)


class MainWindow(QMainWindow):
    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.initUI()
        # self.random_move_timer = QTimer(self)
        # self.random_move_timer.timeout.connect(self.random_move)
        # self.random_move_timer.start(500)  # 每500毫秒执行一次随机移动

    def initUI(self):
        self.setWindowTitle('迷宫环境')
        self.setGeometry(100, 100, 500, 500)
        self.maze_widget = MazeWidget(self.env)
        self.setCentralWidget(self.maze_widget)
        self.show()

    def keyPressEvent(self, event):
        action = None
        if event.key() == Qt.Key_Up:
            action = 0
        elif event.key() == Qt.Key_Down:
            action = 1
        elif event.key() == Qt.Key_Left:
            action = 2
        elif event.key() == Qt.Key_Right:
            action = 3

        if action is not None:
            obs, reward, done, info = self.env.step(action)
            self.maze_widget.update()
            if done:
                print("Goal reached!")
                self.env.reset()

    def random_move(self):
        action = random.choice([0, 1, 2, 3])
        obs, reward, done, info = self.env.step(action)
        self.maze_widget.update()
        if done:
            print("Goal reached!")
            self.env.reset()


def main():
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
    app = QApplication(sys.argv)
    main_window = MainWindow(env)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

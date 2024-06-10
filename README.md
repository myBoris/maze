# 训练DQN玩迷宫，后续会更新其他网络来玩迷宫
## 运行效果
<div><video controls src="https://github.com/myBoris/maze/blob/main/imgs/%E6%B5%8B%E8%AF%95.mp4" muted="false"></video></div>

## 1.环境配置教程
+ 1.使用anaconda创建一个环境 (conda create --name maze python=3.10)
+ 2.安装必要的包,安装命令在 ([command.txt](command.txt)) 
+ + 这是必要的包
+ + `pip install -r requirements.txt`
+ + 这是pytorch和cuda的安装
+ + `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`


## 2.使用教程<br>
+ 直接运行[train_dqn.py](train_dqn.py)进行训练
+ [train_dqn.py](train_dqn.py) 这个文件最下面的有如下代码<br>
+ `if __name__ == '__main__': ` <br>` main() 　#训练用的方法 ` <br>` # test_model() #测试用的方法，这里是测试训练好的模型，测试时打开`  

## 3.环境截图<br>
+ 绿色是开始，蓝色是结束，红色是当前位置，图中红色是处于开始位置，当移动后，红色会改变位置<br>
![maze_env.png](imgs%2Fmaze_env.png)
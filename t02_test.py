if __name__ == '__main__':
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import t02_vfl as vfl

    '''
    # 定义线性回归模型
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LinearRegressionModel, self).__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            x = self.fc(x)
            return x
    '''


    # 定义组类
    class Group:
        def __init__(self, config):
            # self.clients = clients
            self.config = config
            # 这里为了测试简单，先假设每个组的config都一样
            self.X, self.y, self.X_test, self.y_test = vfl.load_data(self.config['overlap'])

        def train(self, weight_a, weight_b):
            # 进行垂直联邦学习训练
            weight = vfl.vertical_logistic_regression(self.X, self.y, self.X_test, self.y_test, self.config, weight_a, weight_b)
            # 将主动方和被动方的参数拼起来
            result = np.concatenate((weight[0], weight[1]))#clientA,clientB
            return result, weight[2]


    # 定义客户端类
    '''
    class Client:
        def __init__(self, data, target):
            self.data = data
            self.target = target
            self.model = LinearRegressionModel(data.shape[1])
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
    '''


    # 定义联邦学习框架类
    class FederatedLearning:
        def __init__(self, groups):
            self.groups = groups

        def train(self, num_epochs):  # ！！！！还没加入各个组的权重
            losses = []  # 用于记录每个训练周期的损失值, 画图
            weight_a = np.zeros(3)  # 用于记录每轮各个组中主动方的参数变化
            weight_b = np.zeros(9)  # 用于记录每轮各个组中被动方的参数变化
            #修改一下，权重完全由样本数量决定试一下
            weight_group=[100,60,70,80,90]
            for epoch in range(num_epochs):
                epoch_losses = []  # 用于记录每个组的损失值
                sum_params = [0] * 12
                for group in self.groups:
                    group_losses = group.train(weight_a, weight_b)
                    epoch_losses.append(group_losses[1])
                    # 接下来开始聚合操作
                    sum_params = [x+(group.config['overlap'])*y for x,y in zip(sum_params, group_losses[0])]  # 这里group前面可以加上对应的权重
                #sum_weights= sum(1/x for x in epoch_losses)
                avg_params = [item / 20000 for item in sum_params]  # 聚合之后的参数
                #weight_a = avg_params[8:11]
                #weight_b = avg_params[:8]
                weight_a = avg_params[0:3]
                weight_b = avg_params[3:12]
                #losses.append(sum(epoch_losses) / len(epoch_losses))  # 计算平均损失值!这里需要加上权重吧
                losses.append(epoch_losses[0]*(10/40) + epoch_losses[1]*(6/40)+ epoch_losses[2]*(7/40) + epoch_losses[3]*(8/40) +epoch_losses[4]*(9/40))
                # self.aggregate()

            return losses

    # 创建客户端和组
    # 这里先假设每个组的config是一样的
    config1 = {
        'n_iter': 30,
        'lambda': 10,
        'lr': 0.05,
        'A_idx': [8, 9, 10],
        'B_idx': [0, 1, 2, 3, 4, 5, 6, 7],
        'overlap': 5000,
    }
    config2 = {
        'n_iter': 30,
        'lambda': 10,
        'lr': 0.05,
        'A_idx': [8, 9, 10],
        'B_idx': [0, 1, 2, 3, 4, 5, 6, 7],
        'overlap': 3000,
    }
    config3 = {
        'n_iter': 30,
        'lambda': 10,
        'lr': 0.05,
        'A_idx': [8, 9, 10],
        'B_idx': [0, 1, 2, 3, 4, 5, 6, 7],
        'overlap': 3500,
    }
    config4 = {
        'n_iter': 30,
        'lambda': 10,
        'lr': 0.05,
        'A_idx': [8, 9, 10],
        'B_idx': [0, 1, 2, 3, 4, 5, 6, 7],
        'overlap': 4000,
    }
    config5 = {
        'n_iter': 30,
        'lambda': 10,
        'lr': 0.05,
        'A_idx': [8, 9, 10],
        'B_idx': [0, 1, 2, 3, 4, 5, 6, 7],
        'overlap': 4500,
    }
    group1 = Group(config1)
    group2 = Group(config2)
    group3 = Group(config3)
    group4 = Group(config4)
    group5 = Group(config5)
    groups =[group1, group2, group3, group4, group5]
    fl=FederatedLearning(groups)
    # 训练每个组
    num_epochs = 10
    losses = fl.train(num_epochs)
    print(losses)

    # 绘制损失曲线
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("loss.png")
    plt.show()

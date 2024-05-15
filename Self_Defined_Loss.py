import torch
import torch.nn as nn

'''重新定义损失函数'''


class CorrLoss(nn.Module):
    def __init__(self, m1=1.0, m2=0.5):
        super(CorrLoss, self).__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, y_pred, y_true, model):
        """
        :param model:
        :param y_pred: 要保证【Batch，1，64】
        :param y_true: 要保证【Batch，1，1】
        :return: loss标量
        """
        y_pred.squeeze_(1)  # 将预测值的第二个维度去掉，变成[Batch, 64]
        y_true.squeeze_(1)  # 将真实标签的第二个维度去掉，变成[Batch, 1]

        """
        计算因子间相关系数Corr(\hat{y}_i, \hat{y}_j)的平方和
        """
        corr_matrix = torch.corrcoef(y_pred.T)  # 注意：需要转置输入矩阵，因为torch.corrcoef预期的输入是特征在最后一维
        avr_corr_without_diag = torch.sum(corr_matrix ** 2)

        """
        计算Corr(\hat{y}_i, y)的和，即因子的IC
        """
        # 将true和pred沿着第二维度拼接，以便它们可以被torch.corrcoef()处理
        # 这里true需要先扩展到pred的形状，使之能够拼接
        data = torch.cat((y_true, y_pred), dim=1)  # 结果tensor的形状为[batches, 65]

        # 计算相关系数矩阵
        corr_matrix = torch.corrcoef(data.T)  # 注意转置，因为torch.corrcoef()计算的是行间的相关系数

        # 从相关系数矩阵中提取true与所有pred列的相关系数
        # true与pred的相关系数位于矩阵的第一行，但排除了第一个元素（true与自身的相关系数）
        true_pred_correlations = corr_matrix[0, 1:]

        '''合成所有的因子的单个的IC'''
        correlation_y_sum = torch.sum(true_pred_correlations)

        '''计算 Corr(\sum_{i=1}^64 \hat{y}_i, y),合成因子IC'''
        sum_y_pred = y_pred.sum(dim=1, keepdim=True)
        correlation_sum_y_pred = self.correlation(sum_y_pred.squeeze(1), y_true.squeeze(1))

        # 计算总的损失值
        loss = self.m1 * (avr_corr_without_diag / 4096) - (
                correlation_y_sum / 64) - self.m2 * correlation_sum_y_pred
        """
        如果是train模式，则返回单个值loss
        """
        if model.training:
            return loss
        else:
            """
            如果是valid/test模式，则返回 1.loss 2.因子间相关系数 3.均值合成因子系数
            4. 64个因子的IC组成的list
            """
            IC_average = self.correlation(sum_y_pred.squeeze(1)/64, y_true.squeeze_(1))
            """这里还有一个问题，即要不要在Corr的内部先除以n"""
        return loss, avr_corr_without_diag / 4096, IC_average

    def correlation(self, x, y):
        xm = x - torch.mean(x, dim=0, keepdim=True)
        ym = y - torch.mean(y, dim=0, keepdim=True)
        r_num = torch.dot(xm, ym)
        r_den = torch.sqrt(torch.sum(xm.pow(2), dim=0) * torch.sum(ym.pow(2), dim=0))
        r = r_num / r_den
        return r



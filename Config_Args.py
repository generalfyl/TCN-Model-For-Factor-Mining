import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='TCN', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=63, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=1, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    parser.add_argument('-data_path', type=str, default='daily_a.par', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='20-return', help='你需要预测的特征列')
    parser.add_argument('-input_size', type=int, default=7, help='你的特征个数不算时间那一列')
    parser.add_argument('-output_size', type=int, default=64, help='挖掘的因子数目')
    parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
    parser.add_argument('-model_dim', type=list, default=[30, 27, 25, 27, 30], help='这个地方是这个TCN卷积的关键部分,它代表了TCN的层数我这里输'
                                                                                    '入list中包含三个元素那么我的TCN就是三层，这个根据你的数据复杂度来设置'
                                                                                    '层数越多对应数据越复杂但是不要超过5层')

    # learning
    parser.add_argument('-lr', type=float, default=0.001, help="学习率")
    parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
    parser.add_argument('-epochs', type=int, default=60, help="训练轮次")
    parser.add_argument('-batch_size', type=int, default=32, help="批次大小")
    parser.add_argument('-save_path', type=str, default='models')

    # model
    parser.add_argument('-hidden_size', type=int, default=64, help="隐藏层单元数")
    parser.add_argument('-kernel_sizes', type=int, default=2)
    parser.add_argument('-layer_num', type=int, default=1)
    # device
    parser.add_argument('-use_gpu', type=bool, default=True)
    # parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")

    return parser.parse_args()

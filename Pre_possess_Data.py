import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset


def create_inout_sequences(input_data, tw, pre_len):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    scaler = MinMaxScaler()
    for i in range(L - tw):
        train_seq_first_five = input_data[i:i + tw, 0:5] / input_data[i,3]

        train_seq_last_two = input_data[i:i + tw, 5:7]
        scaler.fit(train_seq_last_two)
        train_seq_normalized_last_two = scaler.transform(train_seq_last_two)
        train_seq_normalized = torch.cat((train_seq_first_five, train_seq_normalized_last_two), dim=1)
        if (i + tw + pre_len) > len(input_data):
            break
        train_label = input_data[:, -1:][i + tw - pre_len:i + tw]  # + pre_len
        inout_seq.append((train_seq_normalized, train_label))
    return inout_seq


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return sequence, label


class MinMaxScaler():
    def __init__(self):
        self.min = None
        self.max = None
        self.range = None

    def fit(self, data):
        self.min, _ = torch.min(data, dim=0)
        self.max, _ = torch.max(data, dim=0)
        self.range = self.max - self.min

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * self.range) + self.min


def create_dataloader(config, start_dates, end_dates):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    pd.set_option('mode.chained_assignment', None)
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口
    batch_size = config.batch_size
    """
    先是尝试创造一个code list，存储股票代码，然后再去对股票代码做for循环，读取该只股票的数据，计算VWAP，然后存到datastock.csv
    再对datastock.csv读取，做“将特征列移到末尾” + 转化为values的操作
    然后动用create_inout_sequence做存储，做三个存进 xxx_inout_sequence的操作

    将会遇到的难题：train/ valid/ test划分需要严格时间对齐
    """
    # ddtt = pd.read_parquet('daily_processed_A.par')
    #
    # # 按股票代码分组，并对每个分组应用删除 st 操作
    # def remove_st_data(group):
    #     first_st_index = group[group['is_st'] == 1].index.min()  # 找到第一次变成 "st" 股票的日期的索引
    #     if pd.notna(first_st_index):
    #         if (first_st_index - group.index[0]) <= 83:
    #             return pd.DataFrame()
    #         group.drop(group.index[first_st_index - group.index[0]:], inplace=True)  # 删除从第一次变成 "st" 股票的日期开始的所有行
    #
    #     return group
    #
    # # 按股票代码分组，并对每个分组应用 remove_st_data 函数
    # ddtt_filtered = ddtt.groupby('wind_code').apply(remove_st_data)
    # ddtt_filtered.reset_index(drop=True, inplace=True)
    #
    # def calculate_20day_return(df):
    #     close_prices_today = df['close_price']
    #     close_prices_20days_later = df['close_price'].shift(-20)
    #     # close_prices_20days_later.drop(close_prices_20days_later.tail(20).index, inplace=True)
    #     return_20day = ((close_prices_20days_later / close_prices_today) - 1)
    #     """
    #     这里rank是20-return对做标准化
    #     """
    #     return return_20day.tolist()
    #
    # def label_scaling(group):
    #     group['20_day_return'] = group['20_day_return'].rank(pct=True)
    #     return group
    # # 按照股票代码分组
    # grouped = ddtt_filtered.groupby('wind_code')
    # # 对每个股票计算未来20日收益率
    # returns = grouped.apply(calculate_20day_return).explode()
    # # 将收益率拼接为新的列
    # ddtt_filtered['20_day_return'] = returns.values
    #
    # # 对每只股票删除最后20行数据
    # ddtt_filtered.dropna(subset=['20_day_return'], inplace=True)
    # ddtt_filtered.reset_index(drop=True, inplace=True)
    # ddtt_filtered.drop(['is_st'], axis=1, inplace=True)
    # ddtt_filtered = ddtt_filtered.astype({col: float for col in ddtt_filtered.columns
    #                                       if col not in ['trading_date', 'wind_code']})
    #
    # """
    # label根据截面标准化
    # """
    # ddtt_filtered = ddtt_filtered.groupby('trading_date').apply(label_scaling)
    #
    #
    #
    # ddtt_filtered.to_parquet('daily_processed_Filtered.par')
    """
    从这里开始，ddtt_filtered = pd.read_parquet('daily_processed_Filtered.par')
    前面的全部注释掉算了
    """
    ddtt_filtered = pd.read_parquet('daily_processed_Filtered.par')
    ddtt_filtered['trading_date'] = pd.to_datetime(ddtt_filtered['trading_date'])
    grouped = ddtt_filtered.groupby('wind_code')
    train_dataset_list, test_dataset_list, valid_dataset_list = [], [], []
    start_dates = start_dates
    end_dates = end_dates
    # end_dates = ['2018-12-31', '2019-12-31', '2023-09-30']
    for wind_code, group_data in grouped:
        dataStock = group_data
        dataStock.reset_index(drop=True, inplace=True)
        """先把代码drop掉"""
        dataStock.drop(['wind_code'], axis=1, inplace=True)
        # dataStock['trading_date'] = pd.to_datetime(dataStock['trading_date'])
        # dataStock.set_index('trading_date', inplace=True)
        df1 = dataStock.loc[(dataStock['trading_date'] >= start_dates[0]) & (dataStock['trading_date'] <= end_dates[0])]
        df2 = dataStock.loc[(dataStock['trading_date'] >= start_dates[1]) & (dataStock['trading_date'] <= end_dates[1])]
        df3 = dataStock.loc[(dataStock['trading_date'] >= start_dates[2]) & (dataStock['trading_date'] <= end_dates[2])]
        if len(df1) >= 63:
            df1.drop(['trading_date'], axis=1, inplace=True)
            df_np = np.array(df1).astype(np.float32)
            train_data = torch.tensor(df_np)
            train_inout_seq = create_inout_sequences(train_data, train_window, pre_len)
            train_dataset_list.append(train_inout_seq)
        if len(df2) >= 63:
            df2.drop(['trading_date'], axis=1, inplace=True)
            df_np = np.array(df2).astype(np.float32)
            valid_data = torch.tensor(df_np)
            valid_inout_seq = create_inout_sequences(valid_data, train_window, pre_len)
            valid_dataset_list.append(valid_inout_seq)
        if len(df3) >= 63:
            df3.drop(['trading_date'], axis=1, inplace=True)
            df_np = np.array(df3).astype(np.float32)
            test_data = torch.tensor(df_np)
            test_inout_seq = create_inout_sequences(test_data, train_window, pre_len)
            test_dataset_list.append(test_inout_seq)
        '''相当于以下inout_seq才是输入的数据，所以scalar函数尽量写在create_inout_sequences函数里面'''
        """创建dataset, 然后ConcatDataset"""

    train_dataset_list = list(filter(lambda dataset: len(dataset) > 0, train_dataset_list))
    test_dataset_list = list(filter(lambda dataset: len(dataset) > 0, test_dataset_list))
    valid_dataset_list = list(filter(lambda dataset: len(dataset) > 0, valid_dataset_list))
    # with open('train_data_list.pkl', 'wb') as f:
    #     pickle.dump(train_data_list, f)
    #
    # with open('valid_data_list.pkl', 'wb') as f:
    #     pickle.dump(valid_data_list, f)
    #
    # with open('test_data_list.pkl', 'wb') as f:
    #     pickle.dump(test_data_list, f)

    train_dataset = ConcatDataset(train_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)
    valid_dataset = ConcatDataset(valid_dataset_list)
    """
    保存到dataset到本地，方便下次使用：
    """

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=True)
    pd.set_option('mode.chained_assignment', 'warn')
    print("通过滑动窗口共有训练集数据：", len(train_dataset), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_dataset), "转化为批次数据:", len(test_loader))
    print("通过滑动窗口共有验证集数据：", len(valid_dataset), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader

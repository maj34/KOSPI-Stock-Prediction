import pandas as pd
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, Dataset


class StockDataset(Dataset):
    def __init__(self, file_path, time_step=10, train_flag=True):
        # csv 데이터 불러오기
        with open(file_path, "r", encoding="utf-8") as fp:
            data_pd = pd.read_csv(fp)

        self.train_flag = train_flag  # 학습용 데이터를 True로 설정
        self.data_train_ratio = 0.9  # 90%를 학습용 데이터로 사용
        self.T = time_step

        # 학습용 데이터인 경우
        if train_flag:
            self.data_len = int(self.data_train_ratio * len(data_pd))
            data_all = np.array(data_pd['close'])  # KOSPI 종가 데이터 셋 활용
            data_all = (data_all - np.mean(data_all)) / np.std(data_all)  # 데이터 셋 표준화
            self.data = data_all[ : self.data_len]

        # 평가용 데이터인 경우
        else:
            self.data_len = int((1-self.data_train_ratio) * len(data_pd))
            data_all = np.array(data_pd['close'])
            data_all = (data_all - np.mean(data_all)) / np.std(data_all)
            self.data = data_all[-self.data_len : ]

        print("data len:{}".format(self.data_len))  # 학습 시 학습/평가 데이터 개수 출력

    def __len__(self):
        return self.data_len - self.T

    def __getitem__(self, idx):
        return self.data[idx : idx + self.T], self.data[idx + self.T]
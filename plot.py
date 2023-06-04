import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import tqdm
import math
from models import RNN, LSTM  # models.py에서 RNN, LSTM 모델 가져오기
from dataset import StockDataset  # dataset.py에서 KOSPI 데이터 가져오기
from torch.utils.data import DataLoader

stock_file = './stocks/KOSPI.csv'
loadckpt = './checkpoints/KOSPI_TRANSFORMER/checkpoint_010.ckpt'
plotdir = './plot_figure/KOSPI_TRANSFORMER'

def eval_plot():

    # 데이터 셋 설정
    dataset_test = StockDataset(file_path = stock_file, time_step = 10, train_flag=False)  # 평가 단계이므로 train_flag=False
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    loader = tqdm.tqdm(test_loader)

    # RNN, LSTM 중 원하는 모델 설정
    model = RNN(rnn_layer=2, input_size=1, hidden_size=4)
    model = LSTM(lstm_layer=2, input_dim=1, hidden_size=8)

    model.load_state_dict(torch.load(loadckpt))  # 학습 시 저장한 checkpoints 불러오기

    preds = []
    labels = []

    for idx, (data, label) in enumerate(loader):
        data, label = data.float(), label.float()
        output = model(data)  # batch_size(64) 별로 전체 데이터를 나누어서 평가 진행
        preds += (output.detach().tolist())  # 예측값 preds를 리스트에 추가
        labels += (label.detach().tolist())  # 정닶값 label을 리스트에 추가

    fig, ax = plt.subplots()
    data_x = list(range(len(preds)))

    ax.plot(data_x[-60:], preds[-60:], label='predict', color='red')  # 빨간색으로 예측값 lineplot 생성
    ax.plot(data_x[-60:], labels[-60:],label='ground truth', color='blue')  # 파란색으로 정답값 lineplot 생성
    plt.legend()
    plt.show()
    plt.savefig('{}/KOSPI_TRANSFORMER_epoch_010.png'.format(plotdir))  # 각 figure를 png 형태로 저장

if __name__ == '__main__':
    eval_plot()
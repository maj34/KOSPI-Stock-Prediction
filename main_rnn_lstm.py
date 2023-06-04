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
from tensorboardX import SummaryWriter

checkpointdir = './checkpoints'
stock_file = './stocks/KOSPI.csv'
logger = SummaryWriter(checkpointdir)


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, reduction='mean')  # MSE Loss 사용
    return loss


def train(model, dataloader, optimizer):
    # 모델을 학습 단계로 설정
    model.train()
    loader = tqdm.tqdm(dataloader)  

    loss_epoch = 0
    
    for idx, (data, label) in enumerate(loader):    
        data, label = data.float(), label.float()
        output = model(data)  # batch_size(64) 별로 전체 데이터를 나누어서 학습 진행
        optimizer.zero_grad()  # epoch 한 번의 학습이 완료되어지면 gradient를 항상 0으로 초기화
        loss = l2_loss(output, label)  # 손싫 함수는 MSE Loss로 설정
        loss.backward()  # 역전파 진행
        optimizer.step()  # 역전파 단계에서 수집된 변화도로 매개변수 조정
        loss_epoch += loss.detach().item()  # 각 epoch 별 loss 출력
    loss_epoch /= len(loader)
    return loss_epoch


def eval(model, dataloader):
    # 모델을 평가 단계로 설정
    model.eval()
    loader = tqdm.tqdm(dataloader)

    loss_epoch = 0

    for idx, (data, label) in enumerate(loader):
        data, label = data.float(), label.float()
        output = model(data)  # batch_size(64) 별로 전체 데이터를 나누어서 평가 진행
        loss = l2_loss(output, label)  # 손싫 함수는 MSE Loss로 설정
        loss_epoch += loss.detach().item()  # 각 epoch 별 loss 출력
    loss_epoch /= len(loader)
    return loss_epoch


def main():
    # 데이터 셋 설정
    dataset_train = StockDataset(file_path = stock_file, time_step = 10)
    dataset_test = StockDataset(file_path = stock_file, time_step = 10, train_flag=False)

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    
    # RNN, LSTM 중 원하는 모델 설정
    model = RNN(rnn_layer=2, input_size=1, hidden_size=4)
    model = LSTM(lstm_layer=2, input_dim=1, hidden_size=8)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # optimizer는 Adam으로 설정
    total_epoch = 101  # 100 epoch까지 checkpoints 및 결과 plot 생성

    for epoch_idx in range(total_epoch):
        train_loss = train(model, train_loader, optimizer)  # 학습을 진행해 train_loss 계산
        print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))  # 각 step마다 loss 출력
        logger.add_scalar('Train/Loss', train_loss, epoch_idx)  # tensorboardX의 add_scaler를 활용해 각 step에서의 학습 손실 값을 스칼라 형태로 기록

        if epoch_idx % 10 == 0:  # 10 에폭마다 평가용 데이터로 검증
            eval_loss = eval(model, test_loader)  # 평가를 진행해 eval_loss 계산
            print("stage: test, epoch:{:5d}, loss:{}".format(epoch_idx, eval_loss))  # 10번의 step마다 loss 출력
            torch.save(model.state_dict(), "{}/KOSPI_LSTM/checkpoint_{:0>3}.ckpt".format(checkpointdir, epoch_idx))  # 모델의 학습 가중치를 checkpoints로 저장
            logger.add_scalar('Test/Loss', eval_loss, epoch_idx)  # tensorboardX의 add_scaler를 활용해 각 step에서의 평가 손실 값을 스칼라 형태로 기록


if __name__ == '__main__':
    main()

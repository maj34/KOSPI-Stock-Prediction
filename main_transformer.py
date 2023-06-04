import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 

import tqdm
from torch.autograd import Variable
import argparse
import math
import torch.nn.functional as F

checkpointdir = './checkpoints/KOSPI_TRANSFORMER'
plotdir = './plot_figure/KOSPI_TRANSFORMER'

torch.random.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser("Transformer-LSTM")
parser.add_argument("-data_path", type=str, default="./stocks/KOSPI.csv", help="dataset path")

args = parser.parse_args()
time_step = 10


class PositionalEncoding(nn.Module):
    # Transformer의 Positional Encoding 정의
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # 입력 값의 최대 길이만큼 0인 텐서 값 생성
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # sin 주파수 기능 사용
        pe[:, 1::2] = torch.cos(position * div_term)  # cos 주파수 기능 사용
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]  # 각 입력값마다 positional encoding 진행


class TransAm(nn.Module):
    # Transformer Encoder 구조 정의
    def __init__(self,feature_size=64, num_layers=6, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        
        # torch.nn 모듈에 있는 encoder 및 decoder 레이어 설정
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    # decoder 가중치 초기화
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    # decoder에서 다음 값 예측 시 sequence의 다음 값을 모르게 하기 위해 마스킹 함수 정의
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    # 앞서 정의한 함수를 사용해 Transformer Encoder의 순전파 진행
    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask  # mask를 씌워 Mult-head Attn 수행
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class AttnDecoder(nn.Module):
    # Transformer Decoder 구조 정의
    def __init__(self, code_hidden_size, hidden_size, time_step):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        # Model, Layer, Activation Fuction 정의
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        self.tanh = nn.Tanh()
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size,num_layers=1)
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1)

    # 인풋 사이즈의 0값을 갖는 초기 텐서 생성
    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        return Variable(zero_tensor)

    # hidden layer embedding 순서값을 바꾸어 줌
    def embedding_hidden(self, x):
        return x.permute(1, 0, 2)

    # 앞서 정의한 함수를 사용해 Transformer Decoder의 순전파 진행
    def forward(self, h, y_seq):
        h_ = h.transpose(0,1) 
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        h_0 = self.init_variable(1,batch_size, self.hidden_size)
        h_ = torch.cat((h_0,h_),dim=0)

        for t in range(self.T):
            x = torch.cat((d,h_[t,:,:].unsqueeze(0)), 2)
            h1 = self.attn1(x)
            _, states = self.lstm(y_seq[:,t].unsqueeze(0).unsqueeze(2), (h1, s))
            d = states[0]
            s = states[1]
        y_res = self.fc2(self.fc1(torch.cat((d.squeeze(0), h_[-1,:,:]), dim=1)))
        return y_res


class StockDataset(Dataset):
    def __init__(self, file_path, T=time_step, train_flag=True):
        # KOSPI 데이터 불러오기
        with open(file_path, "r", encoding="utf-8") as fp:
            data_pd = pd.read_csv(fp)
        self.train_flag = train_flag  # 학습용 데이터를 True로 설정
        self.data_train_ratio = 0.9  # 90%를 학습용 데이터로 사용
        self.T = T
        
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
            data_all = (data_all-np.mean(data_all))/np.std(data_all)
            self.data = data_all[-self.data_len:]
        print("data len:{}".format(self.data_len))  # 학습 시 학습/평가 데이터 개수 출력

    def __len__(self):
        return self.data_len-self.T

    def __getitem__(self, idx):
        return self.data[idx:idx+self.T], self.data[idx+self.T]


def l2_loss(pred, label):
    loss = torch.nn.functional.mse_loss(pred, label, size_average=True)  # MSE Loss 사용
    return loss


def train_once(encoder, decoder, dataloader, encoder_optim, decoder_optim):
    # encoder, decoder 각각의 모델을 학습 단계로 설정
    encoder.train()
    decoder.train()
    loader = tqdm.tqdm(dataloader)

    loss_epoch = 0

    for idx, (data, label) in enumerate(loader):
        data_x = data.unsqueeze(2)
        data_tran = data_x.transpose(0,1)
        data_x, label ,data_y= data_tran.float(), label.float() ,data.float()
        code_hidden = encoder(data_x)  # batch_size(64) 별로 전체 데이터를 나누어서 encoder 학습 진행
        code_hidden = code_hidden.transpose(0,1)
        output = decoder(code_hidden, data_y)  # batch_size(64) 별로 전체 데이터를 나누어서 decoder 학습 진행

        encoder_optim.zero_grad()  # epoch 한 번의 학습이 완료되어지면 gradient를 항상 0으로 초기화
        decoder_optim.zero_grad()
        loss = l2_loss(output.squeeze(1), label)  # 손실 함수는 MSE Loss로 설정
        loss.backward()  # 역전파 진행
        
        encoder_optim.step()  # 역전파 단계에서 수집된 변화도로 매개변수 조정
        decoder_optim.step()
        loss_epoch += loss.detach().item()  # 각 epoch 별 loss 출력
    loss_epoch /= len(loader)
    return loss_epoch


def eval_once(encoder,decoder, dataloader):
    # encoder, decoder 각각의 모델을 평가 단계로 설정
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)

    loss_epoch = 0

    preds = []
    labels = []

    for idx, (data, label) in enumerate(loader):
        # data: batch, time x 1
        data_x = data.unsqueeze(2)
        data_x, label ,data_y= data_x.float(), label.float(), data.float()
        code_hidden = encoder(data_x)  # encoder를 거쳐 code_hidden 출력
        output = decoder(code_hidden, data_y).squeeze(1)  # decoder를 거쳐 output 출력
        loss = l2_loss(output, label)  # 손실함수는 MSE Loss로 설정
        loss_epoch += loss.detach().item()  # 각 epoch 별 loss 출력
        preds += (output.detach().tolist())  # 예측값 preds를 리스트에 추가
        labels += (label.detach().tolist())  # 정답값 label을 리스트에 추가
    
    preds = torch.Tensor(preds)  # 각 예측값과 정답값을 Tensor 형태로 변환
    labels = torch.Tensor(labels)
    
    # 각 예측값과 정답값 계산
    pred1 = preds[:-1]
    pred2 = preds[1:]
    pred_ = preds[1:]>preds[:-1]
    label1 = labels[:-1]
    label2 = labels[1:]
    label_ = labels[1:]>labels[:-1]
    
    accuracy = (label_ == pred_).sum() / len(pred1)  # 앞서 정의한 예측값과 정답값을 기준으로 accuracy 계산
    loss_epoch /= len(loader)  # 앞서 정의한 예측값과 정답값을 기준으로 loss 값 계산
    return loss_epoch, accuracy


def eval_plot(encoder, decoder, dataloader):
    dataloader.shuffle = False  # 평가 단계이므로 shuffle=False로 설정
    preds = []
    labels = []
    # encoder, decoder 각각의 모델을 평가 단계로 설정
    encoder.eval()
    decoder.eval()
    loader = tqdm.tqdm(dataloader)

    for idx, (data, label) in enumerate(loader):
        data_x = data.unsqueeze(2)
        data_x, label, data_y = data_x.float(), label.float(), data.float()
        code_hidden = encoder(data_x)  # encoder를 거쳐 core_hidden 출력
        output = decoder(code_hidden, data_y)  # decoder를 거쳐 output 출력
        preds += (output.detach().tolist())  # 예측값 preds를 리스트에 추가
        labels += (label.detach().tolist())  # 정답값 label을 리스트에 추가

    fig, ax = plt.subplots()
    data_x = list(range(len(preds)))
    
    ax.plot(data_x, preds, label='predict', color='red')  # 빨간색으로 예측값 lineplot 생성
    ax.plot(data_x, labels,label='ground truth', color='blue')  # 파란색으로 정답값 lineplot 생성
    plt.legend()
    plt.show()
    

def main():
    # train, val 데이터 셋 불러오기
    dataset_train = StockDataset(file_path=args.data_path)
    dataset_val = StockDataset(file_path=args.data_path, train_flag=False
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=64, shuffle=False)

    encoder = TransAm()  # encoder는 앞서 정의한 TransAM 함수 사용
    decoder = AttnDecoder(code_hidden_size=64, hidden_size=64, time_step=time_step)  # decoder는 앞서 정의한 AttnDecoder 함수 사용
    
    # 각 encoder, decoder의 optimizer는 Adam으로 설정
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=0.001)

    total_epoch = 101  # 100 epoch까지 checkpoints 및 결과 plot 생성

    for epoch_idx in range(total_epoch):
        train_loss = train_once(encoder, decoder, train_loader, encoder_optim, decoder_optim)
        print("stage: train, epoch:{:5d}, loss:{}".format(epoch_idx, train_loss))
        
        if epoch_idx % 10 == 0:  # 10 epoch마다 평가용 데이터로 검증
            eval_loss, accuracy = eval_once(encoder, decoder, val_loader)  # 평가를 진행해 eval_loss와 accuracy 계산
            
            print("##### stage: test, epoch:{:5d}, loss:{}, accuracy:{}".format(epoch_idx, eval_loss, accuracy))  # 10번의 step마다 loss 출력
            eval_plot(encoder,decoder, val_loader)
            torch.save(encoder.state_dict(), "{}/checkpoint_{:0>3}.ckpt".format(checkpointdir, epoch_idx))  # 모델의 학습 가중치를 checkpoints로 저장
            plt.savefig("{}/KOSPI_TRANSFORMER_epoch_{}.png".format(plotdir, epoch_idx))  # 각 figure를 png 형태로 저장


if __name__ == "__main__":
    main()



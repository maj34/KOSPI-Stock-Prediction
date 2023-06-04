import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

# RNN 모델 구조
class RNN(nn.Module):
    def __init__(self, rnn_layer=2, input_size=1, hidden_size=4):  # Hyperparameter 설정
        super(RNN, self).__init__()
        self.rnn_layer = rnn_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size = self.input_size,  # 각 문자의 벡터 사이즈
            hidden_size=self.hidden_size,  # RNN hidden neuron 개수
            num_layers=self.rnn_layer,  # RNN hidden layer 개수
            batch_first=True  # batch_size를 첫 번째 입력 값으로 들어가게 함
            # [sequence, batch_size, input_size] -> [batch_size, sequence, input_size]
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def init_hidden(self, x):
        batch_size = x.shape[0]
        init_h = torch.zeros(self.rnn_layer, batch_size, self.hidden_size, device=x.device).requires_grad_()
        # rnn_layer, batch_size, hidden_size 사아즈의 0값을 갖는 초기 텐서 생성
        # requries_grad=True로 설정된 모든 텐서에 대한 gradient 계산
        return init_h

    def forward(self, x, h=None):
        x = x.unsqueeze(2)
        h = h if h else self.init_hidden(x)  # 처음 입력값은 init_hidden으로, 나머지는 각 hidden_size로 설정
        out, h = self.rnn(x, h)  # input_size, hidden_size 값을 넣어 결과 값 생성
        out = self.fc(out[:, -1, :]).squeeze(1)  # fc layer 통과
        return out


# LSTM 모델 구조
class LSTM(nn.Module):
    def __init__(self, lstm_layer=2, input_dim=1, hidden_size=8, rnn_unit=8): # Hyperparameter 설정
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # LSTM hidden neuron 개수
        self.lstm_layer = lstm_layer  # LSTM hidden layer 개수
        self.rnn_unit = rnn_unit  # LSTM에 사용되는 RNN 유닛 개수
        self.emb_layer = nn.Linear(input_dim, hidden_size)  # input_dim -> hidde_size로 선형 연산
        self.out_layer = nn.Linear(hidden_size, input_dim)  # hidden_size -> input_dim으로 선형 연산
        self.lstm = nn.LSTM(input_size=rnn_unit, hidden_size=hidden_size, num_layers=self.lstm_layer, batch_first=True)  # batch_size를 첫 번째 입력 값으로 들어가게 함

    def init_hidden(self, x):
        batch_size = x.shape[0]
        init_h = (torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device),
                  torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device))  
        # 해당 사이즈의 0값을 갖는 초기 텐서 생성
        return init_h

    def forward(self, x, h=None):
        # batch x time x dim
        x = x.unsqueeze(2)
        h = h if h else self.init_hidden(x)  # 처음 입력값은 init_hidden으로, 나머지는 각 hidden_size로 설정
        x = self.emb_layer(x)
        output, hidden = self.lstm(x, h)  # embedding layer를 거친 input과 hidden을 lstm 모델에 넣어 output, hidden 출력
        out = self.out_layer(output[:,-1,:]).squeeze(1)
        return out
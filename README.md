# KOSPI-Stock-Prediction
&nbsp;&nbsp;&nbsp;&nbsp; 해당 과제는 **코스피 데이터 셋을 활용해 앞으로의 주가 등락을 예측하는 Task**이다. 코스피(KOSPI)란 **KO**rea Composite **S**tock **P**rice **I**ndex의 약어로 우리나라의 주가 동향을 대표하는 Index이며 증권 거래소에 상장된 종목들의 주식 가격을 종합적으로 표시한 수치로 시장 전체의 주가 움직임을 이해할 수 있는 지표이다. 전체 데이터 셋은 1981.05.01부터 2022.08.31까지의 일별 KOSPI 데이터로 이루어져 있으며 Open(시가), Close(종가), High(고가), Low(저가), Volume(거래량) 열을 크롤링하여 구축하였다. 그 중 분석에 사용한 열은 **Close(종가) 데이터이며 이를 기반으로 시계열 분석을 진행**하였다.

<br/>

## Environment
```
Python 3.8
PyTorch 1.8
Cudatoolkit 11.1.
```

<br/>

## Install
```
pip3 install -r requirements.txt
```

<br/>

## RNN/LSTM Train & Evaluate
```
python3 main_rnn_lstm.py
```

||Train Loss|Evaluate Loss|
|------|---|---|
|RNN_epoch_0|0.60377|2.29561|
|RNN_epoch_10|0.00169|0.17583|
|RNN_epoch_100|0.00036|0.02924|

||Train Loss|Evaluate Loss|
|------|---|---|
|LSTM_epoch_0|0.31240|0.76642|
|LSTM_epoch_10|0.00110|0.09611|
|LSTM_epoch_100|0.00036|0.01363|

<br/>

## RNN/LSTM Plot
```
python3 plot.py
```

<img src='https://github.com/maj34/KOSPI-Stock-Prediction/assets/75362328/59d0e9c8-0556-447f-8a49-bb906513f9e4' width='80%' height='100%'>

<img src='https://github.com/maj34/KOSPI-Stock-Prediction/assets/75362328/4bafda09-3799-444f-acba-9b56f02673a3' width='80%' height='100%'>


<br/>

## Transformer Train & Evaluate & Plot
```
python3 main_transformer.py
```

||Train Loss|Evaluate Loss|
|------|---|---|
|Transformer_epoch_0|0.08487|0.03479|
|Transformer_epoch_10|0.00070|0.00517|
|Transformer_epoch_100|0.00037|0.00149|

<img src='https://github.com/maj34/KOSPI-Stock-Prediction/assets/75362328/7035c59b-8ec8-4ca0-a726-eceb143592db' width='80%' height='100%'>

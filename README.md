# AICUP-2023 隱私保護與醫學數據標準化競賽：解碼臨床病例、讓數據說故事
隊伍 TEAM_3964

## 環境
* 作業系統: Ubuntu 22.04
* Cuda版本: 11.8

## 安裝
需要安裝以下函式庫:
1.	glob
2.	collections
3.	pandas
4.	transformers
5.	torch
6.	peft
7.	re
8.	random
9.	tqdm
10.	datasets

## 資料
將 第一階段資料集 與 第二階段資料集 解壓縮進 Data資料夾

說明:Data資料夾中的duration.txt跟phone.txt為ChatGpt生成出的訓練資料

## 執行
* 確定已將第一階段資料集與第二階段資料集放入Data資料夾
* 先執行資料前處理程式:Train-Data_pre-processing.py
* 訓練及預測: train&predict.ipynb

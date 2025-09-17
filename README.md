# Novel-Seq2Seq-AT-experiments
Modified version of Novel Seq2Seq for AT dataset (3-block, LSTM decoder, window=24, horizon=12)


This repository contains a **modified version** of the code implementation from the paper:  
**"A Novel Sequence-to-Sequence Based Deep Learning Model for Multistep Load Forecasting" (IEEE TNNLS, 2025)**.  
👉 [Original Paper (IEEE)](https://ieeexplore.ieee.org/abstract/document/10409277/)  
👉 [Original GitHub Repository](https://github.com/BaiRuic/Novel-Seq2Seq-Module)

## 🔹 What’s Different in This Repository?
- Dataset: **AT dataset (Australia)**  
- Forecasting horizon: **12 steps ahead**  
- Window size: **24**  
- Number of basic blocks: **3**  
- Decoder modified to use **LSTM** instead of GRU  

These modifications were made to replicate and extend the experimental setup described in the paper, focusing on the AT dataset with different architectural choices.

## 🔹 Purpose
It is intended to:
- Provide reproducible experiments based on the original work,  
- Share improvements and comparisons on real-world datasets,  
- Serve as a reference for other researchers working on STLF.

### 📊 Results
- Achieved MSE ≈ **0.0083**, MAPE ≈ **4.66%** on AT dataset.  
- Training curve shows smooth convergence with low validation loss. 
Below is the training and validation loss curve for the AT dataset (3-block, LSTM decoder, window=24, horizon=12):
![Training vs Validation Loss](results/loss_curve.png)



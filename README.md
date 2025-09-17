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

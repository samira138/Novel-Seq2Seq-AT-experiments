
This repository contains the implementation of a Short-Term Load Forecasting (STLF) model using a TCN–Seq2Seq architecture.
The goal is to predict 12 hours ahead using the previous 24 hours of data.

The project specifically explores the effect of applying different scaling strategies on:

Model stability

Error metrics (MSE, MAPE, SMAPE)

Temporal oscillation behavior of the target series

📊 Dataset

The dataset used in this project is the AT electricity load dataset.

It includes:

Hourly load values

Meteorological variables

Temporal/cyclical features

Rolling and lag features (engineered)

Input sequence length: 24 hours
Forecasting horizon: 12 hours

🧩 Feature Engineering

We construct a total of 12 input features, divided into four families:

1) Weather Features

temperature

wind_speed

2) Temporal Cyclical Encodings

(Sin/Cos representation of time)

sin_hour, cos_hour

sin_dow, cos_dow

sin_month, cos_month

3) Binary / Indicator Feature

is_weekend

4) Lag & Rolling Statistical Features

load_lag1 (1 hour lag)

load_lag24 (same hour previous day)

load_ma24 (24-hour moving average)

load_std24 (24-hour rolling standard deviation)

These features allow the model to capture:

Short-term and long-term dependencies

Daily/weekly/seasonal periodicity

Weather-driven load behavior

Local volatility patterns

🧪 Scaling Strategy (Critical Part of This Study)

To achieve stability in the TCN–Seq2Seq architecture, a hybrid scaling strategy is used:

X (features) → StandardScaler

Reason:

Preserves natural variance

Keeps cyclical/rolling/weather feature distributions realistic

Prevents over-flattening of heterogeneous feature ranges

Highly stable for convolutional temporal encoders

y (target load) → MinMaxScaler

Reason:

Load is heavy-tailed and oscillatory

StandardScaler amplifies oscillation noise in multi-step forecasting

MinMax keeps target range bounded, smooth, and stable

Prevents SMAPE explosion and validation instability

This combination was found to be the most stable and most accurate, based on experiments.

🧠 Model Architecture: TCN–Seq2Seq

The model consists of:

1) Encoder (TCN)

Dilated causal convolutions

Large receptive field

Excellent at capturing multi-scale temporal patterns

2) Decoder (LSTM-based / GRU-based depending on configuration)

Sequential decoding for 12-step forecasting

The architecture is sensitive to target scaling, so correct normalization of y is essential.
📈 Key Results

Using:

X = StandardScaler

y = MinMaxScaler

TCN–Seq2Seq model (24 → 12)

We achieved:

MSE = 0.0024

MAPE = 3.78%

SMAPE = 7.27%

Stable training & validation loss curves

🎯 Conclusion

This project demonstrates that:

StandardScaler is optimal for heterogeneous input features,

BUT load (y) must be MinMax-scaled to avoid SMAPE instability.

The hybrid scaling approach results in a far more stable model and significantly improves SMAPE without sacrificing MAPE or MS

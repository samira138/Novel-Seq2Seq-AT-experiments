⚙️ Data & Normalization Summary (24→12 Forecasting)

📊 We use the AT load dataset with a sliding window of 24 input hours to forecast the next 12 hours (24→12 multistep horizon).

🔧 All input features (weather, cyclical time encodings, binary flags, lag features, rolling stats) are scaled using Min–Max (0–1) following the preprocessing pipeline in Lu et al., 2025.

🔌 The target load values for the 12-step decoder are normalized with the same Min–Max scaler to ensure consistency during Seq2Seq training.

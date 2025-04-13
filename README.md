# 🧠 TinyLLMs to Transformers — My Custom LLM Roadmap

Welcome to my personal LLM practice repository, where I’m documenting my journey through building, training, and experimenting with increasingly complex models — from scratch-built tokenizers to LSTM-based stock prediction systems. This repo reflects both my theoretical understanding and hands-on implementations.

---

## 📈 Current Stage: Stage 3 – MemoryLLM (LSTM-based Stock Prediction)

### 🚀 Project: `MemoryLLM`
- **Goal**: Predict future stock prices of Google (GOOG) using historical price and technical indicators.
- **Tech Stack**: `Python`, `TensorFlow/Keras`, `pandas`, `matplotlib`
- **Architecture**: LSTM-based deep learning model.
- **Indicators Used**:
  - Moving Averages (SMA/EMA)
  - RSI
  - MACD
  - Bollinger Bands
- **Evaluation Metrics**:
  - RMSE
  - MAE
  - MAPE
  - R² Score
- **Results**:
  - Effective short-term trend prediction
  - Visualization of predictions vs actual values
- **Challenges**:
  - Struggled with long-range dependencies
  - LSTM hidden states are hard to interpret
  - Performance bottleneck due to sequential training

📄 See more: [`LLM5 MemoryLLM.md`](./LLM5%20MemoryLLM.md)

---

## 📚 Roadmap Overview

| Stage | Model Type | Project | Description |
|-------|------------|---------|-------------|
| ✅ 1 | Token-level RNN | `TinyLLM` | Basic character prediction using a mini-RNN |
| ✅ 2 | Word-level RNN | `SmaLLM` | Word prediction using word embeddings |
| ✅ 3 | LSTM (Memory) | `MemoryLLM` | Stock prediction using LSTM with indicators |
| ⏳ 4 | Transformer (Attention) | `CodeLLM` | Planned: Build Transformer-based models |
| ⏳ 5 | Pretrained Models (LLMs) | `LLMind` | Planned: Explore fine-tuning FinGPT / BERT |
| ⏳ 6 | Hybrid + RL + Agentic | TBD | Planned: Agent-based LLMs & multi-modal inputs |

---

## 🔜 Next Stage: Stage 4 – CodeLLM (Transformer Encoder-Decoder)

### 🎯 Goals
- Build a **Transformer encoder-decoder** model for time-series forecasting.
- Integrate **positional encoding**, **multi-head attention**, and **feed-forward layers**.
- Replace recurrence with **global attention** for better performance on long sequences.

### 💡 Why Transformers?
- Capture long-range dependencies better than LSTM
- Faster training through parallelization
- Easier interpretability via attention maps

### 🛠️ Upcoming Projects:
- ✅ Rewriting GOOG stock predictor using a custom Transformer
- 🔄 Comparing LSTM vs Transformer performance (accuracy, speed, interpretability)
- 🧪 Experimenting with [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT), [Informer](https://github.com/zhouhaoyi/Informer2020)

📄 See comparison: [`LLM10 Stage3 vs Stage4.md`](./LLM10%20Stage3%20vs%20Stage4.md)

---

## 🧪 Folder Structure

```
├── TinyLLM/            # Stage 1: Character-level RNN
├── SmaLLM/             # Stage 2: Word-level RNN
├── MemoryLLM/          # Stage 3: LSTM-based GOOG price predictor
├── CodeLLM/            # Stage 4: (Planned) Transformer-based forecasting
├── LLM2 RoadMap.md     # My detailed multi-stage learning plan
├── LLM5 MemoryLLM.md   # Write-up on LSTM stock prediction model
├── LLM10 Stage3 vs Stage4.md  # Deep dive comparison: LSTM vs Transformer
```

---

## 🧠 Knowledge Highlights

- ✅ **Learned**: Sequential modeling, data preprocessing, LSTM architectures, time-series indicators.
- 🔄 **Exploring**: Attention mechanisms, Transformer math, sequence-to-sequence tasks.
- ⏭ **Next**: Implement Transformer for GOOG price prediction using PyTorch or TensorFlow.

---

## ✍️ Author

**[Avrodeep Pal]**

- Machine Learning and NLP Enthusiast
- Exploring end-to-end LLM development from scratch
- Always up for a conversation on Transformers, Agents, or Fintech AI

---

## ⭐️ Star the Repo if You're Also on a Custom LLM Journey!

```
git clone https://github.com/AvrodeepPal/LLMsPractice.git
cd LLMsPractice
```

Feel free to open issues or discussions if you're following a similar path — would love to exchange ideas or collaborate!

---
```

---

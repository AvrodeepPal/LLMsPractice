# 🧠 LLMs from Scratch: First-Principles Engineering & Architecture Practice

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive, first-principles repository dedicated to mastering Large Language Models (LLMs) and sequence modeling architectures. This workspace chronicles a complete pedagogical journey from raw character-level recurrent networks and LSTM memory systems to custom Byte Pair Encoding (BPE), multi-head causal self-attention, complete GPT-2 decoder blocks, pretraining loops, and instruction fine-tuning.

---

## 📌 Table of Contents

- [🔬 Executive Overview & Philosophy](#-executive-overview--philosophy)
- [🗂️ Repository Architecture & Directory Blueprint](#️-repository-architecture--directory-blueprint)
- [🗺️ Multi-Stage Pedagogical Roadmap](#️-multi-stage-pedagogical-roadmap)
- [📐 Mathematical & Architectural Deep Dive](#-mathematical--architectural-deep-dive)
  - [1. Tokenization Paradigms & Vocabulary Engineering](#1-tokenization-paradigms--vocabulary-engineering)
  - [2. Dense Vector & Positional Embeddings](#2-dense-vector--positional-embeddings)
  - [3. Attention Mechanics: From Additive to Causal Multi-Head](#3-attention-mechanics-from-additive-to-causal-multi-head)
  - [4. Transformer Decoder Block Anatomy](#4-transformer-decoder-block-anatomy)
  - [5. Training Dynamics, Loss Formulations & Optimization](#5-training-dynamics-loss-formulations--optimization)
  - [6. Inference Decoding & Sampling Strategies](#6-inference-decoding--sampling-strategies)
- [💻 Hands-On Implementations & Codebase](#-hands-on-implementations--codebase)
  - [Self Practice Laboratory (`LLM Codes/Self/`)](#self-practice-laboratory-llm-codesself)
  - [From-Scratch GPT Series (`LLM Codes/Vizuara/`)](#from-scratch-gpt-series-llm-codesvizuara)
  - [PyTorch Lightning Decoder Suite (`LLM Codes/SQ/`)](#pytorch-lightning-decoder-suite-llm-codessq)
  - [Root Sequence Modeling Notebooks](#root-sequence-modeling-notebooks)
- [📚 Theoretical Knowledge Base (`LLM Notes/`)](#-theoretical-knowledge-base-llm-notes)
  - [GPT Architecture & Engineering Notes (`LLM Notes/GPT/`)](#gpt-architecture--engineering-notes-llm-notesgpt)
  - [Foundational Deep Dives (`LLM Notes/Vizuara/`)](#foundational-deep-dives-llm-notesvizuara)
- [📖 Landmark Research Papers Library (`Theory/`)](#-landmark-research-papers-library-theory)
- [📊 Datasets & Benchmark Corpus Catalog (`Datasets/`)](#-datasets--benchmark-corpus-catalog-datasets)
- [📑 Offline Developer Documentation (`Pages/`)](#-offline-developer-documentation-pages)
- [⚡ Installation, Environment Setup & Quickstart](#-installation-environment-setup--quickstart)
- [💡 Key Takeaways & Architectural Lessons](#-key-takeaways--architectural-lessons)
- [👤 Author & Acknowledgments](#-author--acknowledgments)

---

## 🔬 Executive Overview & Philosophy

Modern natural language processing is driven by Large Language Models built upon the Transformer architecture. While high-level frameworks like Hugging Face allow quick prototyping, true mastery requires building every component from scratch using raw tensor operations.

This repository bridges the gap between abstract mathematical theory and concrete code implementation:
1. **Low-Level Tokenization**: Implementing character-level tokenizers, word-level dictionaries, and Byte Pair Encoding (BPE) merge algorithms.
2. **Vector Space Representations**: Deriving continuous token embeddings, geometric cosine relationships, and learned 1D positional vectors.
3. **Attention Mathematics**: Constructing scaled dot-product attention, causal triangular masks, query-key-value transformations, and multi-head parallelization.
4. **Decoder Architecture**: Assembling Pre-Layer Normalization, Gaussian Error Linear Units (GeLU), two-layer Feed-Forward Networks, and residual skip connections into full GPT-2 blocks.
5. **Autoregressive Pretraining**: Aligning input and target sequences for next-token prediction, calculating cross-entropy loss, and optimizing with AdamW.
6. **Downstream Adaptation**: Transitioning pretrained weights to sequence classification tasks and multi-turn instruction fine-tuning.

---

## 🗂️ Repository Architecture & Directory Blueprint

```
LLMsPractice/
├── CodeLLM.ipynb                                # Root Transformer encoder-decoder time-series & sequence forecasting
├── StocksLSTM.ipynb                             # Root deep LSTM stock market forecasting laboratory
├── README.md                                    # Master documentation & pedagogical guide (~400 lines)
│
├── Datasets/                                    # Raw and curated corpora, financial time-series, & LLM evaluation logs
│   ├── AAPL.csv, AMZN.csv, GOOG.csv, MSFT.csv   # High-resolution stock OHLCV datasets for sequential modeling
│   ├── TSLA.csv, Mobiles.csv                    # Financial time-series and tabular benchmark datasets
│   ├── the-verdict.txt, article.txt             # Literary and article text corpora for language pretraining
│   ├── gpt_sentences.txt, sentence_start.txt    # Synthetic prompt generation and completion datasets
│   ├── question_set.txt, llm_ans.txt            # Instruction-following question sets and target answers
│   └── llama_answers.csv, mistral_answers.csv   # Model inference output comparisons and evaluations
│
├── LLM Codes/                                   # Source code implementations & runnable notebooks
│   ├── Self/                                    # Independent experimental builds & end-to-end scripts
│   │   ├── 1. TinyLLM2_CharLevel.ipynb          # Character-level autoregressive text generator
│   │   ├── 2. MemoryLLM3_Stocks.ipynb           # LSTM-based sequential indicator network
│   │   ├── 3. CodeLLM.ipynb, 6. CodeLLM.ipynb   # Transformer decoder sequence models
│   │   ├── 4. TokenLLM.ipynb, 5. TokenLLM2.ipynb# Subword tokenization pipelines & Gutenberg processing
│   │   ├── Stocks.py                            # Production-grade stock data preprocessor & LSTM pipeline
│   │   └── tokenllm.py                          # Gutenberg dataset scraping & tokenization engine
│   ├── Vizuara/                                 # 15-part end-to-end GPT implementation series
│   │   ├── 01. Byte_Pair_Encoding.ipynb         # Custom BPE tokenization from scratch
│   │   ├── 02. Data_Loader_Input_Output_Pairs.ipynb # Sliding window sequence pair generation
│   │   ├── 03. Token_Embeddings.ipynb           # Token embedding lookup tables
│   │   ├── 04. Vector_Embedding.ipynb           # Vector geometry and embedding operations
│   │   ├── 05. Position_Embeddings.ipynb        # Learned 1D positional encodings
│   │   ├── 06. LLM_Data_Preprocessing.ipynb     # Full text-to-tensor preprocessing pipeline
│   │   ├── 07. Attention_Mechanism.ipynb        # Dot-product attention and similarity weighting
│   │   ├── 08. Trainable_Self_Attention.ipynb   # Query, Key, Value parameter matrices
│   │   ├── 09. Causal_Attention.ipynb           # Lower-triangular masked attention
│   │   ├── 10. Multi-Head Attention.ipynb       # Parallel multi-head attention module
│   │   ├── 11. LLM Architecture.ipynb           # Complete GPT-2 transformer block assembly
│   │   ├── 12. LLM Pretraining.ipynb            # Full pretraining loop with loss optimization
│   │   ├── 13. Model Weights Loaded.ipynb       # Loading official OpenAI GPT-2 weights
│   │   ├── 14. Architecture Classification finetuning.ipynb # Sequence classification head
│   │   ├── 15. Instruction fine-tuning dataset prep loading.ipynb # Instruction tuning pipeline
│   │   └── gpt_download3.py                     # Direct weights downloader for GPT-2 models
│   └── SQ/                                      # Specialized implementations
│       └── decoder_transformers_with_pytorch_and_lightning_v2.ipynb
│
├── LLM Notes/                                   # 54 comprehensive Markdown architectural write-ups
│   ├── GPT/                                     # 20 notes detailing GPT evolution & design patterns
│   └── Vizuara/                                 # 34 granular notes covering every mathematical building block
│
├── Theory/                                      # Landmark peer-reviewed NLP & LLM research papers
│   ├── Attention is All You Need.pdf            # Vaswani et al. (2017)
│   ├── Improving Language Understanding...pdf   # Radford et al. (GPT-1, 2018)
│   ├── Language Models are Unsupervised...pdf   # Radford et al. (GPT-2, 2019)
│   ├── Language Models are Few-Shot Learners.pdf# Brown et al. (GPT-3, 2020)
│   ├── LoRA.pdf                                 # Hu et al. (2021) Parameter-Efficient Fine-Tuning
│   ├── Efficient Memory Management...pdf        # Kwon et al. (2023) PagedAttention
│   ├── Gaussian Error Linear Units (GeLU).pdf   # Hendrycks & Gimpel (2016)
│   ├── Instruction Tuning With Loss Over...pdf  # Modern instruction alignment research
│   ├── Measuring Massive Multitask...pdf        # Hendrycks et al. (2020) MMLU Benchmark
│   ├── Neural Machine Translation By...pdf      # Bahdanau et al. (2014) Additive Attention
│   ├── Visualizing the Loss Landscape...pdf     # Li et al. (2018) Loss surfaces & skip connections
│   └── A New Algorithm for Data Compression.pdf # Classic BPE data compression foundations
│
└── Pages/                                       # Archived official PyTorch docs & visualizers for offline study
    ├── Datasets & DataLoaders, Module, Sequential, Softmax, Dropout, Embedding docs
    ├── torch.Tensor.masked_fill_, torch.tril, torch.triu, torch.utils.data docs
    ├── register_buffer vs register_parameter documentation
    └── Visualizing A Neural Machine Translation Model (Jay Alammar)
```

---

## 🗺️ Multi-Stage Pedagogical Roadmap

| Stage | Model Paradigm | Implemented Project | Core Mathematical & Architectural Concepts |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Character-Level RNN** | `TinyLLM` | Character vocabulary, one-hot vectors, hidden state recurrence $h_t = \tanh(W x_t + U h_{t-1} + b)$, Cross-Entropy loss |
| **Stage 2** | **Word-Level Embeddings** | `SmaLLM` | Word tokenization, continuous dense lookup tables, vocabulary frequency thresholding, OOV tokens |
| **Stage 3** | **Recurrent Memory (LSTM)** | `MemoryLLM` / `StocksLSTM` | Forget, input, and output gates ($f_t, i_t, o_t$), cell state memory $c_t$, long-range gradient retention |
| **Stage 4** | **Attention & Transformer Decoder** | `AttentionLLM` / `CodeLLM` | Scaled dot-product attention, causal mask ($\text{tril}$), multi-head projections, positional encoding |
| **Stage 5** | **Full GPT-2 Pretraining** | `Vizuara Series` (01–13) | Pre-LayerNorm, GeLU FFN, weight tying, sliding window data loaders, checkpoint loading |
| **Stage 6** | **Fine-Tuning & Alignment** | `Vizuara Series` (14–15) | Sequence classification heads, instruction tuning, prompt templates, LoRA low-rank adaptation |

---

## 📐 Mathematical & Architectural Deep Dive

```
                             ┌────────────────────────────────────────┐
                             │          Raw Text Sequence             │
                             └───────────────────┬────────────────────┘
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │     Byte Pair Encoding Tokenizer       │
                             └───────────────────┬────────────────────┘
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │ Token Embedding + Position Embedding   │
                             └───────────────────┬────────────────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │        Dropout        │
                                     └───────────┬───────────┘
                                                 ▼
                ┌────────────────────────► [Transformer Block] ◄─────────────────────────┐
                │                                │                                       │
                │                   ┌────────────┴───────────┐                           │
                │                   ▼                        │ (Residual Skip)           │
                │             LayerNorm 1                    │                           │
                │                   │                        │                           │
                │                   ▼                        │                           │
                │       Masked Multi-Head Attention          │                           │
                │                   │                        │                           │
                │                   ▼                        ▼                           │
                │               Dropout ──► (+) Add Connection                           │
                │                                │                                       │
                │                   ┌────────────┴───────────┐                           │
                │                   ▼                        │ (Residual Skip)           │
                │             LayerNorm 2                    │                           │
                │                   │                        │                           │
                │                   ▼                        │                           │
                │        Feed-Forward (Linear+GeLU)          │                           │
                │                   │                        │                           │
                │                   ▼                        ▼                           │
                │               Dropout ──► (+) Add Connection                           │
                │                                │                                       │
                └─────────────────────── (Repeat N Times) ───────────────────────────────┘
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │    Final LayerNorm    │
                                     └───────────┬───────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │  Linear Output Head   │
                                     └───────────┬───────────┘
                                                 ▼
                                     ┌───────────────────────┐
                                     │ Logits & Probabilities│
                                     └───────────────────────┘
```

### 1. Tokenization Paradigms & Vocabulary Engineering
- **Character-Level**: Minimal vocabulary ($V \approx 60\text{--}100$), zero out-of-vocabulary (OOV) tokens, but requires extremely long context windows to capture semantic concepts.
- **Word-Level**: Explicit word semantics, but vocabulary scales unsustainably ($V > 50{,}000$), causing heavy OOV failures on unseen words.
- **Byte Pair Encoding (BPE)**: A subword tokenization algorithm that iteratively merges the most frequent adjacent character or byte pairs in the corpus.
  - Generates subwords like `["trans", "former", "##s"]`.
  - Balances vocabulary compactness ($V = 50{,}257$ in GPT-2) with the ability to represent any arbitrary unicode string without OOV errors.

### 2. Dense Vector & Positional Embeddings
- **Token Embeddings ($W_e$)**: Given a sequence of token IDs $\mathbf{x} = [x_1, x_2, \dots, x_T]$, each index is mapped to a continuous vector:
  $$\mathbf{e}_t = W_e[x_t] \in \mathbb{R}^{d_{\text{model}}}$$
- **Positional Embeddings ($W_p$)**: Because self-attention is permutation-invariant, positional encodings must be injected:
  $$\mathbf{z}_t = \mathbf{e}_t + \mathbf{p}_t$$
- **Learned Positional Embeddings (GPT-2)**: Directly optimizes an embedding matrix $W_p \in \mathbb{R}^{T_{\text{max}} \times d_{\text{model}}}$.
- **Sinusoidal Positional Embeddings (Vaswani et al.)**:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### 3. Attention Mechanics: From Additive to Causal Multi-Head
- **Bahdanau Additive Attention**: Computes alignment score using a feed-forward layer: $e_{ij} = \mathbf{v}_a^T \tanh(W_a \mathbf{s}_{i-1} + U_a \mathbf{h}_j)$.
- **Scaled Dot-Product Attention**: Projects input vectors into Queries ($Q$), Keys ($K$), and Values ($V$):
  $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
- **Causal Attention Mask ($M$)**: An upper triangular matrix preventing tokens from attending to subsequent tokens:
  $$M_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$
- **Multi-Head Attention (MHA)**: Divides $d_{\text{model}}$ into $h$ heads of dimension $d_k = d_{\text{model}} / h$:
  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

### 4. Transformer Decoder Block Anatomy
- **Pre-Layer Normalization**: Normalizing inputs before each sub-layer stabilizes forward activations and backpropagated gradients:
  $$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
- **Feed-Forward Network (FFN)**: Two linear transformations with a non-linear activation expand the hidden dimensionality by $4\times$:
  $$\text{FFN}(\mathbf{x}) = \text{GeLU}(\mathbf{x}W_1 + \mathbf{b}_1)W_2 + \mathbf{b}_2$$
- **Gaussian Error Linear Unit (GeLU)**:
  $$\text{GeLU}(x) = x \cdot \Phi(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)$$
- **Residual Skip Connections**: $\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{SubLayer}(\text{LN}(\mathbf{x}^{(l)}))$, mitigating vanishing gradient problems across deep stacks.
- **GPT-2 Family Architecture Specifications**:

| Model Variant | Parameters | Layers ($L$) | Hidden Dimension ($d$) | Attention Heads ($h$) | Head Dimension ($d_k$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-2 Small** | 124M | 12 | 768 | 12 | 64 |
| **GPT-2 Medium** | 355M | 24 | 1024 | 16 | 64 |
| **GPT-2 Large** | 774M | 36 | 1280 | 20 | 64 |
| **GPT-2 XL** | 1558M | 48 | 1600 | 25 | 64 |

### 5. Training Dynamics, Loss Formulations & Optimization
- **Causal Next-Token Prediction**: For an input sequence of length $T$, the targets are shifted by 1 position:
  $$\text{Input: } [x_1, x_2, \dots, x_{T-1}] \longrightarrow \text{Target: } [x_2, x_3, \dots, x_T]$$
- **Cross-Entropy Loss**:
  $$\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P(x_{t+1} \mid x_1, \dots, x_t) = -\frac{1}{T-1} \sum_{t=1}^{T-1} \left( \hat{z}_{t, x_{t+1}} - \log \sum_{v=1}^V \exp(\hat{z}_{t, v}) \right)$$
- **Perplexity Metric**: $\text{PPL} = \exp(\mathcal{L})$, representing the effective branching factor during next-token selection.
- **AdamW Optimization**: Decoupled weight decay regularization to preserve generalization without corrupting adaptive gradient moment estimates.

### 6. Inference Decoding & Sampling Strategies
- **Greedy Decoding**: $x_{t+1} = \arg\max_i (z_i)$ — deterministic, prone to repetitive loops.
- **Temperature Scaling ($T$)**: Modulates logit sharpness before Softmax:
  $$P(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
- **Top-$K$ Sampling**: Truncates candidate logits to the top $K$ most probable tokens before renormalizing.
- **Top-$P$ (Nucleus) Sampling**: Selects the smallest token subset whose cumulative probability exceeds threshold $P$.

---

## 💻 Hands-On Implementations & Codebase

### Self Practice Laboratory (`LLM Codes/Self/`)
- [`1. TinyLLM2_CharLevel.ipynb`](./LLM%20Codes/Self/1.%20TinyLLM2_CharLevel.ipynb): Character-level neural language model with custom sequence generation.
- [`2. MemoryLLM3_Stocks.ipynb`](./LLM%20Codes/Self/2.%20MemoryLLM3_Stocks.ipynb): Deep LSTM memory network predicting price trends with technical indicators.
- [`3. CodeLLM.ipynb`](./LLM%20Codes/Self/3.%20CodeLLM.ipynb) & [`6. CodeLLM.ipynb`](./LLM%20Codes/Self/6.%20CodeLLM.ipynb): Transformer decoder experiments with sequence-to-sequence targets.
- [`4. TokenLLM.ipynb`](./LLM%20Codes/Self/4.%20TokenLLM.ipynb) & [`5. TokenLLM2.ipynb`](./LLM%20Codes/Self/5.%20TokenLLM2.ipynb): Automated tokenizer token-ID encoding and decoding flows.
- [`Stocks.py`](./LLM%20Codes/Self/Stocks.py): Full standalone stock data preprocessor, feature pipeline (RSI, MACD, Bollinger Bands), and LSTM model.
- [`tokenllm.py`](./LLM%20Codes/Self/tokenllm.py): Project Gutenberg novel downloader, sentence boundary segmenter, and Hugging Face tokenizer bridge.

### From-Scratch GPT Series (`LLM Codes/Vizuara/`)
1. `01. Byte_Pair_Encoding.ipynb`: Subword vocabulary extraction and merge rule derivation.
2. `02. Data_Loader_Input_Output_Pairs.ipynb`: PyTorch `Dataset` and `DataLoader` sliding context windows.
3. `03. Token_Embeddings.ipynb`: Matrix mapping from token IDs to dense vectors.
4. `04. Vector_Embedding.ipynb`: Vector space properties, cosine similarities, and geometric intuition.
5. `05. Position_Embeddings.ipynb`: Implementation of 1D absolute learned position encodings.
6. `06. LLM_Data_Preprocessing.ipynb`: Complete preprocessing pipeline from raw text to tensor batches.
7. `07. Attention_Mechanism.ipynb`: Raw dot product similarity and attention weight distributions.
8. `08. Trainable_Self_Attention.ipynb`: Query ($W_Q$), Key ($W_K$), and Value ($W_V$) parameter matrices.
9. `09. Causal_Attention.ipynb`: Upper triangular masking using `torch.tril` and `masked_fill_`.
10. `10. Multi-Head Attention.ipynb`: Efficient batched multi-head splitting, computation, and recombination.
11. `11. LLM Architecture.ipynb`: Complete GPT-2 block assembly with Pre-LN, GeLU FFN, and residuals.
12. `12. LLM Pretraining.ipynb`: Full pretraining loop with loss tracking, target alignment, and generation check.
13. `13. Model Weights Loaded.ipynb`: Transferring official OpenAI GPT-2 checkpoints into custom PyTorch modules.
14. `14. Architecture Classification finetuning.ipynb`: Replacing the output head for sentiment and text classification.
15. `15. Instruction fine-tuning dataset prep loading.ipynb`: Prompt-response formatting and instruction alignment.
16. `gpt_download3.py`: Direct weights downloader for GPT-2 models (`124M`, `355M`, `774M`, `1558M`).

### PyTorch Lightning Decoder Suite (`LLM Codes/SQ/`)
- [`decoder_transformers_with_pytorch_and_lightning_v2.ipynb`](./LLM%20Codes/SQ/decoder_transformers_with_pytorch_and_lightning_v2.ipynb): Modular decoder-only transformer built using PyTorch Lightning with distributed training support.

### Root Sequence Modeling Notebooks
- [`CodeLLM.ipynb`](./CodeLLM.ipynb): Comprehensive Transformer modeling notebook for multi-variate sequence forecasting.
- [`StocksLSTM.ipynb`](./StocksLSTM.ipynb): In-depth time-series experimentation comparing Recurrent architectures with modern attention.

---

## 📚 Theoretical Knowledge Base (`LLM Notes/`)

### GPT Architecture & Engineering Notes (`LLM Notes/GPT/`)
A structured collection of 20 detailed study guides covering architectural decisions:
- **`LLM01 TinyLLM.md` - `LLM04 SmaLLM.md`**: Transitioning from character models to vocabulary-backed word models.
- **`LLM05 MemoryLLM.md` & `LLM10 Stage3 vs Stage4.md`**: In-depth trade-off analysis comparing LSTMs and Transformers.
- **`LLM06 CodeLLM.md` - `LLM09 Training.md`**: Pretrained model economics, GPU memory profiles, and training loops.
- **`LLM13 Pos Embed.md` - `LLM16 Handle OOV.md`**: Input embedding strategies, vector math, and handling unknown tokens.
- **`LLM17 Causal Attention Code.md` - `LLM20 GPT2-Small Block.md`**: Masked attention mechanics, LayerNorm placement, and the complete 12-layer GPT-2 topology.

### Foundational Deep Dives (`LLM Notes/Vizuara/`)
34 modular markdown files dissecting every sub-component:
- **Concepts**: GenAI fundamentals, Pre-training vs Fine-tuning, Encoder-Decoder vs Decoder-only, BERT vs GPT.
- **Data & Embeddings**: BPE tokenization algorithms, sliding window dataloaders, vector spaces, and positional encodings.
- **Attention Evolution**: RNN bottlenecks, Bahdanau additive attention, scaled dot-product attention, causal masks, and multi-head parallelization.
- **Neural Sub-layers**: GELU activation derivations, feed-forward dimension expansion, residual additions, and Pre-LN stability.
- **Training & Decoding**: Cross-entropy calculation over shifts, perplexity metrics, temperature scaling, top-$K$, and nucleus top-$P$ sampling.

---

## 📖 Landmark Research Papers Library (`Theory/`)

The `Theory/` directory houses the core research literature that established modern natural language processing:

| Paper Title | Key Contribution to LLM Architecture |
| :--- | :--- |
| **Attention Is All You Need** *(Vaswani et al., 2017)* | Introduced the Transformer; replaced recurrence with multi-head self-attention. |
| **Improving Language Understanding (GPT-1)** *(Radford et al., 2018)* | Established generative pre-training + discriminative fine-tuning paradigm. |
| **Language Models are Unsupervised Multitask Learners (GPT-2)** *(Radford et al., 2019)* | Demonstrated zero-shot task transfer across diverse text distributions. |
| **Language Models are Few-Shot Learners (GPT-3)** *(Brown et al., 2020)* | Scaled autoregressive models to 175B parameters; in-context few-shot learning. |
| **LoRA: Low-Rank Adaptation of Large Language Models** *(Hu et al., 2021)* | Freezes base weights and injects trainable low-rank rank-decomposition matrices. |
| **Efficient Memory Management (PagedAttention)** *(Kwon et al., 2023)* | Non-contiguous KV-cache memory allocation for high-throughput LLM serving. |
| **Gaussian Error Linear Units (GELUs)** *(Hendrycks & Gimpel, 2016)* | Probabilistic continuous non-linearity adopted by GPT, BERT, and modern transformers. |
| **Neural Machine Translation by Jointly Learning to Align and Translate** *(Bahdanau et al., 2014)* | First introduction of soft attention mechanisms in seq2seq translation models. |
| **Measuring Massive Multitask Language Understanding (MMLU)** *(Hendrycks et al., 2020)* | Benchmark standard for evaluating world knowledge and multi-task reasoning. |
| **Visualizing the Loss Landscape of Neural Nets** *(Li et al., 2018)* | Filter normalization techniques explaining why skip connections smooth loss surfaces. |
| **Instruction Tuning With Loss Over Instructions** *(2023)* | Masking instruction prompt tokens during loss computation to improve response alignment. |
| **A New Algorithm for Data Compression** *(1994)* | Byte-pair encoding compression algorithms foundational to subword tokenizers. |

---

## 📊 Datasets & Benchmark Corpus Catalog (`Datasets/`)

- **Financial Time-Series (`AAPL.csv`, `AMZN.csv`, `GOOG.csv`, `MSFT.csv`, `TSLA.csv`)**: Historical price and volume data used to benchmark sequence memory in LSTMs vs causal attention models.
- **Pretraining Corpora (`the-verdict.txt`, `article.txt`)**: Clean text corpuses used for vocabulary construction, BPE training, and next-token prediction loops.
- **Synthetic Prompt Datasets (`gpt_sentences.txt`, `mistral_phi2_sentences.txt`, `sentence_start.txt`)**: Evaluation prompts for sampling analysis.
- **Model Output Evaluations (`llama_answers.csv`, `mistral_answers.csv`, `llm_answers.csv`)**: Model response benchmarks for comparative inference studies.
- **Tabular Datasets (`student_depression_dataset.csv`, `internet_usage.csv`, `Mobiles.csv`)**: Multi-modal tabular benchmarks for classification fine-tuning.

---

## 📑 Offline Developer Documentation (`Pages/`)

Local HTML snapshots preserving critical PyTorch API references and pedagogical guides for offline research:
- **PyTorch Core APIs**: `nn.Module`, `nn.Sequential`, `nn.Embedding`, `nn.Dropout`, `nn.Softmax`, `torch.utils.data.DataLoader`.
- **Tensor Operations**: `torch.tril` (lower triangular mask), `torch.triu`, and `torch.Tensor.masked_fill_`.
- **Advanced PyTorch Design**: `register_buffer` vs `register_parameter` lifecycle and state dict persistence.
- **Visual Reference**: Jay Alammar's *Visualizing A Neural Machine Translation Model (Seq2Seq with Attention)*.

---

## ⚡ Installation, Environment Setup & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/AvrodeepPal/LLMsPractice.git
cd LLMsPractice
```

### 2. Configure Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # Or CPU wheel
pip install tensorflow transformers datasets tiktoken pytorch-lightning pandas numpy matplotlib seaborn tqdm scikit-learn
```

### 4. GPU & Hardware Acceleration
- **Apple Silicon (M1/M2/M3/M4)**: Automatically utilizes PyTorch `mps` backend (`torch.device("mps")`).
- **NVIDIA GPUs**: Configured for CUDA execution with mixed-precision training (`torch.cuda.amp.autocast`).

---

## 💡 Key Takeaways & Architectural Lessons

1. **Attention Eliminates Sequential Bottlenecks**: Unlike LSTMs whose memory degrades with sequence length, self-attention maintains direct $O(1)$ computational paths between all tokens.
2. **Causal Masking is Essential for Generative Decoders**: Setting future attention logits to $-\infty$ ensures the model strictly learns $P(w_t \mid w_1, \dots, w_{t-1})$.
3. **Pre-LN Stabilizes Deep Training**: Placing Layer Normalization before the multi-head attention and feed-forward sub-layers enables training deep networks without gradient explosion.
4. **Tokenization Directly Dictates Model Capability**: Subword tokenization via BPE balances vocabulary size and context window efficiency while avoiding out-of-vocabulary failures.
5. **Decoupled Weight Decay Matters**: Using AdamW prevents regularized weights from distorting adaptive learning rate moments, significantly improving generalization.

---

## 👤 Author & Acknowledgments

**Avrodeep Pal**
- Passionate about Deep Learning, Natural Language Processing, and Generative AI Architectures.
- Focused on building production-grade machine learning pipelines and demystifying LLMs from first principles.
- GitHub: [@AvrodeepPal](https://github.com/AvrodeepPal)
- Collaboration: Open issues, discussions, or pull requests are warmly welcomed!

⭐ *If you find this repository helpful in your LLM learning journey, feel free to star the repo and share it!*


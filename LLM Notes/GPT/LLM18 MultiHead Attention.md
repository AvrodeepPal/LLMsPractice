This image outlines the **step-by-step flow of multi-head attention in Transformers**, particularly from the perspective of PyTorch tensor operations. Let’s break each step down in **clear detail**, using a typical Transformer setup:

---

### 🔷 **Step 1: Reduce the projection dim to match desired output dim**
- **Purpose**: Project the input embedding `x` (of shape `[batch_size, seq_len, d_model]`) into a smaller dimensional space suitable for attention.
- This is done via learned linear layers: `W_q`, `W_k`, `W_v` for **queries**, **keys**, and **values**.
- These projections usually reduce `d_model` → `num_heads × head_dim`.

---

### 🔷 **Step 2: Use a Linear layer to combine head outputs**
- After computing the outputs from all heads, we concatenate them and pass them through another linear layer (like `W_o`) to combine them into a single output tensor.
- This lets the model learn how to combine the info from different heads.

---

### 🔷 **Step 3: Tensor shape: `(b, num_tokens, d_out)`**
- This is the **original input** after linear projection:
  `x ∈ [batch_size, seq_len, d_out]`
- It’s still one big vector per token; no splitting into heads yet.

---

### 🔷 **Step 4: Split into multiple heads**
- You reshape the tensor to separate the heads:
  From → `(b, seq_len, d_out)`
  To → `(b, seq_len, num_heads, head_dim)`
  where `head_dim = d_out // num_heads`
- This is an important step so that each head can perform attention on a different subspace.

---

### 🔷 **Step 5: Transpose to `(b, num_heads, seq_len, head_dim)`**
- Why? Because we want to parallelize attention computation **per head**.
- Each head will now attend over the sequence independently.

---

### 🔷 **Step 6: Compute dot product for each head**
- Compute attention scores using:
  \[
  \text{Attention} = \frac{QK^T}{\sqrt{d_k}}
  \]
- This gives raw scores for each token attending to every other token in the sequence, per head.

---

### 🔷 **Step 7: Mask truncated to number of tokens**
- **Masking** is crucial for causal attention (like in GPT). It ensures:
  - Each token can **only attend to itself and past tokens**, not future ones.
- The mask is usually a triangular matrix (`b, num_heads, seq_len, seq_len`).

---

### 🔷 **Step 8: Apply the mask to attention scores**
- The attention mask is **added** (with `-inf` for masked positions), so that after softmax, those positions become zero.

---

### 🔷 **Step 9: Tensor shape: `(b, num_tokens, n_heads, head_dim)`**
- After applying attention weights to the values `V`, we have the result per head.
- It’s often still in shape `(b, num_heads, seq_len, head_dim)` before transposing back to merge.

---

### 🔷 **Step 10: Combine heads**
- Transpose and reshape back:
  `(b, seq_len, num_heads × head_dim)` → basically recombine all the heads back into a single representation per token.
- This forms the final output of the multi-head attention block before any final projection.

---

### 🔷 **Step 11: Add an optional linear projection**
- This final `Linear` layer (like `W_o`) maps the combined attention outputs back to the model’s dimensionality (`d_model`).
- It helps the model blend and reinterpret what was learned by the different heads.

---

### ✅ Summary Flow:
```text
Input (b, seq_len, d_model)
 ↓
Linear Q, K, V → (b, seq_len, num_heads, head_dim)
 ↓
Transpose to (b, num_heads, seq_len, head_dim)
 ↓
Attention: Q @ K^T → Apply mask → Softmax → Weigh V
 ↓
Output from each head → transpose and reshape
 ↓
Concat heads → Linear projection
 ↓
Final output: (b, seq_len, d_model)
```

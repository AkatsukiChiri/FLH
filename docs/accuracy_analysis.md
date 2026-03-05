# FLH 量化模型精度略差原因分析

## 1. 可能原因概览

| 类别 | 可能原因 | 影响 |
|------|----------|------|
| 数据流 | LinearFLH 输入未做 H 变换 | 高 |
| 量化误差 | 8bit 权重量化 + 8bit 激活量化 | 中 |
| 分组 | group_size=128 分组量化/分组 H | 中 |
| 参考不一致 | embedding 后 H 仅量化模型有 | 低 |

---

## 2. 数据流：LinearFLH 约定与当前实现

**约定**：单侧 Hadamard 的 LinearFLH 存的是 `W_single = W @ H`，且 **forward 只做 `y = x @ W_single^T`**，不自己做 H。因此要得到“原始线性层”结果 `y_orig = x_orig @ W^T`，必须 **在外部先把输入变到 H 空间**，即 `x = x_orig @ H`，这样：
- `y = x @ (W@H)^T = x_orig @ H @ H^T @ W^T = x_orig @ W^T`（用到 H 正交性）。

**当前实现检查**：

- **Attention**  
  - 第一段：`hidden_states = quantizer1(hidden_states)`（仅量化，无 H）→ 直接 `q_proj(hidden_states)` / `k_proj` / `v_proj`。  
  - 若此处不先对 `hidden_states` 做分组 H，则送入 QKV 的是“未 H 空间”，与上述约定不符，**会系统性地错一层线性**，通常会导致明显精度下降而不是“略差”。若你观察到的是“略差”，需要再确认是否在别处（例如上层 down_proj 输出已在 H 空间）已经等价地补过一次 H。

- **MLP**  
  - `x = quantizer1(x)` → `gate_proj(x)`、`up_proj(x)`。  
  - gate_proj/up_proj 也是单侧 Hadamard 的 LinearFLH，同样约定输入应为 H 空间。若这里没有在 `gate_proj/up_proj` 前对 `x` 做分组 H，则与约定不一致，也会带来误差。

**结论**：若设计上要求“所有单侧 Hadamard 的 Linear 的输入都应是 H 空间”，则当前在 **QKV 前** 以及 **gate_proj/up_proj 前** 若缺少一次分组 Hadamard，是精度变差的重要候选原因。需要在不违反“不补充多余 hadamard 层”的前提下，确认这两处是否应各保留一次 H（不是“多余”，而是数据流必需）。

---

## 3. 量化与分组带来的误差

- **权重量化**：W8 本身会引入舍入误差；`weight_group_size=128` 时按组求 scale，组内动态范围大时误差会比 per-tensor 略大。
- **激活量化**：A8 + `act_group_size=128`，分组统计 min/max 再量化，同样会略增误差。
- **Hadamard 分组**：`fast_hadamard_transform(..., group_size=act_group_size)` 与量化分组一致（例如 128）时，设计上是自洽的；若某处误用 `group_size=-1` 或与权重/激活分组不一致，可能在某几层放大误差。

建议做小实验：暂时把 `weight_group_size` / `act_group_size` 改为 -1（或 256），看精度是否明显变好，以判断分组是否为主要因素。

---

## 4. Embedding 后 Hadamard 仅量化模型有

当前仅在 **FLH_LlamaForCausalLM.forward** 里在 embedding 之后做了分组 H：

```python
inputs_embeds = self.model.embed_tokens(input_ids)
inputs_embeds = flh.nn.fast_hadamard_transform(inputs_embeds, group_size=self.act_group_size, normalize=True)
```

若对比的是：
- **量化模型 vs 原始 Llama**：原始没有这步 H，第一层看到的分布不同，会有一致性偏差。
- **量化模型 vs FP16 FLH**：若 FP16 也没有在 embedding 后做 H，同样存在系统偏差。

若希望“公平对比”，要么两边都在 embedding 后做同样的 H，要么两边都不做。

---

## 5. down_proj 双侧 Hadamard 与下一层衔接

down_proj 设为双侧 Hadamard 后：
- 输入：应为 H 空间（由 quantizer2 输出提供，正确）。
- 输出：`(x @ W^T) @ H^T`，即 **在 H 空间**。

下一层是：`input_layernorm` → `quantizer1` → 然后进 QKV。若进 QKV 前设计上需要“先转到 H 空间”，而当前代码在 **QKV 前没有**对 `hidden_states` 做 H，那么上一层的 H 空间输出到这一层就一直没有被“用对”，会累积误差。这也与第 2 点一致：**QKV 前是否缺一次 H** 是关键检查点。

---

## 6. 建议的检查与修改（不增加“多余”层）

1. **确认数据流**  
   - 在 **Attention**：在 `q_proj`/`k_proj`/`v_proj` 前，对 `hidden_states`（即 quantizer1 输出）做一次 `fast_hadamard_transform(..., group_size=act_group_size, normalize=True)`，保证送入的是 H 空间。  
   - 在 **MLP**：在 `gate_proj`/`up_proj` 前，对 `x`（即 quantizer1 输出）做一次同样的分组 H。  
   这两处若设计上本就应该有，则不属于“多余”层，而是与 LinearFLH 约定一致的必要步骤。

2. **对比实验**  
   - `weight_group_size=128` vs `-1`（或 256）；`act_group_size=128` vs `-1`，看精度变化。  
   - 若有可能，对比“量化 vs 同结构的 FP16 FLH”（FP16 也做同样的 embedding 后 H、输出前不对 logits 做 H），以排除“参考模型不一致”的影响。

3. **输出端**  
   当前“所有层之后、输出之前”的 H 已注释掉（未对 `outputs.logits` 做 H），这是合理的，因为 logits 需要在词表空间做 argmax，不应再在最后一维做 H。

按上述顺序排查，通常能定位到“略差”的主要来源；若补上 QKV 前和 gate/up 前的 H 后精度明显恢复，即可确认是数据流与 LinearFLH 约定不一致导致的。

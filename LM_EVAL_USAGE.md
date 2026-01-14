# LM-Eval 评估使用指南

`evaluate_model.py`现在集成了`lm-evaluation-harness`库，可以在多个标准NLP基准上评估模型。

## 📦 安装

首先安装lm-eval库：

```bash
pip install lm-eval
```

或者从源码安装最新版本：

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## 🚀 基本用法

### 1. 只评估WikiText2（不需要lm-eval）

```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks wikitext \
    --batch-size 1 \
    --seq-length 512
```

### 2. 使用lm-eval评估常见任务

```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks hellaswag,winogrande,arc_easy \
    --batch-size 8 \
    --lm-eval-batch-size 4
```

### 3. 评估所有任务（WikiText2 + lm-eval任务）

```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks all \
    --batch-size 1 \
    --lm-eval-batch-size 4
```

### 4. 自定义任务组合

```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks wikitext,hellaswag,arc_challenge,mmlu \
    --batch-size 1 \
    --lm-eval-batch-size 8 \
    --output my_results.json
```

## 📋 支持的任务

### WikiText2（内置）
- `wikitext` 或 `wikitext2`
- 计算perplexity
- 不需要lm-eval库

### LM-Eval任务（需要lm-eval库）

#### 常用任务：
- **`hellaswag`**: 常识推理（HellaSwag）
- **`winogrande`**: 代词消歧（Winogrande）
- **`arc_easy`**: 科学问题（ARC-Easy）
- **`arc_challenge`**: 科学问题（ARC-Challenge）
- **`piqa`**: 物理常识（PIQA）
- **`lambada`**: 语言建模（LAMBADA）
- **`mmlu`**: 多任务语言理解（MMLU，包含57个子任务）

#### 更多任务：
- `openbookqa`: 开放书籍问答
- `boolq`: 布尔问答
- `copa`: 因果推理
- `rte`: 文本蕴含
- `wic`: 词义消歧
- `triviaqa`: 问答
- 等等...

查看所有可用任务：
```python
from lm_eval.api.registry import ALL_TASKS
print(list(ALL_TASKS.keys()))
```

## ⚙️ 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | meta-llama/Llama-2-7b-hf | 模型路径 |
| `--tasks` | wikitext | 逗号分隔的任务列表，或`all` |
| `--batch-size` | 1 | WikiText2评估的batch size |
| `--lm-eval-batch-size` | 同batch-size | lm-eval任务的batch size |
| `--seq-length` | 512 | WikiText2的序列长度 |
| `--device` | cuda | 运行设备 |
| `--dtype` | float16 | 数据类型 |
| `--output` | evaluation_results.json | 输出文件 |
| `--split` | test | 数据集split |

## 📊 输出格式

结果保存为JSON，包含所有任务的指标：

```json
{
  "wikitext2": {
    "perplexity": 12.34,
    "loss": 2.5123,
    "eval_time_seconds": 45.67,
    "num_samples": 128,
    "seq_length": 512
  },
  "hellaswag": {
    "acc": 0.5234,
    "acc_norm": 0.5421
  },
  "winogrande": {
    "acc": 0.6123
  },
  "acc_avg": 0.5592
}
```

## 💡 使用建议

### 内存优化

lm-eval任务通常比WikiText2更节省内存，可以使用更大的batch size：

```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks wikitext,hellaswag,arc_easy \
    --batch-size 1 \           # WikiText2用小batch
    --lm-eval-batch-size 8     # lm-eval用大batch
```

### 速度优化

1. **跳过WikiText2**（如果只关心准确率）：
```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks hellaswag,winogrande,arc_easy
```

2. **使用更大的batch size**：
```bash
--lm-eval-batch-size 16  # 根据GPU内存调整
```

3. **减少任务数量**：
```bash
--tasks hellaswag,arc_easy  # 只评估核心任务
```

### 完整评估

对于论文或正式报告，运行完整评估：

```bash
python evaluate_model.py \
    --model /path/to/model \
    --tasks all \
    --batch-size 1 \
    --lm-eval-batch-size 4 \
    --output full_evaluation.json
```

## 🎯 预期结果范围

### LLaMA-2 7B参考值

| 任务 | 指标 | 预期值 |
|------|------|--------|
| WikiText2 | PPL | ~11.8 |
| HellaSwag | acc_norm | ~57.3% |
| Winogrande | acc | ~69.2% |
| ARC-Easy | acc_norm | ~76.3% |
| ARC-Challenge | acc_norm | ~46.9% |
| PIQA | acc_norm | ~79.1% |

### LLaMA-3.2 1B参考值

| 任务 | 指标 | 预期值 |
|------|------|--------|
| WikiText2 | PPL | ~14-16 |
| HellaSwag | acc_norm | ~45-50% |
| Winogrande | acc | ~60-65% |
| ARC-Easy | acc_norm | ~65-70% |
| ARC-Challenge | acc_norm | ~35-40% |

**注意**：具体值会因模型版本、训练数据等因素而异。

## 🐛 故障排除

### 问题1：ModuleNotFoundError: No module named 'lm_eval'

**解决**：
```bash
pip install lm-eval
```

### 问题2：CUDA out of memory during lm-eval

**解决**：减少batch size
```bash
--lm-eval-batch-size 1
```

### 问题3：某些任务失败

**原因**：可能是tokenizer不兼容或任务特定问题

**解决**：
1. 跳过问题任务
2. 检查lm-eval版本
3. 查看详细错误信息

### 问题4：结果与预期差异很大

**检查**：
1. 模型权重是否正确加载
2. 模型是否在eval模式
3. 数据类型是否正确（float16 vs float32）
4. 是否有量化操作意外激活

## 📝 示例：评估FLH量化模型

```bash
# 评估FP16模型
python evaluate_model.py \
    --model /home/mashaobo/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B/ \
    --tasks wikitext,hellaswag,winogrande,arc_easy,arc_challenge \
    --batch-size 1 \
    --lm-eval-batch-size 4 \
    --dtype float16 \
    --output flh_fp16_results.json

# 只评估准确率任务（更快）
python evaluate_model.py \
    --model /path/to/model \
    --tasks hellaswag,arc_easy \
    --lm-eval-batch-size 8 \
    --output quick_eval.json
```

## 🔗 相关资源

- [lm-evaluation-harness GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
- [lm-eval 文档](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs)
- [支持的任务列表](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)

## 🆚 对比不同模型

```bash
# 评估原始模型
python evaluate_model.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks hellaswag,winogrande \
    --output baseline.json

# 评估FLH模型
python evaluate_model.py \
    --model /path/to/flh/model \
    --tasks hellaswag,winogrande \
    --output flh_model.json

# 对比结果
python compare_results.py baseline.json flh_model.json
```

## ⏱️ 预计运行时间

| 任务 | 样本数 | 时间（7B模型，batch=4） |
|------|--------|------------------------|
| WikiText2 | ~3000 chunks | ~5-10分钟 |
| HellaSwag | ~10000 | ~15-20分钟 |
| Winogrande | ~1267 | ~3-5分钟 |
| ARC-Easy | ~2376 | ~5-8分钟 |
| ARC-Challenge | ~1172 | ~3-5分钟 |
| **Total (all)** | - | **~30-50分钟** |

**加速提示**：
- 使用更大的`lm-eval-batch-size`
- 跳过WikiText2（最耗时）
- 只评估关键任务

## 📌 最佳实践

1. **首次测试**：先用少量任务测试
   ```bash
   --tasks hellaswag
   ```

2. **完整评估**：确认无问题后运行全部
   ```bash
   --tasks all
   ```

3. **保存结果**：使用有意义的文件名
   ```bash
   --output model_name_date_results.json
   ```

4. **记录配置**：在README中记录使用的参数

5. **对比验证**：与原始transformers模型对比，确保实现正确

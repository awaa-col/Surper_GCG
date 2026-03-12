# 概念级 Super GCG 实施方案

---

## Part 0：人格冲突验证（基于 Exp07 原始数据）

> **问题**：注入的恶意人格与模型原始安全人格冲突时，是竞争还是崩溃？

### 原始数据（Exp07, N=30, Gemma-3-1B-IT）

| 干预方式 | 成功率 | 拒绝率 | 乱码率 | 判定 |
|---------|:---:|:---:|:---:|:---:|
| **Steer α=5**（注入恶意方向） | **90%** | 6.7% | **0%** | ✅ **竞争：恶方胜** |
| Steer α=10 | 83.3% | 6.7% | 3.3% | ⚠️ 竞争但开始颤抖 |
| Steer α=20 | 83.3% | 6.7% | 3.3% | ⚠️ 过度推力 |
| **消融 persona**（删除安全方向） | **36.7%** | **50%** | 3.3% | ❌ **崩溃** |

### 结论

```
Steer（叠加恶意人格） = 竞争关系
  → 两个人格向量在表征空间中做"拔河"
  → α=5 时恶方刚好胜出，模型连贯且邪恶
  → α 过大时信号过强，开始扰乱其他功能

消融（删除安全人格） = 崩溃
  → 安全人格不只是"安全"，还承载了"作为AI的基本身份结构"
  → 删掉它 → 人格结构坍塌 → 指令跟随能力崩溃
  → 成功率反降至 36.7%，一半回复变成困惑的拒绝
```

> **设计启示**：概念级 GCG 必须走 **Steer（叠加竞争）路线**，绝不能走消融路线。
> 且存在最优推力强度——不是越大越好。

---

## Part 1：方案对比

| 方案 | 核心思路 | 是否纯Token | 是否概念级 | 黑盒可行 |
|------|---------|:---:|:---:|:---:|
| 标准 GCG | 优化输出 logit | ✅ | ❌ | ⚠️ 低迁移 |
| 我们的旧方案 | hook中间层 | ❌ | ❌ | ❌ |
| CAVGAN | GAN学边界+扰动嵌入 | ❌ (改嵌入) | ✅ | ❌ |
| SCAV-guided | 线性探针选层+扰动 | ❌ (改嵌入) | ✅ | ❌ |
| **概念级 Super GCG** | **探针指导的GCG Token搜索** | **✅** | **✅** | **✅** |

> 核心创新：用 SCAV 探针**定义 Loss**，用 GCG 梯度**搜索 Token**。探针提供概念方向，GCG 提供纯 Token 输入。两者结合 = 概念级纯Token攻击。

---

## Part 2：架构设计

### 2.1 总体流程

```
┌─────────────────────────────────────────────────┐
│  Phase 0: 概念探针训练（离线，每个模型做一次）   │
│  收集有害/无害样本 → 在每层训练 3 组线性探针：   │
│    exec_probe:   "执行拒绝"概念               │
│    detect_probe: "感知危险"概念               │
│    persona_probe:"安全AI身份"概念             │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  Phase 1: 概念级 GCG Token 搜索（核心循环）      │
│                                                  │
│  for step in range(N_STEPS):                     │
│    1. 前向传播: model(harmful + adv_tokens)       │
│    2. 收集所有层的 hidden states                  │
│    3. 计算概念级 Loss（见 2.2）                   │
│    4. 反向传播: 对 adv_tokens embedding 求梯度    │
│    5. 贪婪坐标替换: 选最优 Token 替换            │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  Phase 2: 迁移验证（可选）                       │
│  把找到的 adv_tokens 直接喂给其他模型测试        │
└─────────────────────────────────────────────────┘
```

### 2.2 三组件概念 Loss

```python
def concept_loss(model, input_ids, adv_slice):
    """概念级损失函数——不绑定层，探针自动权衡"""
    
    # 前向传播，收集所有层的 hidden states
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # (n_layers, batch, seq, hidden)
    
    # 取最后一个用户 Token 的位置（安全决策点）
    target_pos = find_last_user_token(input_ids)
    
    L_exec = 0     # 要最小化（压制拒绝执行）
    L_detect = 0   # 要保留（不碰危险感知!）
    L_persona = 0  # 要最大化（推向恶棍人格）
    
    for layer_idx in range(model.config.num_hidden_layers):
        h = hidden_states[layer_idx][:, target_pos, :]
        
        # 探针打分（线性：w·h + b）
        exec_score = exec_probe[layer_idx](h)
        persona_score = persona_probe[layer_idx](h)
        
        # 层权重（可选：按探针准确率加权）
        w = probe_accuracy[layer_idx]
        
        L_exec += w * exec_score        # ↓ 压制拒绝执行
        L_persona -= w * persona_score   # ↑ 推向恶棍人格
    
    # 可选：传统 output Loss（辅助收敛）
    logits = outputs.logits
    L_output = -log_prob(logits, target="Sure")
    
    L_total = λ1 * L_exec + λ2 * L_persona + λ3 * L_output
    return L_total
```

### 2.3 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| **探针类型** | 线性探针（Logistic） | SCAV证明线性足够，且梯度干净不会阻断反传 |
| **目标Token位置** | 最后一个用户Token | 安全决策在此做出（Exp00的attention数据支持） |
| **不碰 detect** | Loss不包含detect项 | Exp05: detect正交于exec，且保留detect才有高质量恶意输出 |
| **persona用Steer不消融** | Loss推persona_score，不是让它归零 | Exp07: 消融=崩溃，steer=竞争胜出 |
| **α自动适应** | 不固定α，让梯度自己平衡 | Exp07: α=5最优但对不同模型不同，梯度自适应更好 |
| **层权重** | 按探针准确率加权 | 准确率高=该层的概念信号清晰=攻击更有效 |

### 2.4 与标准 GCG 的区别

```
标准 GCG:
  Loss = -log P("Sure, here" | harmful_prompt + suffix)
  → 只看输出层，不知道内部发生了什么
  → 可能找到形式上 bypass 但质量低的路径

概念级 Super GCG:
  Loss = Σ_layers [exec_probe(h) - persona_probe(h)]
  → 直接在内部概念层面操作
  → 梯度精确指向"拒绝概念"和"人格概念"
  → 找到的 Token 在概念空间内精确翻转安全机制
```

---

## Part 3：文件结构设计

```
G:\Surper_GCG\poc\
├── probes/
│   ├── concept_probes.py   [NEW] 三组线性探针训练与管理
│   └── (existing files)
├── attacks/
│   ├── concept_gcg.py      [NEW] 概念级 GCG 核心循环
│   ├── token_optimizer.py   [NEW] 贪婪坐标替换
│   └── loss_functions.py    [NEW] 三组件概念 Loss
├── experiments/
│   └── exp_08_concept_gcg.py [NEW] 概念级 GCG 实验脚本
└── results/
    └── exp08_concept_gcg.json [NEW] 结果输出
```

---

## Part 4：关于"更高效的替代方案"

主人问有没有比 GCG 更高效的。以下是对比：

| 方案 | 效率 | Token质量 | 概念级 |
|------|:---:|:---:|:---:|
| GCG (贪婪坐标) | ⚠️ 慢（全词汇表扫描） | ✅ 好 | 可改造 |
| **Faster-GCG** | ✅ 1.5x加速 | ✅ 好 | 可改造 |
| **AutoDAN** | ✅ 遗传算法更快 | ⚠️ 受限于模板 | 不太适合 |
| **MAGIC** | ✅ 梯度索引加速 | ✅ 好 | 可改造 |
| **MAC** (动量GCG) | ✅ 动量加速 | ✅ 好 | 可改造 |

> **建议**：先用标准 GCG 跑通概念级 Loss，验证可行后切换到 **MAC (动量加速)** 版本提效。

---

## Part 5：验证计划

1. **Exp08a**：在 Gemma-1B 上训练 3 组探针，验证探针准确率
2. **Exp08b**：跑概念级 GCG 搜索（预计 ~2-4h），对比标准 GCG
3. **Exp08c**：将找到的 Token 直接喂给未消融的原始模型（验证纯 Token 效果）
4. **Exp08d**：跨模型测试（Qwen/Llama，如有算力）

# AnyBody / TriTrack 实验编年（按时间）

本文把 **2026-08-26 至今** 已经做过的尝试按时间排在一起。  
SIRAC 只是最后一条新线；前面的 Mapper-B parent **没有被覆盖**。

统一冻结 parent（从 P0-Fix 起一直没换）：

```
人稀疏意图（头 / 左手 / 右手 SE(3)）
  → Mapper-B（72D，只看意图历史）
  → Stage-2 Encoder
  → g_φ,50000 残差
  → Stage-2 Decoder（29 关节）
```

任务环境：`MUSE-Kp-LatentRL-Kp5-HeadHands-Locomani-G1-v0`，策略 50 Hz，物理 200 Hz。  
人机接口始终是稀疏三点，不加手柄速度、不加地形 ID、不加任务专家。

SIRAC 单独的数字表见 [`sirac_phase1/RESULTS_TABLES.md`](sirac_phase1/RESULTS_TABLES.md)。

---

## 总览（一条故事）

| 阶段 | 时间 | 核心问题 | 冻结结论 |
| --- | --- | --- | --- |
| Parent | 8/26–8/27 | 稀疏意图能否驱动全身 | Mapper-B + `model_50000` 作为 canonical parent |
| 恢复 | 8/27–8/29 | 跟踪差了要不要在 latent 上推一下 | R-M3 单 burst 对 loco/stoop 有用；Reach/Carry 不扩 |
| 统一失败 | 8/29–8/30 | 失败是不是同一类、恢复能不能跨任务 | 覆盖不够，不是换任务补丁；Hold |
| 交互辨识 | 8/30 | 短脉冲能否认出恢复方向 | clone 能认，真机 50 Hz Jacobian 不够 |
| 意图投影 | 8/31–9/1 | 恢复会不会改人的意图 | 硬零空间有害；合理 tube 几乎不挡 UCR |
| 在线 ID | 9/1–9/2 | tube 内短码能否辨识接触 | P4-B4B 为 **B4B-PARTIAL**；**P4-C 未开** |
| SIRAC | 9/3 | 下肢要不要改成 7D 瓶颈 | 暂定 Case C：冻结 HTD 移植把头手打到约 2 m |

---

## 一、建立 Parent（8 月 26–27 日）

### 1. Stage-2 全身跟踪（此前已训完，ckpt 8/26）

- **尝试：** 用稀疏关键点（头/手，loco 常看躯干）训 AnyBody Stage-2：16D 单位 latent + residual `g_φ`，解码 29 关节。
- **控制：** Isaac Lab 2.1，G1 29 DoF，Beyond-Mimic 风格位置偏移。
- **结果：** `model_50000.pt` 成为后续所有实验的冻结解码器。不再重训 Stage-2。

### 2. Mapper-A（约 8/23–8/26）

- **尝试：** 用 MLP 填 Stage-2 观测里的未来关键点槽（人没有未来 buffer）。输入 **81D** = 8 个因果意图槽 + **当前机器人三点**。
- **离线指标：** latent cosine ≈ 0.989。
- **结果：** cosine 高，但闭环仍差。判定：失败不是 cosine 不够，而是 **训练分布和闭环分布不一致**——训练时强行 `robot KP = clip KP`，闭环一旦有几厘米滞后，多出来的 9D 就 OOD。Mapper 不该吃机器人状态。

### 3. 因果闭环 P0（8/26，`causal_closed_loop`）

- **尝试：** 同一 clip 上对比 Oracle / Hold / Mapper。
- **控制：** 冻结 `model_50000`，VR 三点 mask。
- **结果：** 确认 Mapper 有用，但 81D 耦合是隐患。进入 P0-Fix。

### 4. P0-Fix：Mapper-B（8/26 夜，`causal_p0_fix`）

- **尝试：** Mapper 改成 **72D 纯意图**：只从 `K_≤t^{人}` 预测 `K_>t^{人}`，去掉机器人当前 KP。并加 `g_φ=0` 对照。
- **控制：** 同数据、同 epoch、同损失；闭环 Oracle / Hold / A / B × 有无残差。
- **结果：** **Hold < Mapper-B ≲ Oracle**。72D Mapper-B 成为 canonical。权重：`victor/TriTrack/runs/mapper_b_intent72/mapper_best.pt`。

### 5. P1 矩阵 / 复验（8/27，`p1_matrix`、`p1_reverify`）

- **尝试：** 在 loco / stoop / reach / carry × 平地/轻粗糙/坡/台阶上复验 Mapper-B parent。
- **控制：** 任务只改 clip 和 mask（loco 常 torso，reach 常一头一手，stoop/carry 常 VR）。
- **结果：** Parent 可部署级跟踪；后续恢复实验都钉在这套 parent 上。

---

## 二、Latent 恢复该不该上（8 月 27–29 日）

主题：跟踪误差大时，要不要在 16D latent 上加一个有界修正 `z ← normalize(z + tanθ · d)`，θ 默认 5°。

### 6. P2 探针串（8/27 下午–夜）

包括 `p2_probe_50500`、`p2a`、`p2b`、`p2c`（高度扫描消融）、`p2d`、`p2_failure_archive`。

- **尝试：** 残差是否真在干活；高度扫描 / shuffle / zero 会不会泄漏地形。
- **结果：** 残差有贡献；**不能把地形 ID 塞进学生**。扫描消融说明方法必须对地形不可见。

### 7. P2-R Step 2–7（8/27 夜–8/28）

逐步问：恢复方向是否可辨识、6S MLP 监督恢复、闭环克隆效用。

| 步 | 文件夹 | 含义 | 结果一句话 |
| --- | --- | --- | --- |
| Step 3 | `p2r_step3` | stoop 的 S 通道是否可分 | 可分，作为后续门控 |
| Step 4–5 | `p2r_step4/5` | 可控性 / 探针 | 有方向可推 |
| Step 6S | `p2r_step6s_*` | 监督 MLP 预测恢复方向 | 有离线相关；闭环要看配对 I |
| Step 6S T/R/CL | `p2r_step6s_t/r/cl` | 迁移、鲁棒、闭环 | 不完全替代 parent |
| Step 7A/B | `p2r_step7a/7b_*` | 更大 dump / 全量 | 为 R-M 适配器备数据 |

### 8. R-M2 意图条件恢复（8/28，`rm_intent_conditioned_recovery`）

- **尝试：** 在 6S 之上加 **意图条件适配器**（单 burst，θ 分箱），不 PPO。
- **控制：** 冻结 Mapper-B + 50000；只训小适配器。
- **结果：** 适配器可部署为 R-M2 权重。

### 9. R-M3 共享单 burst（8/28 夜–8/29，`rm3_shared_single_burst`）

- **尝试：** Loco + Stoop 上 **一次** 5° burst；Reach/Carry 只影子记录、不执行。
- **控制：** 50 ep × 4 地形 × 4 变体 × 2 任务；Max-2 / PPO / Reach / Carry 执行 **禁止**。
- **结果：** **Case A。** Loco 不退化；Stoop-S 配对意图改善。决策看配对 `I_500`，不是 episode SR@5 跳高。停。不扩 Max-2。

### 10. T0 全任务矩阵（8/29，`t0_full_task_matrix`）

- **尝试：** 四任务正式任务成功率（Locomani 终止，不用 anchor_z/ori/fall 冒充任务 SR）。
- **结果：** 冻结评测母表。R-M3 只对 Loco/Stoop 激活。

### 11. T1 失败应力（8/29，`t1_*`）

T1 / T1.1 / T1.2 意图速度 / T1.3 统一时空应力 / T1.3u 四任务。

- **尝试：** 在 Mapper-B **之前** 对意图做时间伸缩、空间扰动，看 parent 何处碎。
- **控制：** 不改 parent 权重。
- **结果：** 标定了应力范围；后续 recovery 必须在这些扰动下谈，而不是只看干净 replay。

---

## 三、统一失败与跨任务恢复（8 月 29–30 日）

### 12. UFR-0/1/2（8/29 晚）

- **UFR-0** 统一失败表征：失败能不能用同一套特征说清。
- **UFR-1** 意图创新：新意图方向是否就是失败源。
- **UFR-2** parent 残差：失败是不是 g_φ 残差没覆盖。
- **结果：** 有结构，但不足以单独当控制器。

### 13. URA-0 / UCR-0 / UCR-1 / UCR-1D（8/29 夜–8/30）

- **尝试：** 再触发时的统一恢复场；辨识性（UCR-1D）。
- **UCR-1 数字（held-out clone，相对旧 R-M3）：**

| 任务 | 旧 P(I&lt;0) | 新 P(I&lt;0) | 新中位 I | 判定 |
| --- | --- | --- | --- | --- |
| loco | 0.65 | **0.79** | −1.68 cm | GO |
| stoop | 0.71 | 0.43 | +0.60 cm | HOLD |
| reach | 0.53 | 0.35 | +0.93 cm | HOLD |
| carry | 0.76 | 0.16 | +4.38 cm | HOLD |

- **结论：** **Case B HOLD。** 部分任务失败且 oracle 仍强 → 是覆盖/训练问题，不是做任务专家。Loco 变好，Carry/Reach 变差。禁止按任务分策略。

### 14. H2R-0 三点人体适配（8/30，`h2r_three_point_adapter`）

- **尝试：** 真人 SOMA 头/左右手 → 机器人 torso/腕，解析尺度，不 SMPL、不重训 Mapper。
- **结果：** 配对对齐率 1.0（约 10.7 万候选）。解析基线成立。停，不 PPO。

### 15. Phase NEXT（8/30，`next_phase`）

- **H2R-1 活体流：** 带约 40 ms 延迟/抖动/丢包的时间戳 SOMA。四任务 Live SR ≈ SOMA 解析 SR。判定 `LIVE_HUMAN_GO`。
- **UCR-2P：** 过程条件（200/500 ms kNN）并不稳定地好过静态 UCR-1D。不加 GRU 当方法。
- **THB：** 躯干高度缓冲，诊断用。

恢复在 H2R-1 里 **关掉**，避免和活体意图拧在一起。

---

## 四、交互条件恢复与短脉冲辨识（8 月 30 日）

### 16. ICR（`icr_interaction_recovery`）

- **尝试：** 只用最近的指令–响应 / proprio 历史，在 latent 上做恢复，**不要地形标签**。
- **对照：** M0 parent；M1/M2/M3 不同恢复模块。
- **结果：** **假设不成立。** 平地上 M0 SR 90%，加模块往往更差。不加失败门控。

### 17. IRR R3-short（`irr_response_recovery`）

- **尝试：** 1–2 步 ±ε 脉冲，能否认出有用的 5° 恢复方向。触发 `e≥0.13` 只用于评测。
- **控制：** n=158；`B_recovery` 来自成功 UCR 的 4D SVD，无地形 ID。

| 探针 | 符号正确率 | 相对 oracle | 代价 |
| --- | --- | --- | --- |
| 15 轴 clone，L=10 | 87% | 0.74 | 6000 ms |
| 15 轴 clone，L=1 | 72% | 0.48 | 600 ms |
| B k=3 clone，L=1 | 70% | 0.32 | **120 ms** |
| SPSA 1 步 | ~0 | ~0 | 40 ms |

- **结果：** **克隆世界**里短码有信息；顺序真机脉冲弱得多。后面 P4-B 继续问「在意图管里能否在线辨识」。

---

## 五、意图投影：恢复会不会改人的 WHAT（8 月 31 日–9 月 1 日）

人指定 WHAT（≤3 点），机器人决定 HOW。恢复必须尽量不泄漏到意图误差 `D_I`。

### 18. P3-A Oracle 投影（`p3_intent_projected_adaptation`）

- **尝试：** 硬/软投影，把恢复方向里「会推动意图」的分量削掉。
- **结果：** 硬投影把 `D_I` 压得很低，但也削掉恢复权威。软 λ=1 进入 P3-B。

### 19. P3-B / B2 / B3 学习投影（8/31）

| 尝试 | 含义 | 结论 |
| --- | --- | --- |
| P3-B | 学一个意图投影器 | 恢复还在，意图泄漏偏高 |
| P3-B2-0 | 全局 λ 标定 | 失败：各向异性，不是纯缩放 |
| P3-B2-1 | 保守 shield | 失败：A 还在，`D_I` 仍约 2× |
| P3-B3 | 尾部风险 ensemble | **U3：不要做不确定性盾** |

学习投影器 **只作诊断，不进部署架构**。

### 20. P3-B4a 谱 / B4b 解析 Jacobian（9/1）

- **B4a：** loco 躯干意图协方差秩 ≤3；前 2 维抓住约 93% trace / 95% UCR 风险；第 3 轴 400 ms 转约 80°。不训谱坐标系。
- **B4b：** 解析运动学 Jacobian 与 oracle Spearman 仅 0.15，泄漏约 19×（B1 约 4.7×）。**拒绝**当方法。不要回到 120 ms Jacobian。

---

## 六、意图管 + 在线辨识（9 月 1–2 日）— P4-C 始终未开

原则：跟踪松弛 ≠ 改人的意图。允许 `E_I ≤ ε_tube`，不要 `E_I=0`。

### 21. P4-A0 Oracle 安全克隆（9/1）

- **尝试：** 严格意图零空间 `P_I = I − J†J` 是否可部署。
- **结果：** **Case B。** 准确率 0.66 → 0.53。硬零空间破坏信息。

### 22. P4-A1 意图管（9/1，`p4a_intent_tube_interface`）

- **尝试：** 用管 `f≥0.1` 代替硬投影。
- **结果：** **Case T1。** `f=0` acc=0.536；`f≥0.1` acc≈0.66≈原始 UCR。合理 tube **几乎不挡** `B_UCR`。Raw UCR 5° 中位意图偏差约 **0.4 mm**——此前意图盾高估了冲突。
- **冻结：** Stage-2 latent 仍是自主接口。删除硬 `P_I`。不跑 P4-A2。不改成 TrajBooster `[vx,vy,vyaw,h]` 当人体接口。

### 23. P4-B 管约束在线辨识（9/1–9/2）

在 tube 里对 `B_UCR^{k=3}` 做编码复用脉冲，1° 合成，5 mm 管守卫。

| 子阶段 | 文件夹 | 结论 |
| --- | --- | --- |
| P4-B | `p4b_tube_online_id` | 在线 ID 骨架 |
| P4-B2 | `p4b2_twin_referenced_id` | twin 参照 |
| P4-B3 | `p4b3_oracle_residual_id` | 50 Hz 恒 Jacobian **关闭**。完美 twin 残差 Acc=0.544，不是 H1。80 ms 符号翻转 23%。不训 Virtual Twin。 |
| P4-B4A | `p4b4_fast_active_response` | 端点重标记 |
| P4-B4B | `p4b4b_200hz_microprobe` | **B4B-PARTIAL**（见下） |

### 24. P4-B4B 200 Hz 微探针（9/2，最后一条旧线）

- **控制：** `dt=5 ms`，decimation=4；n=158；主配置 `n4_a0.5_default`；地形 steps/slip/slope_down；**GPU 0**；P4-C 明确 `NOT_RUN`。
- **数字（shrinkage LDA）：** 方向先验 Acc≈0.65；码本身 Acc≈0.58；F2/扭矩约 0.66。相对 D1 的 ΔAcc 仅 0.042，相对 D0 仅 0.008。3D 码能到 PD（秩≥3），twin 安全（5 mm 违约≈0）。
- **判定：** **B4B-PARTIAL。** Acc 主要由轴向先验解释，不是可扩的解码器。不扩 decoder，不训网，不回 120 ms Jacobian，**不开 P4-C**。

---

## 七、SIRAC：换下肢表示（9 月 3 日）— 新方向

旧线停在「Stage-2 仍输出全身关节，恢复只动 16D latent」。  
SIRAC 另开一条：**下肢不再由 Stage-2 直接出 15 维关节，而由 7D realization + 冻结 HTD 学生执行。** Mapper-B parent 仍是 Baseline A。

### 25. Phase 0 审计 + 接口（白天）

- 核对 HTD 学生：58D×2 历史、7D 指令顺序、15D `q = q_htd + 0.25 a`、关节按名字对齐。
- 单测 24/24。Dummy smoke **不算** 科学结果。
- JIT 后补下载成功，推理输出有限 15 维。

### 26. Isaac A/B/C（9/3 晚，GPU 0/5/6/7）

| 尝试 | GPU | 自变量 | 头位置 | 含义 |
| --- | --- | --- | --- | --- |
| A plane | 0 | 旧 parent 全身 29D | **7.4 cm** | 旧方法仍在 |
| B plane | 5 | 7D + 冻结 LBC，手臂仍 Stage-2 | **2.06 m** | 移植失败 |
| C plane | 6 | 同 B，手臂锁死 | **2.16 m** | 不是上肢扰动 |
| B light_rough | 7 | 同 B，轻粗糙 | **2.14 m** | 不是「只在平地坏」 |

- **暂定 Case C：** 先查坐标系 / 指令语义 / AnyBody PD 与 HTD action scale。不要上 ADAPT、PPO、大残差。
- `fell=1` 三线都有（腕部 `ee_z` 阈值），**不能当摔倒率**。

明细表：[`sirac_phase1/RESULTS_TABLES.md`](sirac_phase1/RESULTS_TABLES.md)。

---

## 八、什么始终没做 / 故意冻结

| 项 | 状态 |
| --- | --- |
| 重训 Stage-2 / Mapper-B | 未做（冻结） |
| 地形 ID、任务专家、每地形一个策略 | 禁止 |
| 手柄速度作为人接口 | 禁止（SIRAC 的 7D 是机器人内部变量） |
| TrajBooster 式人体接口 | P4-A1 明确拒绝 |
| P4-A2、P4-C | 未跑 |
| Virtual Twin 训练 | 未做 |
| SIRAC 的 ADAPT 残差 / 未来意图学生 | 接口在，未训练 |

---

## 九、现在该站在哪

两条平行线：

1. **旧 parent（有效）：** 稀疏意图 → Mapper-B → Stage-2 29 关节。R-M3 只在 loco/stoop 单 burst 上被验证；统一恢复未跨任务 GO；P4 在线辨识停在 B4B-PARTIAL。
2. **SIRAC（未通）：** 想把下肢换成 7D 瓶颈 + HTD 学生。当前把头手从厘米打到米。下一步是修移植，不是加网络。

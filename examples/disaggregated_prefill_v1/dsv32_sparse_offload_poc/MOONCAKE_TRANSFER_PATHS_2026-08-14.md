# DeepSeek-V3.2 Sparse KV Offload Mooncake 传输路径调查结论

日期：2026-08-14

本文记录本分支对 DeepSeek-V3.2 Online P/D + Sparse KV Offload 传输路径的
源码分析、真实 NPU 探针和最终决策。它专门回答以下问题：

- 当前真实模型闭环使用哪条路径；
- mentor 建议的 Host-to-Host 路径与当前路径有什么区别；
- 为什么普通 pinned Host、framework swapped Full KV 和普通 NPU staging
  不能混为一谈；
- 哪些路径已经通过、哪些已经失败，以及结果究竟证明了什么；
- 下一步应该继续什么、停止什么。

## 1. 固定术语

后续统一使用以下名称，不再单独使用含义不明确的“Host DDR”“Host KV”等简称。

| 统一名称 | 分配方式/位置 | 暴露的关键地址 | 当前用途 |
|---|---|---|---|
| **普通 NPU staging Tensor** | 普通 `torch.empty(..., device="npu")`，物理位于 HBM | 普通 NPU 地址 | 模型写入、Mooncake Ascend 传输、写入 swapped Full KV |
| **framework swapped Full KV** | `torch_npu.empty_with_swapped_memory(...)`，`device` 为 NPU、物理位于 Host DDR | 当前框架暴露 NPU/SVM alias | 持久保存 Full NoPE/RoPE KV，供 CANN Gather 读取 |
| **pinned Host relay** | `torch.empty(..., device="cpu", pin_memory=True)`，物理位于 Host DDR | 普通 CPU Host 地址 | Mooncake TCP Host-to-Host 传输的临时中转缓冲区 |
| **Indexer NPU KV** | 普通 NPU Tensor，物理位于 HBM | 普通 NPU 地址 | Lightning Indexer/Top-K 相关 KV，经 Mooncake Ascend 传输 |
| **selected NPU KV** | Gather 输出的普通 NPU Tensor，物理位于 HBM | 普通 NPU 地址 | Sparse FlashAttention 输入 |

关键区别：

```text
framework swapped Full KV 和 pinned Host relay 虽然物理上都位于 Host DDR，
但它们是不同 allocator 创建的独立 Tensor，拥有不同地址语义，不能互相改名、
view 或零拷贝转换。
```

`empty_with_swapped_memory` 的官方定义是“device 信息为 NPU、实际内存在 Host
侧的特殊 Tensor”。CANN Gather 官方 Host Offload 示例使用的也是这种特殊 Tensor，
不是普通 CPU pinned Tensor：

- [torch_npu `empty_with_swapped_memory` 文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/at_npu-native-empty_with_swapped_memory.md)
- [CANN Gather 算子文档](https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/docs/custom-npu_gather_selection_kv_cache.md)
- [CANN Gather 官方示例](https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/examples/test_npu_gather_selection_kv_cache.py)

## 2. 当前唯一真实模型闭环路径：`npu_staging`

实际数据路径必须完整写成：

```text
P 模型算子写入 P 普通 NPU staging
              │
              ├── 本地 copy P-NPU→P-swapped
              │        ▼
              │   P framework swapped Full KV
              │   （持久化本轮 touched slots，支持本地 Full KV 生命周期）
              │
              └── Mooncake Ascend NPU→NPU
                       ▼
                  D 普通 NPU staging
                       │
                       └── 本地 copy D-NPU→D-swapped
                                ▼
                           D framework swapped Full KV
                                │
                                └── CANN Gather 从 swapped/SVM 读取
                                         ▼
                                    D selected NPU KV
                                         │
                                         ▼
                                 Sparse FlashAttention

P Indexer NPU KV ── Mooncake Ascend NPU→NPU ──> D Indexer NPU KV
```

这里有两条互不替代的 P 端动作：

1. `persist_updated_slots()` 把 P 普通 NPU staging 中的 touched rows 写入
   P framework swapped Full KV；
2. Mooncake Ascend 直接以 P 普通 NPU staging 为发送源，把 Full KV 传到
   D 普通 NPU staging。

D 端收到后，先把 Full KV 从 D 普通 NPU staging 写入 D framework swapped
Full KV，完成后再 ACK，随后 Gather 从 D framework swapped Full KV 中选出
Top-K KV 到 D selected NPU KV。

### 这条路径为什么可以

- Mooncake Ascend 已验证可以注册和传输普通 NPU Tensor；
- 当前 runtime 已验证普通 NPU Tensor 可以通过 basic-slice `copy_` 写入
  framework swapped Full KV；
- CANN Gather 已验证可以读取 framework swapped Full KV 的 NPU/SVM alias；
- 每一个相邻接口都有真实 NPU 证据。

### 已取得的真实模型证据

- DeepSeek-V3.2 W4A8，运行时 BF16 KV；
- 61 个主模型层；
- TP8 Prefill + TP8 Decode；
- Online P/D + MooncakeLayerwiseConnector + Sparse KV Offload；
- `max_model_len=4096`；
- 五个用例覆盖 Top-K 阈值以下、阈值以上、约 3K、近 4K 和请求状态重置；
- 每个请求均有 `Prefill=8, Decode=8`；
- 输出：`FINAL 4K ONLINE PD SUITE: PASSED`；
- 验证修订：`16f7d033d0f45581b76c31a0e995020b0ee6af8a`。

该结果证明端到端正确性，不证明性能最优、跨物理节点、高并发或生产可用。

## 3. mentor 建议的目标路径

mentor 关注的是避免跨 P/D 的 Full KV 进入 Decode NPU，从而减少对 Decode TPOT
的潜在影响。理想路径应当精确写成：

```text
P 普通 NPU staging
        │
        └── 本地 copy P-NPU→P-swapped
                 ▼
            P framework swapped Full KV
                 │
                 │ Mooncake Host transport 读取同一 allocation 的 CPU 地址 H_P
                 │ 并写入 D allocation 的 CPU 地址 H_D
                 ▼
            D framework swapped Full KV
                 │
                 │ CANN Gather 使用同一 allocation 的 NPU/SVM 地址 S_D
                 ▼
            D selected NPU KV

P Indexer NPU KV ── Mooncake Ascend NPU→NPU ──> D Indexer NPU KV
```

它要求每个 framework swapped Full KV allocation 同时暴露：

```text
CPU Host 地址 H：供 Mooncake Host transport 读写
NPU/SVM 地址 S：供 NPU copy 和 CANN Gather 访问
H 与 S 必须映射到同一组 Host DDR 物理页，并具有明确的生命周期和同步语义
```

### 当前为什么跑不通

当前 Python/Connector 只拿到 framework swapped Tensor 的 NPU/SVM alias。
在现有 Mooncake 0.3.12.post1 + CANN 9.1 + torch_npu 2.10 dev runtime 中：

- Mooncake AscendDirect/HIXL 直接传 swapped `data_ptr()` 失败；
- Mooncake TCP 可以传普通 CPU/pinned Host 地址，但当前拿不到 swapped allocation
  对应的原始 CPU Host 地址；
- 普通 pinned Host 地址又不能直接作为 CANN Gather 的 Full KV 输入。

因此阻塞点不是物理上不存在 Host 内存，而是当前软件栈没有暴露“同一 swapped
allocation 的 CPU 地址 H + NPU/SVM 地址 S”这一双地址契约。

## 4. 2026-08-14 实际实现并通过的兼容性 `host_relay`

这条路径必须完整写成：

```text
                              ┌── 本地 copy ① P-NPU→P-swapped
P 普通 NPU staging ──────────┤
                              │        ▼
                              │   P framework swapped Full KV
                              │
                              └── 本地 copy ② P-NPU→P-pinned
                                       ▼
                                  P pinned Host relay
                                       │
                                       └── Mooncake TCP Host-to-Host
                                                ▼
                                           D pinned Host relay
                                                │
                                                └── 本地 copy ③ D-pinned→D-NPU
                                                         ▼
                                                    D 普通 NPU staging
                                                         │
                                                         └── 本地 copy ④ D-NPU→D-swapped
                                                                  ▼
                                                             D framework swapped Full KV
                                                                  │
                                                                  └── CANN Gather
                                                                           ▼
                                                                      D selected NPU KV

P Indexer NPU KV ── Mooncake Ascend NPU→NPU ──> D Indexer NPU KV
```

Mooncake TCP 成功传输的是独立的 **pinned Host relay**，不是 P framework
swapped Full KV。P framework swapped Full KV 与 P pinned Host relay 都由
P 普通 NPU staging 填充，是两次独立 copy。

### 这条路径为什么可以

- P 普通 NPU staging → P pinned Host relay：普通 NPU→CPU pinned copy 可用；
- Mooncake TCP：普通/pinned Host buffer 注册、传输和析构已通过；
- D pinned Host relay → D 普通 NPU staging：普通 CPU pinned→NPU copy 可用；
- D 普通 NPU staging → D framework swapped Full KV：已验证的 basic-slice
  NPU→swapped copy；
- Gather：读取的仍然是它已支持的 framework swapped Full KV。

### 实际探针结果

| 探针 | 结果 | 精确证明范围 |
|---|---|---|
| `probe-host-transfer` | `2 passed`，52.17s | Mooncake TCP 可传普通 Host 与 pinned Host buffer |
| `probe-hybrid-transfer` | `1 passed`，31.75s | 同一 worker 中 Ascend TE 与 TCP TE 可以共存，Ascend→TCP→Ascend 序列未破坏 |
| `probe-host-relay` | `1 passed`，29.99s | 上述四次 copy + TCP + swapped Gather 的最小 BF16/真实维度链路正确 |

### 为什么不能把它作为性能候选

- Full KV 仍然经过 D 普通 NPU staging；
- 相比 `npu_staging` 多了 P-NPU→P-pinned 和 D-pinned→D-NPU；
- 没有消除 mentor 关心的 Decode NPU Full-KV 流量；
- 当前只通过最小真实 NPU 探针，没有通过 16 卡真实模型；
- 即使真实模型能跑，也不能从架构上预期它优于 `npu_staging`。

因此它只能保留为接口诊断证据，不应继续产品化或用于性能 A/B。

## 5. 已经失败的直接路径

### 5.1 Mooncake 直接传 framework swapped Full KV

尝试路径：

```text
P framework swapped Full KV
        │
        └── Mooncake AscendDirect/HIXL
                 ▼
            D framework swapped Full KV
```

真实结果：失败。

关键错误：

```text
rtsHostRegister execution failed, reason=driver error:invalid handle
Mooncake mixed sparse-offload transfer failed: result=-1
```

当前测试保留为默认不运行的诊断性 xfail：

```text
tests/ut/distributed/kv_transfer/a3_2/
test_mooncake_transfer_engine_npu.py::
test_mooncake_sparse_offload_mixed_memory_transfer
```

它证明当前 Mooncake/HIXL 不能直接使用框架暴露的 swapped/SVM `data_ptr()`；
不证明未来版本或新的双地址 API 永远不能支持。

### 5.2 D pinned Host 直接 `copy_` 到 D framework swapped Full KV

尝试路径：

```text
D pinned Host relay
        │
        └── Tensor.copy_ 直接写 swapped
                 ▼
            D framework swapped Full KV
```

真实结果：第一版 `probe-host-relay` 中接收端成功收到 TCP payload，但在执行验证后
子进程退出，没有产生最终结果，整体测试失败。提交 `08bd5117d` 把它改成：

```text
D pinned Host relay
→ D 普通 NPU staging
→ D framework swapped Full KV
```

同一探针随后通过。

因此当前稳定边界是 NPU→swapped basic-slice copy；不能把普通 pinned CPU Tensor
零拷贝“变成”swapped Tensor，也不能把直接 pinned→swapped `copy_` 当作已支持能力。

### 5.3 CANN Gather 直接读取普通 pinned Host Full KV

尝试路径：

```text
D pinned Host Full KV
        │
        └── CANN npu_gather_selection_kv_cache
                 ▼
            D selected NPU KV
```

真实结果：失败。

关键错误：

```text
MTE accesses an invalid GM address or the cross-device memory access times out
error code 507035
```

这证明当前 Gather 的“Host Full KV 支持”不能解释为“任意普通 CPU pinned Tensor
都可直接传入”。官方示例所使用的是 framework swapped/SVM Tensor。

## 6. 更早的 MemCache 探索与同类问题

在切换 Mooncake 以前，MemCache mixed-memory 路线也曾尝试同时处理：

```text
Full NoPE/RoPE：framework swapped Full KV
Indexer：普通 NPU Tensor
```

当时先遇到 `Invalid direct 4 for batch copy`，之后改用普通 NPU staging 和公开
L2G/G2L 方向又出现 segmentation fault/double free。虽然 MemCache API 与 Mooncake
不同，但它同样提前暴露了“传输后端不能天然理解 framework swapped 地址”的问题。

因此后来的 Mooncake `npu_staging` 并非随机选择，而是为了找到所有相邻组件都已
验证支持的公共内存边界，先闭环真实模型正确性。

## 7. Mooncake Store 为什么不解决当前问题

Mooncake Store 提供 Key/Object、分布式容量、放置、查询、淘汰和生命周期管理，
底层数据移动仍依赖 TransferEngine。它可以管理 DRAM/可选 VRAM/SSD 中的对象，
但不会自动为一个 Store DRAM allocation 同时生成 Gather 可访问的 NPU/SVM alias。

因此在没有证明 Store segment 同时满足以下条件以前，不应接入 Mooncake Store：

```text
Mooncake 可通过 CPU 地址写入
+
CANN Gather 可通过 NPU/SVM 地址读取
+
两个地址映射到同一 allocation
```

当前 Online P/D 是当前请求的 KV handoff，不需要为了这个问题引入共享 KV Pool。

## 8. 最终证据矩阵

| 数据路径/接口 | 状态 | 证据等级 | 结论 |
|---|---|---|---|
| 普通 NPU → framework swapped | 通过 | 真实 NPU + 真实模型 | 当前唯一稳定写入 swapped 的方式 |
| framework swapped → Gather → selected NPU | 通过 | 真实 NPU + 真实模型 | 当前 Sparse Gather 正式输入路径 |
| 普通 NPU → Mooncake Ascend → 普通 NPU | 通过 | 真实 NPU + 真实模型 | 当前 P/D Full KV 与 Indexer 传输基线 |
| pinned Host → Mooncake TCP → pinned Host | 通过 | 真实 NPU 最小探针 | 只证明普通 Host transport |
| Ascend TE 与 TCP TE 同进程共存 | 通过 | 真实 NPU 最小探针 | 只证明双引擎生命周期 |
| framework swapped → Mooncake | 失败 | 真实 NPU | 当前 swapped/SVM 地址不能直接注册/传输 |
| pinned Host → framework swapped 直接 `copy_` | 失败 | 真实 NPU | 当前不能省略 NPU staging bridge |
| pinned Host → Gather | 失败 | 真实 NPU | 当前 Gather 不接受该普通 CPU Tensor 接口 |
| `host_relay` 四次 copy 兼容链 | 通过 | 真实 NPU 最小探针 | 功能 fallback，不是性能目标 |
| `npu_staging` 真实 61 层 TP8+TP8 | 通过 | 真实模型端到端 | 当前唯一正式可运行基线 |
| P swapped → D swapped → Gather | 未打通 | 两个关键直接接口已失败 | 等待双地址/注册能力，不继续应用层绕路 |

## 9. 最终决策

### 继续

1. 保留 `SPARSE_KV_TRANSFER_MODE=npu_staging` 作为默认和正式基线；
2. 在真实 Prefill/Decode 重叠负载下 profile 该基线，测量 TTFT、TPOT
   p50/p95/p99、Full-KV Mooncake 传输、D NPU staging→swapped copy、ACK wait、
   Gather 和 SFA；
3. 向 CANN/torch_npu/Mooncake 负责人提交最小复现和接口需求。

下层满足以下任一能力后，才重新实现 mentor 目标路径：

- 暴露同一个 `empty_with_swapped_memory` allocation 的 CPU Host 地址 H 和
  NPU/SVM 地址 S；
- Mooncake AscendDirect/HIXL 原生接受 framework swapped allocation；
- CANN Gather 正式接受普通 pinned Host Tensor；
- 或提供另一种明确保证 Mooncake 写入与 Gather 读取同一物理页的统一 allocator。

### 停止

1. 不把 `host_relay` 当作性能候选；
2. 不运行 16 卡 `host_relay` 真实模型 A/B；
3. 不继续追加同类 pinned/swapped bridge 探针；
4. 不接 Mooncake Store；
5. 不宣称 Host-to-Host Sparse KV 路径已经实现；
6. 不把最小探针通过表述成真实模型或性能收益。

## 10. 一句话结论

当前唯一由真实 DeepSeek-V3.2 模型验证可运行的路径是：

```text
P 普通 NPU staging
→ Mooncake Ascend NPU→NPU
→ D 普通 NPU staging
→ 本地 copy D-NPU→D-framework-swapped
→ CANN Gather
→ D selected NPU KV
```

mentor 建议的目标路径在架构上合理，但当前软件栈缺少同一 framework swapped
allocation 的 CPU Host 地址与 NPU/SVM 地址双重暴露，不能通过继续增加应用层 copy
来真正实现；`host_relay` 只是一个已经完成使命的兼容性诊断实验。

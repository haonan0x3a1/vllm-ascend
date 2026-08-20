# DeepSeek-V3.2 Sparse KV Offload Online P/D PoC

本目录把同节点 DeepSeek-V3.2 Sparse Host KV Offload 的 Online P/D
启动、预检、验收和证据归档固化为一个入口。它面向正确性 PoC，
不是性能或生产部署脚本。

Mooncake 传输路径、内存类型、真实 NPU 探针和最终 Stop/Go 结论见
[MOONCAKE_TRANSFER_PATHS_2026-08-14.md](MOONCAKE_TRANSFER_PATHS_2026-08-14.md)。

已验证基线：

- 真实 DeepSeek-V3.2 W4A8 权重，运行时 BF16 KV；
- 61 个主模型层，TP8 Prefill + TP8 Decode，Expert Parallel；
- `MooncakeLayerwiseConnector` 逐层 NPU staging；
- framework swapped Full KV、Lightning Indexer Top-K、CANN Gather 和 SFA；
- `max_model_len=4096`，Prompt 覆盖 1818–3618 tokens；
- 超过 `index_topk=2048` 的请求强制执行 8 个 Decode tokens；
- 单请求模式下每个请求完成 8 个 Prefill 和 8 个 Decode rank 生命周期。

## 一次性配置

在服务器容器内执行：

```bash
cd /workspace/w50062541/code/vllm-ascend/examples/disaggregated_prefill_v1/dsv32_sparse_offload_poc
cp --update=none config.example.env config.env
```

根据实际服务器检查并修改 `config.env`。该文件被 Git 忽略，不会提交服务器路径和端口。

配置至少包括：模型路径、Mooncake Python、自定义 OPP、P/D 卡号、API 端口和两个
KV 控制端口基址。共享服务器上的端口和 NPU 所有权是动态状态，
不能写死为“永远可用”。

## 每次启动

先确保 Prefill、Decode 和 Proxy 均未运行，然后执行：

```bash
bash run.sh preflight
```

`preflight` 验证路径、依赖 API、卡号映射、API/KV 控制端口以及 Mooncake ADXL
端口余量。为避免复用过 Prefill/Decode 终端时只看到半组卡，它会在自己的临时
进程中忽略继承的 `ASCEND_RT_VISIBLE_DEVICES`；启动 Prefill/Decode 时仍按
`config.env` 分别设置卡号。它只能验证 NPU 可见性，不能判断共享卡的实际所有权；
还必须确认 `npu-smi info` 中配置的 16 张卡没有其他人的进程。

## Host 传输可行性门槛

当前已验证基线使用 Mooncake Ascend transport 将逐层缓存传入 Decode 普通 NPU
staging。为了评估 Full KV 改走
`P framework swapped Full KV -> D framework swapped Full KV` 是否能减少对
Decode NPU 的干扰，先单独验证当前 Ascend Mooncake wheel 的 TCP Host transport：

```bash
bash run.sh probe-host-transfer
```

该命令不启动模型，分别使用普通 Host 内存和 pinned Host 内存运行两进程传输，
每种内存连续传输 20 个带 guard 的 payload，并检查注册、反注册和进程析构。
测试会在子进程启动前设置 `MC_FORCE_TCP=1`，并要求 Mooncake 原生日志明确出现
`MC_FORCE_TCP is set, using TCP transport only`，避免把 Ascend transport 误认为
Host TCP。该选择机制来自
[Mooncake v0.3.12.post1 TransferEngine 初始化逻辑](https://github.com/kvcache-ai/Mooncake/blob/v0.3.12.post1/mooncake-transfer-engine/src/transfer_engine_impl.cpp#L205-L233)。
脚本只有看到 `2 passed` 才返回成功，`2 skipped` 不算通过；完整输出保存在
`LOG_DIR/dsv32-mooncake-host-transfer-probe.log`，且不得出现 segfault/double free。

这只是新路径的底层 Stop/Go 门槛，不代表 Connector 或真实模型已经切换到 Host
relay。通过后还要依次验证本地 NPU-to-Host、Host-to-NPU-to-swapped/Gather，
以及 Full KV Host transport 与 Indexer NPU transport 的组合生命周期；任一步失败
都不应改动当前默认的 NPU staging 路径。

Host-to-Host 两种内存均通过后，继续验证同一个 worker 进程里的双引擎生命周期：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 bash run.sh probe-hybrid-transfer
```

其中 `8,9` 只是示例；每次运行前必须用 `npu-smi info` 选择两张当时确实空闲的
物理卡。脚本要求显式传入且恰好两张卡，不会清除该映射或擅自触碰物理 0、1 卡；
测试内部看到的逻辑 `npu:0/1` 分别映射到这两张物理卡。

该命令先创建原有 Mooncake Ascend engine，再临时设置 `MC_FORCE_TCP=1` 创建独立的
Host TCP engine，随后立刻恢复进程环境。它依次执行：

```text
Ascend Indexer payload A
  -> TCP pinned-Host Full-KV payloads
  -> Ascend Indexer payload B
```

第二次 Ascend 传输位于 TCP 传输之后，用来确认新增 Host engine 没有破坏原 NPU
engine。发送端和接收端都必须分别出现 Ascend 与 TCP-only 原生日志，使用不同 RPC
端口，通过 payload/guard 校验，并在反注册和析构后正常退出。脚本只有看到 `1 passed`
才返回成功，证据保存在
`LOG_DIR/dsv32-mooncake-hybrid-transfer-probe.log`。

这个测试通过也只确定“双引擎可共存”的实现前提。后续兼容性探针虽然验证了：

```text
P 普通 NPU staging -> P pinned Host relay -> D pinned Host relay
-> D 普通 NPU staging -> D framework swapped Full KV -> Gather
```

但它没有消除 Decode 端 Full KV 经过普通 NPU staging，因此不构成目标传输路径。

双引擎门槛通过后，曾使用下面的兼容性探针验证 Host 数据可以经现有
NPU staging 桥接进入 swapped Full KV：

```bash
ASCEND_RT_VISIBLE_DEVICES=8,9 bash run.sh probe-host-relay
```

同样要把 `8,9` 换成当时空闲的两张物理卡。该测试只运行 BF16 和真实
DeepSeek-V3.2 Full-KV 维度：NoPE `[4,128,1,512]`、RoPE `[4,128,1,64]`。
它只复制非恒等逻辑映射 `[[2,1]]` 使用的物理块 2、1，并在一个用例内验证：

```text
Prefill NPU staging
  -> Prefill pinned Host
  -> Mooncake TCP Host-to-Host
  -> Decode pinned Host
  -> Decode ordinary NPU staging
  -> Decode framework swapped Full KV
  -> CANN Gather
  -> Decode selected NPU KV
```

测试还会确认物理块 0、3 与 relay guard 未被修改，检查 Host/Ascend 双引擎的
原生日志、内存反注册和进程析构。Decode 侧使用普通 NPU staging 作为 Host buffer
到 swapped Full KV 的本地桥接，与生产 Connector 已验证的 basic-slice persistence
路径保持一致；不直接把 pinned Host 地址写入 swapped/SVM alias。脚本只有看到
`1 passed` 才返回成功，证据保存在
`LOG_DIR/dsv32-mooncake-host-relay-probe.log`。它证明的是兼容性 fallback，
并没有消除 Decode 端 Full KV 经过 NPU staging，因此不是目标 Host 路径，
也不能据此开始性能对比。

用于决定路线的 direct Host Gather 硬件门槛为：

```bash
ASCEND_RT_VISIBLE_DEVICES=8 bash run.sh probe-direct-host-gather
```

其中 `8` 必须替换为当时空闲的一张物理卡。该测试把真实 DeepSeek-V3.2 BF16
Full-KV 维度放在普通 pinned Host Tensor 中，使用非恒等物理块映射 `[[2,1]]`，
直接调用真实 CANN `npu_gather_selection_kv_cache`，不经过 Decode NPU Full-KV
staging，也不使用 swapped Tensor。测试在独立子进程执行，检查 selected KV 数值、
Host 源数据和 guard，并要求子进程正常退出。

该门槛已在当前 CANN/torch_npu/Mooncake 环境的真实 NPU 上运行并失败，关键错误为
`MTE accesses an invalid GM address`。这说明当前公开 Tensor/算子接口不能让 CANN
Gather 直接读取普通 pinned Host Tensor。停止扩展 `host_relay`，保留已验证的
`npu_staging` 基线，并把同一 swapped allocation 的 CPU Host 地址/NPU SVM alias
支持作为下层依赖问题。除非 CANN、torch_npu 或 Mooncake 的相关接口发生变化，
否则不再重复运行该探针或追加同类 bridge 探针。

## MemFabric BM 双视图 Gather G1

目标 Host-to-Host 路径的第一硬件门槛，不再使用普通 pinned Host Tensor。它直接
验证 MemFabric BM 的同一块 DRAM pool 是否可以同时提供：

```text
LOCAL_HOST VA
└── 仅验证地址转换；Host 数据读写通过 BM H2G/G2H API

LOCAL_DEVICE VA
└── 构造 torch NPU Tensor alias，供 NPU copy 和 Gather 使用
```

该探针位于
`tests/ut/distributed/kv_transfer/a3_2/test_memfabric_bm_dual_view_gather_npu.py`，
默认跳过，必须在确认一张空闲 A3 NPU 后显式启用。例如物理 8 卡空闲时：

```bash
cd /workspace/w50062541/code/vllm-ascend

ASCEND_RT_VISIBLE_DEVICES=8 \
VLLM_ASCEND_RUN_MEMFABRIC_BM_GATHER_GATE=1 \
pytest -sv \
  tests/ut/distributed/kv_transfer/a3_2/test_memfabric_bm_dual_view_gather_npu.py::test_memfabric_bm_dual_view_host_full_kv_gather_gate
```

测试完全运行在独立子进程中，依次验证：

1. framework swapped Full KV 可以执行真实 Gather，作为当前环境的正对照；
2. MemFabric BM `HOST` GVA 可转换为 `LOCAL_HOST` 和 `LOCAL_DEVICE` 两个地址；
3. `LOCAL_DEVICE` 地址可按 model runner 的 raw-int8、`view/as_strided` 方式构造成
   BF16 NoPE/RoPE NPU Tensor；
4. `persist_updated_slots` 可以执行 copy ①：普通 NPU staging → BM Host Full KV；
5. 使用不同数据通过 BM `H2G` 写入同一 Host pool 后，真实 Gather 能从 NPU alias
   读取新数据，而不是误读 copy ① 的旧数据；
6. 非目标 block、tensor 间隔和前后 guard 未被修改；
7. 所有 Tensor view 在 BM handle 释放前销毁，进程正常退出且不出现
   segfault/double free。

MemFabric 1.1.2 的 DRAM segment 要求按 1 GiB 对齐，因此探针默认创建 1 GiB
BM pool，但只初始化和检查 Full-KV 布局实际覆盖的区域。该版本返回的
`LOCAL_HOST` VA 不应由 Python `ctypes` 直接解引用；这不是目标数据路径的必要
条件，CPU 侧初始化与校验统一使用 BM `H2G/G2H`，而核心门槛仍是
`LOCAL_DEVICE` alias 能否被 NPU copy 和真实 Gather 读取。

只有输出包含：

```text
MemFabric BM dual-view Host Full-KV -> Gather G1 PASSED
1 passed
```

才判定 G1 为 Go。以下任一情况均为 No-Go 或下层依赖阻塞：

- `gva_to_va(..., LOCAL_DEVICE)` 不存在或返回 0；
- pointer → NPU Tensor 构造接口不可用；
- NPU staging → BM alias 的 `index_copy_` 失败；
- Gather 报 invalid GM address、进程超时或异常退出；
- Gather 仍读到 Host 写入前的旧数据；
- guard 损坏、析构崩溃或 double free。

该测试只使用单 rank BM `SDMA` 验证 allocator、双地址和 Gather 契约，不验证
`HOST_RDMA` 或跨机可见性。G1 通过后才能继续 G2：

```text
P BM Host Full KV
→ MemFabric HOST_RDMA
→ D BM Host Full KV
→ completion/visibility fence
→ D Gather
```

## MemFabric BM 同机双进程 G2a

当前开发环境是一台 16-NPU 服务器，Prefill/Decode 只是用不同 NPU 和独立进程
模拟分离，因此不能宣称跨机 `HOST_RDMA` 已验证。G2 拆成两个门槛：

- G2a：同一物理 Host 上的双进程、双 NPU、两份 BM Host allocation；验证
  P copy①、BM `G2G`、完成通知以及 D `LOCAL_DEVICE` alias 的真实 Gather；
- G2b：未来两台物理 Host 上使用相同数据契约和 `HOST_RDMA`，补测 RNIC、远端
  注册、跨机可见性、fence 和性能。

G2a 默认使用适合同机语义的 `HOST_SHM`。确认两张空闲 NPU 后，例如使用物理
0 卡作为 P、物理 8 卡作为 D：

```bash
cd /workspace/w50062541/code/vllm-ascend

python -X faulthandler -m \
  tests.ut.distributed.kv_transfer.a3_2.test_memfabric_bm_same_host_pd_gather_npu \
  --prefill-physical-device 0 \
  --decode-physical-device 8 \
  --protocol host_shm
```

只有输出包含以下内容才判定 G2a 为 Go：

```text
MemFabric BM same-host P -> D Host Full-KV -> Gather G2a PASSED
```

也可以把协议切成 `host_rdma` 做同机 HCOM 兼容性 smoke；运行前需要把 wheel
内的 `memfabric_hybrid/lib` 加入 `LD_LIBRARY_PATH`。即使该 smoke 通过，也不能
替代 G2b 的真实跨机 `HOST_RDMA` 验收。

## 选择生产传输路径

`config.env` 中的 `SPARSE_KV_TRANSFER_MODE` 控制 Full KV 的逐层 P/D 路径：

```bash
# 已验证基线，也是缺省值
SPARSE_KV_TRANSFER_MODE=npu_staging

# 新的 Host relay 对照路径
SPARSE_KV_TRANSFER_MODE=host_relay
```

`npu_staging` 保持原路径：Full KV 与 Indexer 都通过 Mooncake Ascend transport
进入 Decode 普通 NPU staging，再通过本地 NPU→swapped copy 把 Full KV 持久化到
Decode framework swapped Full KV。

`host_relay` 当前是默认关闭的兼容性/诊断模式，使用两套共存的 Mooncake engine：

```text
Full KV: P 普通 NPU staging --本地 NPU→pinned copy--> P pinned Host relay
         --Mooncake TCP--> D pinned Host relay
         --本地 pinned→NPU copy--> D 普通 NPU staging
         --本地 NPU→swapped copy--> D framework swapped Full KV
Indexer: P Indexer NPU KV --Mooncake Ascend NPU→NPU--> D Indexer NPU KV
```

Decode 只有在 Host relay 已桥接到 swapped Full KV、Indexer 也已传完后才确认该层。
这个实现仍让 Full KV 经过 Decode NPU，不是 mentor 所指的最终 Host 路径，当前
不得用它得出性能收益结论。`run.sh` 会主动清除外部 `MC_FORCE_TCP`；不要手工导出它，否则现有
Ascend engine 可能被错误初始化成 TCP。两种模式使用带模式名的独立日志和结果
文件，避免覆盖对照证据。

使用三个终端，按顺序启动：

```bash
# 终端一
bash run.sh decode

# 终端二；先等待 Decode ready
bash run.sh prefill

# 终端三；先等待 Prefill ready
bash run.sh proxy
```

可以从第四个终端检查服务：

```bash
bash run.sh ready decode
bash run.sh ready prefill
bash run.sh ready proxy
```

## 正确性验收与证据归档

服务全部 ready 后，在第四个终端执行：

```bash
bash run.sh validate
bash run.sh collect
```

`validate` 包含 Top-K 阈值以下、阈值以上、约 3K、约 3.6K 和请求状态重置
五个用例，并检查每个请求的 P/D rank 生命周期及目标致命错误签名。
计时仅用于发现异常卡顿，不构成性能结论。

`collect` 将日志、验收 JSON、Git revision、运行时版本、模型元数据哈希、
NPU 状态和传输生命周期归档到 `OUTPUT_DIR` 下的带时间戳目录。

通过后，在 Proxy、Prefill、Decode 三个服务终端依次按 `Ctrl+C`。不要在共享服务器
使用会影响同事 Ray/Python 进程的宽泛 `pkill`。

## 当前边界

- 同节点正确性已验证；跨节点仍需独立验证。
- 当前只支持单请求、Eager、`block_size=128`、BF16/FP16 KV。
- 不支持 Prefix Cache、Sparse C8、MTP/Speculative Decode 或 DSA CP/PCP/DCP。
- 逐层 staging 采用同步 ACK；尚未证明通信计算重叠或性能收益。
- `host_relay` 只是默认关闭的兼容性诊断路径，不是性能候选；它使用 TCP 验证
  pinned Host relay 传输，尚未证明跨节点 RDMA/RoCE/UB Host transport，也不使用
  Mooncake Store。
- 如果端口被 Ray 等共享服务占用，应修改 `config.env` 选择完整空闲端口段，
  不要终止不属于本任务的进程。

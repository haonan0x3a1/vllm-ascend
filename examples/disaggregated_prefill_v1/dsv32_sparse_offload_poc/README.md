# DeepSeek-V3.2 Sparse KV Offload Online P/D PoC

本目录把同节点 DeepSeek-V3.2 Sparse Host KV Offload 的 Online P/D
启动、预检、验收和证据归档固化为一个入口。它面向正确性 PoC，
不是性能或生产部署脚本。

已验证基线：

- 真实 DeepSeek-V3.2 W4A8 权重，运行时 BF16 KV；
- 61 个主模型层，TP8 Prefill + TP8 Decode，Expert Parallel；
- `MooncakeLayerwiseConnector` 逐层 NPU staging；
- Host Full KV、Lightning Indexer Top-K、CANN Gather 和 SFA；
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

当前已验证基线使用 Mooncake Ascend transport 将逐层缓存传入 Decode NPU
staging。为了评估 Full KV 改走 `P Host DDR -> D Host DDR` 是否能减少对 Decode
NPU 的干扰，先单独验证当前 Ascend Mooncake wheel 的 TCP Host transport：

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

这个测试通过也只确定“双引擎可共存”的实现前提；下一门槛仍是生产内存链路
`P NPU -> P pinned Host -> D pinned Host -> D swapped Full KV -> Gather`。在该链路通过
以前，不应把 Host relay 接入生产 Connector。

双引擎门槛通过后，运行最后一个独立硬件探针：

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
`LOG_DIR/dsv32-mooncake-host-relay-probe.log`。这是 Host relay 接入生产 Connector 前
的最后一个独立探针；通过后不再增加微型门槛，直接进入 Connector 集成。

## 选择生产传输路径

`config.env` 中的 `SPARSE_KV_TRANSFER_MODE` 控制 Full KV 的逐层 P/D 路径：

```bash
# 已验证基线，也是缺省值
SPARSE_KV_TRANSFER_MODE=npu_staging

# 新的 Host relay 对照路径
SPARSE_KV_TRANSFER_MODE=host_relay
```

`npu_staging` 保持原路径：Full KV 与 Indexer 都通过 Mooncake Ascend transport
进入 Decode NPU staging，再把 Full KV 持久化到 Decode swapped Host cache。

`host_relay` 使用两套共存的 Mooncake engine：

```text
Full KV: P NPU staging -> P pinned Host -> Mooncake TCP
         -> D pinned Host -> D NPU staging -> D swapped Full KV
Indexer: P NPU ---------------- Mooncake Ascend ----------------> D NPU
```

Decode 只有在 Host relay 已桥接到 swapped Full KV、Indexer 也已传完后才确认该层。
两个模式复用相同的请求、block 映射、Gather 和 SFA 路径，便于后续做正确性及
性能 A/B。`run.sh` 会主动清除外部 `MC_FORCE_TCP`；不要手工导出它，否则现有
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
- `host_relay` 当前使用 TCP 作为功能和性能基线；尚未证明跨节点 RDMA/RoCE/UB
  Host transport，也不使用 Mooncake Store。
- 如果端口被 Ray 等共享服务占用，应修改 `config.env` 选择完整空闲端口段，
  不要终止不属于本任务的进程。

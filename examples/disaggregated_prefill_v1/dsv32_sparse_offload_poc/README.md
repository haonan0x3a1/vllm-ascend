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
relay。通过后还要依次验证本地 NPU-to-Host、Host-to-swapped/Gather，以及 Full KV
Host transport 与 Indexer NPU transport 的组合生命周期；任一步失败都不应改动
当前默认的 NPU staging 路径。

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
- 如果端口被 Ray 等共享服务占用，应修改 `config.env` 选择完整空闲端口段，
  不要终止不属于本任务的进程。

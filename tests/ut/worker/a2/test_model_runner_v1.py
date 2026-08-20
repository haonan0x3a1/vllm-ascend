import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_sparse_offload_populates_cpu_slot_mapping(self):
        runner = self._build_runner()
        runner.ascend_config = SimpleNamespace(sparse_kv_offload=SimpleNamespace(enabled=True))
        runner.input_batch = SimpleNamespace(block_table=MagicMock())
        req_indices = np.array([0, 0], dtype=np.int64)
        positions = np.array([0, 1], dtype=np.int64)

        runner._compute_sparse_kv_offload_slot_mapping_cpu(
            req_indices,
            positions,
        )

        runner.input_batch.block_table.compute_slot_mapping_cpu.assert_called_once_with(
            req_indices,
            positions,
        )

    def test_disabled_sparse_offload_leaves_cpu_slot_mapping_untouched(self):
        runner = self._build_runner()
        runner.ascend_config = SimpleNamespace(sparse_kv_offload=SimpleNamespace(enabled=False))
        runner.input_batch = SimpleNamespace(block_table=MagicMock())

        runner._compute_sparse_kv_offload_slot_mapping_cpu(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )

        runner.input_batch.block_table.compute_slot_mapping_cpu.assert_not_called()

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))

    @patch("vllm_ascend.worker.model_runner_v1.torch_npu.empty_with_swapped_memory")
    def test_allocate_swapped_cache_uses_torch_npu_allocator(
        self,
        mock_empty_with_swapped_memory,
    ):
        runner = self._build_runner()
        mock_empty_with_swapped_memory.return_value = torch.empty(
            1024,
            dtype=torch.int8,
        )

        result = runner._allocate_swapped_int8_cache_tensor(1024)

        self.assertEqual(result.numel(), 1024)
        mock_empty_with_swapped_memory.assert_called_once_with(
            (1024,),
            dtype=torch.int8,
            device=runner.device,
        )

    def test_memfabric_bm_mode_is_selected_from_connector_config(self):
        runner = self._build_runner()
        runner.vllm_config.kv_transfer_config = MagicMock()
        runner.vllm_config.kv_transfer_config.get_from_extra_config.return_value = (
            "memfabric_bm"
        )

        self.assertTrue(runner._uses_memfabric_bm_full_kv())

    @patch("vllm_ascend.worker.model_runner_v1.torch.npu.current_device")
    @patch("vllm_ascend.worker.model_runner_v1.get_tp_group")
    @patch(
        "vllm_ascend.distributed.kv_transfer.utils.memfabric_bm_allocator."
        "MemFabricBMRuntimeConfig"
    )
    @patch(
        "vllm_ascend.distributed.kv_transfer.utils.memfabric_bm_allocator."
        "MemFabricBMFullKVAllocator"
    )
    def test_allocate_memfabric_cache_reuses_one_worker_pool(
        self,
        mock_allocator_cls,
        mock_runtime_config_cls,
        mock_get_tp_group,
        mock_current_device,
    ):
        runner = self._build_runner()
        runner.vllm_config.kv_transfer_config = MagicMock()
        mock_get_tp_group.return_value.rank_in_group = 3
        mock_current_device.return_value = 3
        allocator = mock_allocator_cls.return_value
        allocator.allocate_int8.side_effect = [
            torch.empty(16, dtype=torch.int8),
            torch.empty(8, dtype=torch.int8),
        ]

        first = runner._allocate_memfabric_bm_int8_cache_tensor(16)
        second = runner._allocate_memfabric_bm_int8_cache_tensor(8)

        self.assertEqual(first.numel(), 16)
        self.assertEqual(second.numel(), 8)
        mock_allocator_cls.assert_called_once_with(
            mock_runtime_config_cls.from_kv_transfer_config.return_value
        )
        mock_runtime_config_cls.from_kv_transfer_config.assert_called_once_with(
            runner.vllm_config.kv_transfer_config,
            tp_rank=3,
            device_id=3,
        )
        self.assertEqual(allocator.allocate_int8.call_count, 2)

    @patch("vllm.v1.worker.gpu_model_runner.GPUModelRunner.shutdown")
    def test_shutdown_releases_parent_tensors_before_memfabric_pool(
        self,
        mock_parent_shutdown,
    ):
        runner = self._build_runner()
        allocator = MagicMock()
        runner._memfabric_bm_full_kv_allocator = allocator
        order = []
        mock_parent_shutdown.side_effect = lambda: order.append("parent")
        allocator.close.side_effect = lambda: order.append("allocator")

        runner.shutdown()

        self.assertEqual(order, ["parent", "allocator"])
        self.assertFalse(hasattr(runner, "_memfabric_bm_full_kv_allocator"))

    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_bind_host_mode_shares_one_prefill_cache_across_sparse_layers(
        self,
        mock_get_layers,
    ):
        runner = self._build_runner()
        runner.ascend_config = SimpleNamespace(
            sparse_kv_offload=SimpleNamespace(
                enabled=True,
                mode="host",
            )
        )

        first_layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(first_layer)
        first_layer.impl = MagicMock()
        second_layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(second_layer)
        second_layer.impl = MagicMock()
        first_name = "model.layers.0.self_attn.attn"
        second_name = "model.layers.1.self_attn.attn"
        mock_get_layers.return_value = {
            first_name: first_layer,
            second_name: second_layer,
        }
        kv_caches = {
            first_name: (
                torch.empty(4, 128, 1, 2),
                torch.empty(4, 128, 1, 1),
                torch.empty(4, 128, 1, 4),
            ),
            second_name: (
                torch.empty(4, 128, 1, 2),
                torch.empty(4, 128, 1, 1),
                torch.empty(4, 128, 1, 4),
            ),
        }

        runner._bind_sparse_kv_offload_prefill_cache(kv_caches)

        first_prefill = first_layer.impl.set_sparse_kv_offload_prefill_cache.call_args.args[0]
        second_prefill = second_layer.impl.set_sparse_kv_offload_prefill_cache.call_args.args[0]
        self.assertIs(first_prefill, second_prefill)
        self.assertEqual(first_prefill[0].shape, kv_caches[first_name][0].shape)
        self.assertEqual(first_prefill[1].shape, kv_caches[first_name][1].shape)
        first_layer.impl.initialize_sparse_kv_offload_workspace.assert_called_once_with(kv_caches[first_name])
        second_layer.impl.initialize_sparse_kv_offload_workspace.assert_called_once_with(kv_caches[second_name])

    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_host_mode_memory_plan_uses_local_sparse_layer_layout(
        self,
        mock_get_layers,
    ):
        runner = self._build_runner()
        runner.ascend_config = SimpleNamespace(
            sparse_kv_offload=SimpleNamespace(
                enabled=True,
                mode="host",
            )
        )
        runner.block_size = 128
        runner.kv_cache_dtype = torch.bfloat16
        runner.vllm_config.kv_transfer_config = MagicMock()
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            index_topk=2048,
        )

        first_layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(first_layer)
        first_layer.impl = SimpleNamespace(has_indexer=True)
        second_layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(second_layer)
        second_layer.impl = SimpleNamespace(has_indexer=False)
        first_name = "model.layers.0.self_attn.attn"
        second_name = "model.layers.1.self_attn.attn"
        mock_get_layers.return_value = {
            first_name: first_layer,
            second_name: second_layer,
        }
        kv_cache_config = SimpleNamespace(
            num_blocks=5,
            kv_cache_groups=[
                SimpleNamespace(layer_names=[first_name, second_name]),
            ],
        )

        plan = runner.get_sparse_kv_offload_memory_plan(kv_cache_config)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.indexer_cache_bytes, 5 * 128 * 128 * 2)
        self.assertEqual(plan.indexer_alignment_bytes, 2 * 1024 * 1024)
        self.assertEqual(plan.shared_prefill_bytes, 5 * 128 * 576 * 2)
        self.assertEqual(
            plan.shared_prefill_alignment_bytes,
            2 * 1024 * 1024,
        )
        self.assertEqual(plan.selected_cache_bytes, 2 * 2048 * 576 * 2)
        runner.validate_sparse_kv_offload_memory(
            kv_cache_config,
            plan.total_bytes,
        )
        with self.assertRaisesRegex(
            ValueError,
            "requires .* bytes of persistent NPU memory",
        ):
            runner.validate_sparse_kv_offload_memory(
                kv_cache_config,
                plan.total_bytes - 1,
            )

    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_reset_host_mode_selection_state_for_sparse_mla_layers(
        self,
        mock_get_layers,
    ):
        runner = self._build_runner()
        runner.ascend_config = SimpleNamespace(
            sparse_kv_offload=SimpleNamespace(
                enabled=True,
                mode="host",
            )
        )

        sparse_layer = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(sparse_layer)
        sparse_layer.impl = MagicMock()
        mock_get_layers.return_value = {
            "model.layers.0.self_attn.attn": sparse_layer,
        }

        runner._reset_sparse_kv_offload_selection_state()

        resetter = sparse_layer.impl.reset_sparse_kv_offload_selection_state
        resetter.assert_called_once_with()

    @patch.object(NPUModelRunner, "_reset_sparse_kv_offload_selection_state")
    @patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        return_value=None,
    )
    def test_update_states_resets_selection_for_new_pd_request(
        self,
        mock_super_update_states,
        mock_reset_selection,
    ):
        runner = self._build_runner()
        runner.use_async_scheduling = False
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id="request-0")],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        )

        runner._update_states(scheduler_output)

        mock_super_update_states.assert_called_once_with(scheduler_output)
        mock_reset_selection.assert_called_once_with()

    @patch.object(NPUModelRunner, "_reset_sparse_kv_offload_selection_state")
    @patch(
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states",
        return_value=None,
    )
    def test_update_states_preserves_selection_for_continuing_decode(
        self,
        mock_super_update_states,
        mock_reset_selection,
    ):
        runner = self._build_runner()
        runner.use_async_scheduling = False
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["request-0"]),
        )

        runner._update_states(scheduler_output)

        mock_super_update_states.assert_called_once_with(scheduler_output)
        mock_reset_selection.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_sparse_layer_without_indexer_allocates_only_mla_kv_cache(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.sparse_head_dim = (512, 64, 128)
        runner.kv_cache_dtype = torch.bfloat16
        runner.shared_kv_cache_layers = {}
        runner.ascend_config = MagicMock()
        runner.ascend_config.is_sparse_c8_layer.return_value = False
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(has_indexer=False)
        layer_name = "model.layers.1.self_attn.attn"
        mock_get_layers.return_value = {layer_name: attn_module}

        spec = runner.get_kv_cache_spec()[layer_name]
        self.assertEqual(spec.sparse_head_dim, (512, 64, 0))

        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes * 2,
                    shared_by=[layer_name],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                )
            ],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
        raw_k_cache, raw_v_cache = raw_caches[layer_name]

        self.assertEqual(raw_k_cache.numel(), 2 * 16 * 512 * 2)
        self.assertEqual(raw_v_cache.numel(), 2 * 16 * 64 * 2)


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        runner.use_compress = False
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.get_ascend_config")
    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable, mock_get_ascend_config):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_reduce_sample = False
        mock_get_ascend_config.return_value = mock_ascend_config

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.sampling_metadata.top_k = None
        input_batch.num_reqs = 2
        input_batch.top_k_cpu = None
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])

    def test_placeholder_spec_tokens_are_sanitized_only_for_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, 33, -1])

    def test_placeholder_sanitization_is_scoped_to_current_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=2,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, -1])

    def test_mtp3_placeholder_metadata_is_preserved_before_sanitizing_forward(self):
        runner = self._build_runner()
        runner.pcp_size = 1
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner._arange_scratch = np.empty(8, dtype=np.int32)
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1, -1, -1]},
        )

        spec_decode_metadata = runner._calc_spec_decode_metadata(
            num_draft_tokens=np.array([3], dtype=np.int32),
            cu_num_scheduled_tokens=np.array([4], dtype=np.int32),
            num_pcp_pads=None,
        )
        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(spec_decode_metadata.draft_token_ids.tolist(), [-1, -1, -1])
        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 0, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, -1, -1])


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        runner.use_compress = False
        return runner

    def test_finalize_dump_data_stops_stop_capable_debugger(self):
        runner = self._build_runner()

        runner._finalize_dump_data()

        runner.debugger.stop.assert_called_once_with()
        runner.debugger.step.assert_called_once_with()
        self.assertFalse(runner._debugger_started)

    def test_finalize_dump_data_steps_graph_debugger_without_stop(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)

        runner._finalize_dump_data()

        debugger.step.assert_called_once_with()
        self.assertTrue(runner._debugger_started)

    def test_start_dump_data_noop_when_already_started(self):
        runner = self._build_runner(MagicMock(spec=["start", "step"]))

        runner._start_dump_data()

        runner.debugger.start.assert_not_called()
        runner.debugger.step.assert_not_called()
        self.assertTrue(runner._debugger_started)


class TestCorrectOptimisticSeqLensCpu(unittest.TestCase):
    """Regression tests for async spec-decode seq_lens correction.

    The helper must synchronize the device->host copy event *before* reading
    ``valid_sampled_token_count_cpu``. Reading it early consumes stale counts
    and corrupts the CPU seq_lens, which surfaced as an accuracy regression on
    DeepSeek-V4 (its compressed-KV slot mapping is built from these seq_lens).
    """

    def _build_runner(self, optimistic, prev_positions, prev_drafts, counts_cpu):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.optimistic_seq_lens_cpu = optimistic
        runner.prev_positions = SimpleNamespace(np=prev_positions)
        runner.prev_num_draft_tokens = SimpleNamespace(np=prev_drafts)
        runner.valid_sampled_token_count_cpu = counts_cpu
        return runner

    def test_synchronizes_before_host_read(self):
        num_reqs = 3
        # Optimistic (all drafts assumed accepted):
        #   prev_computed=[100,200,50], prev_drafts=[2,3,1], sched=[3,4,2]
        #   optimistic = prev_computed + (prev_drafts + 1) + sched
        optimistic = torch.tensor([106, 208, 54], dtype=torch.int64)
        prev_positions = np.array([0, 1, 2], dtype=np.int64)
        prev_drafts = np.array([2, 3, 1], dtype=np.int32)

        # CPU buffer initially holds STALE counts (== drafts + 1, i.e. "all
        # accepted"). If the helper reads before synchronizing, the correction
        # is a no-op and the assertion below fails.
        counts_cpu = torch.tensor([3, 4, 2], dtype=torch.int32)
        # The true counts that the async copy delivers on synchronize().
        true_counts = np.array([2, 1, 2], dtype=np.int32)

        runner = self._build_runner(optimistic, prev_positions, prev_drafts, counts_cpu)
        event = MagicMock()
        event.synchronize.side_effect = lambda: counts_cpu.copy_(torch.from_numpy(true_counts))
        runner.valid_sampled_token_count_event = event

        runner._correct_optimistic_seq_lens_cpu(num_reqs)

        event.synchronize.assert_called_once()
        # correction = (prev_drafts + 1 - true_counts) = [1, 3, 0]
        # corrected  = optimistic - correction          = [105, 205, 54]
        np.testing.assert_array_equal(optimistic.numpy(), np.array([105, 205, 54]))

    def test_asserts_event_present(self):
        runner = self._build_runner(
            torch.tensor([10], dtype=torch.int64),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int32),
            torch.tensor([1], dtype=torch.int32),
        )
        runner.valid_sampled_token_count_event = None
        with self.assertRaises(AssertionError):
            runner._correct_optimistic_seq_lens_cpu(1)


if __name__ == "__main__":
    unittest.main()

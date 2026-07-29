#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import KVTransferConfig, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (
    SparseKVOffloadConfig,
    clear_ascend_config,
    get_ascend_config,
    init_ascend_config,
)
from vllm_ascend.utils import clear_enable_sp, enable_sp, get_flashcomm2_config_and_validate


class TestAscendConfig(TestBase):
    @staticmethod
    def _clean_up_ascend_config(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            clear_enable_sp()
            try:
                func(*args, **kwargs)
            finally:
                clear_ascend_config()
                clear_enable_sp()

        return wrapper

    @staticmethod
    def _make_model_config(
        total_num_attention_heads: int = 32,
        total_num_kv_heads: int = 8,
        is_deepseek_mla: bool = False,
    ):
        return SimpleNamespace(
            is_deepseek_mla=is_deepseek_mla,
            use_mla=is_deepseek_mla,
            enforce_eager=True,
            model_arch_config=SimpleNamespace(total_num_attention_heads=total_num_attention_heads),
            get_total_num_kv_heads=lambda: total_num_kv_heads,
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertFalse(ascend_config.sparse_kv_offload.enabled)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_with_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": False,
            },
            "multistream_overlap_shared_expert": True,
            "eplb_config": {"num_redundant_experts": 2},
            "refresh": True,
            "enable_kv_nz": False,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.eplb_config.num_redundant_experts, 2)
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertFalse(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_enable_npugraph_ex(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": True},
            "refresh": True,
        }
        ascend_compilation_config = init_ascend_config(test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertTrue(ascend_compilation_config.enable_static_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_mooncake_c8_kv_cache_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_consumer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_multi_connector_mooncake_c8_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_role": "kv_consumer",
                    }
                ]
            },
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_allows_layerwise_c8_kv_cache_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeLayerwiseConnector",
            kv_role="kv_consumer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertIsNotNone(ascend_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_allows_mha_mooncake_c8_kv_cache_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_consumer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config(
            total_num_attention_heads=8,
            total_num_kv_heads=8,
        )

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertIsNotNone(ascend_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_mooncake_c8_kv_cache_producer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_producer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_mooncake_c8_kv_cache_both_role(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_both",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_falls_back_to_envs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.parallel_config.tensor_parallel_size = 4
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "1",
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "2",
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
                "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "2",
                "MSMONITOR_USE_DAEMON": "1",
                "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": "0",
                "VLLM_ASCEND_ENABLE_NZ": "2",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.enable_matmul_allreduce)
        self.assertEqual(ascend_config.enable_fused_mc2, 2)
        self.assertFalse(ascend_config.enable_mlapo)
        self.assertTrue(ascend_config.enable_flashcomm1)
        self.assertEqual(ascend_config.enable_flashcomm2_parallel_size, 2)
        self.assertTrue(ascend_config.msmonitor_use_daemon)
        self.assertFalse(ascend_config.enable_transpose_kv_cache_by_block)
        self.assertEqual(ascend_config.weight_nz_mode, 2)
        mock_info_once.assert_any_call(
            "AscendConfig.enable_mlapo falls back to environment variable VLLM_ASCEND_ENABLE_MLAPO with value False. "
            "Please use additional_config.enable_mlapo instead, because VLLM_ASCEND_ENABLE_MLAPO will be "
            "removed in the next release."
        )
        mock_info_once.assert_any_call(
            "AscendConfig.weight_nz_mode falls back to environment variable VLLM_ASCEND_ENABLE_NZ with value 2. "
            "Please use additional_config.weight_nz_mode instead, because VLLM_ASCEND_ENABLE_NZ will be removed "
            "in the next release."
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_skips_default_env_fallback_logs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        with patch.dict(os.environ, {}, clear=True):
            init_ascend_config(test_vllm_config)

        fallback_logs = [
            call.args[0]
            for call in mock_info_once.call_args_list
            if "falls back to environment variable" in call.args[0]
        ]
        self.assertEqual(fallback_logs, [])

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_overrides_envs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "enable_matmul_allreduce": False,
            "enable_fused_mc2": 0,
            "enable_mlapo": True,
            "enable_flashcomm1": False,
            "enable_flashcomm2_parallel_size": 0,
            "msmonitor_use_daemon": False,
            "enable_transpose_kv_cache_by_block": True,
            "weight_nz_mode": 1,
        }
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "1",
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "2",
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
                "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "2",
                "MSMONITOR_USE_DAEMON": "1",
                "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": "0",
                "VLLM_ASCEND_ENABLE_NZ": "2",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertFalse(ascend_config.enable_matmul_allreduce)
        self.assertEqual(ascend_config.enable_fused_mc2, 0)
        self.assertTrue(ascend_config.enable_mlapo)
        self.assertFalse(ascend_config.enable_flashcomm1)
        self.assertEqual(ascend_config.enable_flashcomm2_parallel_size, 0)
        self.assertFalse(ascend_config.msmonitor_use_daemon)
        self.assertTrue(ascend_config.enable_transpose_kv_cache_by_block)
        self.assertEqual(ascend_config.weight_nz_mode, 1)
        mock_info_once.assert_any_call("AscendConfig.enable_mlapo is set from additional_config with value True.")
        mock_info_once.assert_any_call("AscendConfig.weight_nz_mode is set from additional_config with value 1.")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}, clear=True)
    def test_enable_flashcomm1_config_overrides_disabled_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"enable_flashcomm1": True}
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)
        self.assertTrue(ascend_config.enable_flashcomm1)
        self.assertTrue(enable_sp(test_vllm_config))

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_sp_falls_back_to_env_without_current_config(self, mock_check_and_update_config):
        clear_enable_sp()
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}),
            patch("vllm.config.get_current_vllm_config", side_effect=AssertionError),
        ):
            self.assertTrue(enable_sp())

    @_clean_up_ascend_config
    @patch("vllm_ascend.utils.logger.warning_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_flashcomm2_warning_uses_enable_flashcomm1_config(self, mock_check_and_update_config, mock_warning_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.parallel_config.tensor_parallel_size = 4
        test_vllm_config.kv_transfer_config = None
        ascend_config = type(
            "MockAscendConfig",
            (),
            {
                "enable_flashcomm2_parallel_size": 2,
                "layer_sharding": None,
                "enable_flashcomm1": True,
                "finegrained_tp_config": type("MockFinegrainedTPConfig", (), {"oproj_tensor_parallel_size": 0})(),
            },
        )()

        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}):
            self.assertEqual(get_flashcomm2_config_and_validate(ascend_config, test_vllm_config), 2)

        flashcomm1_warning = (
            "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
        )
        self.assertNotIn(flashcomm1_warning, [call.args[0] for call in mock_warning_once.call_args_list])

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_get_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_clear_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_with_dump_config_materializes_fixed_file(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        dump_config = {"task": "tensor", "level": "L1", "dump_path": "/tmp/msprobe_dump"}
        test_vllm_config.additional_config = {"dump_config": dump_config}

        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNotNone(ascend_config.dump_config_path)
        assert ascend_config.dump_config_path is not None
        expected_path = os.path.join(os.getcwd(), ".vllm_ascend", "msprobe", "msprobe_dump_config.json")
        self.assertEqual(ascend_config.dump_config_path, expected_path)
        self.assertTrue(os.path.exists(ascend_config.dump_config_path))
        with open(ascend_config.dump_config_path, encoding="utf-8") as file:
            persisted = json.load(file)
        self.assertEqual(persisted, dump_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_and_path_conflict(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"dump_config_path": "/tmp/config.json", "dump_config": {"task": "tensor"}}
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_type_validation(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"dump_config": "/tmp/config.json"}
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_recreates_for_new_vllm_config(self, mock_fix_incompatible_config):
        first_vllm_config = VllmConfig()
        first_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": False,
            }
        }
        first_ascend_config = init_ascend_config(first_vllm_config)
        self.assertFalse(first_ascend_config.ascend_compilation_config.enable_npugraph_ex)

        second_vllm_config = VllmConfig()
        second_ascend_config = init_ascend_config(second_vllm_config)
        self.assertIsNot(first_ascend_config, second_ascend_config)
        self.assertTrue(second_ascend_config.ascend_compilation_config.enable_npugraph_ex)


class TestSparseKVOffloadConfig(TestBase):
    @staticmethod
    def _make_vllm_config():
        return SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    index_topk=2048,
                    model_type="deepseek_v32",
                ),
                enforce_eager=True,
                max_model_len=4096,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            cache_config=SimpleNamespace(
                block_size=128,
                enable_prefix_caching=False,
                num_gpu_blocks_override=None,
            ),
            speculative_config=None,
            kv_transfer_config=None,
            additional_config={},
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
            ),
        )

    def test_parse_defaults_and_mirror_mode(self):
        self.assertEqual(SparseKVOffloadConfig.from_dict(None), SparseKVOffloadConfig())
        self.assertEqual(
            SparseKVOffloadConfig.from_dict(
                {
                    "enabled": True,
                    "mode": "mirror",
                }
            ),
            SparseKVOffloadConfig(enabled=True, mode="mirror"),
        )
        self.assertEqual(
            SparseKVOffloadConfig.from_dict(
                {
                    "enabled": True,
                    "mode": "host",
                }
            ),
            SparseKVOffloadConfig(enabled=True, mode="host"),
        )

    def test_parse_rejects_invalid_values(self):
        invalid_configs = [
            (True, "must be a dict"),
            ({"enabled": 1}, "enabled must be a bool"),
            ({"mode": 1}, "mode must be a string"),
            ({"mode": "unsupported"}, "must be one of"),
            ({"unknown": True}, "unsupported keys"),
        ]
        for raw_config, expected_error in invalid_configs:
            with self.subTest(raw_config=raw_config), self.assertRaisesRegex(ValueError, expected_error):
                SparseKVOffloadConfig.from_dict(raw_config)

    def test_validate_accepts_single_request_eager_mirror(self):
        config = SparseKVOffloadConfig(enabled=True)
        config.validate(self._make_vllm_config(), enable_sparse_c8=False)

    def test_validate_host_sets_capacity_override(self):
        vllm_config = self._make_vllm_config()
        config = SparseKVOffloadConfig(enabled=True, mode="host")

        config.validate(vllm_config, enable_sparse_c8=False)

        self.assertEqual(vllm_config.cache_config.num_gpu_blocks_override, 33)

    def test_validate_host_accepts_layerwise_memcache_pd(self):
        vllm_config = self._make_vllm_config()
        vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector="AscendStoreConnector",
            kv_connector_extra_config={
                "backend": "memcache",
                "use_layerwise": True,
            },
            kv_role="kv_producer",
        )

        SparseKVOffloadConfig(enabled=True, mode="host").validate(
            vllm_config,
            enable_sparse_c8=False,
        )

    def test_validate_host_rejects_unsupported_pd_and_capacity(self):
        cases = [
            (
                SimpleNamespace(
                    kv_connector="MooncakeLayerwiseConnector",
                    kv_connector_extra_config={},
                    kv_role="kv_producer",
                ),
                "requires AscendStoreConnector",
            ),
            (
                SimpleNamespace(
                    kv_connector="AscendStoreConnector",
                    kv_connector_extra_config={
                        "backend": "memcache",
                        "use_layerwise": False,
                    },
                    kv_role="kv_producer",
                ),
                "use_layerwise=true",
            ),
            (
                SimpleNamespace(
                    kv_connector="AscendStoreConnector",
                    kv_connector_extra_config={
                        "backend": "mooncake",
                        "use_layerwise": True,
                    },
                    kv_role="kv_producer",
                ),
                "backend='memcache'",
            ),
            (
                SimpleNamespace(
                    kv_connector="AscendStoreConnector",
                    kv_connector_extra_config={
                        "backend": "memcache",
                        "use_layerwise": True,
                    },
                    kv_role="kv_both",
                ),
                "kv_producer/kv_consumer",
            ),
        ]
        config = SparseKVOffloadConfig(enabled=True, mode="host")
        for kv_transfer_config, expected_error in cases:
            vllm_config = self._make_vllm_config()
            vllm_config.kv_transfer_config = kv_transfer_config
            with (
                self.subTest(expected_error=expected_error),
                self.assertRaisesRegex(ValueError, expected_error),
            ):
                config.validate(vllm_config, enable_sparse_c8=False)

        vllm_config = self._make_vllm_config()
        vllm_config.cache_config.num_gpu_blocks_override = 32
        with self.assertRaisesRegex(ValueError, "required=33"):
            config.validate(vllm_config, enable_sparse_c8=False)

    def test_validate_rejects_unsupported_runtime_combinations(self):
        cases = [
            (
                lambda cfg: setattr(
                    cfg.model_config.hf_text_config,
                    "model_type",
                    "glm_moe_dsa",
                ),
                "supports only DeepSeek-V3.2",
            ),
            (
                lambda cfg: setattr(cfg.model_config, "enforce_eager", False),
                "requires --enforce-eager",
            ),
            (
                lambda cfg: setattr(cfg.scheduler_config, "max_num_seqs", 2),
                "requires --max-num-seqs 1",
            ),
            (
                lambda cfg: setattr(cfg.cache_config, "block_size", 16),
                "requires KV cache block_size=128",
            ),
            (
                lambda cfg: setattr(cfg.cache_config, "enable_prefix_caching", True),
                "does not support prefix caching",
            ),
            (
                lambda cfg: setattr(cfg, "speculative_config", object()),
                "does not support speculative decoding",
            ),
            (
                lambda cfg: setattr(cfg, "kv_transfer_config", object()),
                "cannot be combined with KV transfer",
            ),
            (
                lambda cfg: cfg.additional_config.update({"enable_dsa_cp": True}),
                "does not support DSA context parallelism",
            ),
            (
                lambda cfg: setattr(
                    cfg.parallel_config,
                    "decode_context_parallel_size",
                    2,
                ),
                "does not support DSA context parallelism",
            ),
        ]
        config = SparseKVOffloadConfig(enabled=True)
        for mutate, expected_error in cases:
            vllm_config = self._make_vllm_config()
            mutate(vllm_config)
            with (
                self.subTest(expected_error=expected_error),
                self.assertRaisesRegex(
                    ValueError,
                    expected_error,
                ),
            ):
                config.validate(vllm_config, enable_sparse_c8=False)

        with self.assertRaisesRegex(ValueError, "does not support sparse C8"):
            config.validate(self._make_vllm_config(), enable_sparse_c8=True)

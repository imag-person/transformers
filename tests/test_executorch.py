# Copyright 2025 HuggingFace Inc.
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

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers import AutoModelForCausalLM, set_seed
from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.executorch import (
    TorchExportableModuleForDecoderOnlyLM,
    TorchExportableModuleWithHybridCache,
    TorchExportableModuleWithStaticCache,
    convert_and_export_with_cache,
)
from transformers.testing_utils import require_torch


@require_torch
class ExecutorchModalityInputsTest(unittest.TestCase):
    def test_static_cache_module_forward_with_modality_inputs(self):
        """Test TorchExportableModuleWithStaticCache forwards optional image and audio inputs."""

        class DummyModel:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(logits=torch.ones(1, 1, 2))

        module = TorchExportableModuleWithStaticCache.__new__(TorchExportableModuleWithStaticCache)
        torch.nn.Module.__init__(module)
        module.model = DummyModel()
        module.static_cache = SimpleNamespace(
            layers=[SimpleNamespace(cumulative_length=torch.zeros(1, dtype=torch.long))]
        )

        pixel_values = torch.zeros(1, 3, 16, 16)
        input_features = torch.zeros(1, 80, 8)
        output = module.forward(
            input_ids=torch.tensor([[1]]),
            pixel_values=pixel_values,
            input_features=input_features,
            cache_position=torch.tensor([0]),
        )

        self.assertEqual(output.shape, (1, 1, 2))
        self.assertIs(module.model.kwargs["pixel_values"], pixel_values)
        self.assertIs(module.model.kwargs["input_features"], input_features)

    def test_convert_and_export_with_cache_uses_modality_inputs(self):
        """Test convert_and_export_with_cache adds inferred and explicit modality examples to export kwargs."""

        from transformers.integrations import executorch as executorch_module

        model = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(image_size=16, num_channels=3),
                num_mel_bins=80,
                max_source_positions=4,
            ),
            device=torch.device("cpu"),
        )

        with (
            patch.object(executorch_module, "TorchExportableModuleWithStaticCache", return_value=object()),
            patch.object(executorch_module, "is_torch_greater_or_equal", return_value=True),
            patch.object(executorch_module.torch.export, "export", return_value="exported") as export_mock,
        ):
            self.assertEqual(convert_and_export_with_cache(model), "exported")
            kwargs = export_mock.call_args.kwargs["kwargs"]
            self.assertEqual(kwargs["pixel_values"].shape, (1, 3, 16, 16))
            self.assertEqual(kwargs["input_features"].shape, (1, 80, 8))

            explicit_pixel_values = torch.ones(1, 3, 8, 8)
            convert_and_export_with_cache(
                model,
                example_modality_inputs={"pixel_values": explicit_pixel_values, "input_features": None},
            )
            kwargs = export_mock.call_args.kwargs["kwargs"]
            self.assertIs(kwargs["pixel_values"], explicit_pixel_values)
            self.assertNotIn("input_features", kwargs)


@require_torch
class ExecutorchTest(unittest.TestCase):
    def setUp(self):
        set_seed(42)
        self.model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.model.eval()

        # Create generation config with static cache for the model
        self.model.generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )

        self.input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        self.inputs_embeds = torch.randn(1, 3, self.model.config.hidden_size)
        self.cache_position = torch.arange(3, dtype=torch.long)

    def test_static_cache_module_forward(self):
        """Test TorchExportableModuleWithStaticCache forward with both input types"""
        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )

        # Set generation config on model
        self.model.generation_config = generation_config
        module = TorchExportableModuleWithStaticCache(self.model)

        # Test with input_ids
        eager_output_ids = self.model(input_ids=self.input_ids, use_cache=False).logits
        wrapped_output_ids = module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_ids, wrapped_output_ids, atol=1e-4, rtol=1e-4)

        # Test with inputs_embeds
        eager_output_embeds = self.model(inputs_embeds=self.inputs_embeds, use_cache=False).logits
        wrapped_output_embeds = module.forward(inputs_embeds=self.inputs_embeds, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_embeds, wrapped_output_embeds, atol=1e-4, rtol=1e-4)

    def test_hybrid_cache_module_forward(self):
        """Test TorchExportableModuleWithHybridCache forward with both input types"""
        config = self.model.config
        config.sliding_window = 16
        config.layer_types = ["full_attention"] * config.num_hidden_layers

        generation_config = GenerationConfig(
            use_cache=True,
            cache_implementation="hybrid",
            cache_config={"batch_size": 1, "max_cache_len": 32, "device": "cpu"},
        )

        # Set generation config on model
        self.model.generation_config = generation_config
        module = TorchExportableModuleWithHybridCache(self.model)

        # Test with input_ids
        eager_output_ids = self.model(input_ids=self.input_ids, use_cache=False).logits
        wrapped_output_ids = module.forward(input_ids=self.input_ids, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_ids, wrapped_output_ids, atol=1e-4, rtol=1e-4)

        # Test with inputs_embeds
        eager_output_embeds = self.model(inputs_embeds=self.inputs_embeds, use_cache=False).logits
        wrapped_output_embeds = module.forward(inputs_embeds=self.inputs_embeds, cache_position=self.cache_position)
        torch.testing.assert_close(eager_output_embeds, wrapped_output_embeds, atol=1e-4, rtol=1e-4)

    def test_decoder_only_lm_export_validation(self):
        """Test TorchExportableModuleForDecoderOnlyLM export validation"""
        module = TorchExportableModuleForDecoderOnlyLM(self.model)

        # Should fail with both input_ids and inputs_embeds
        with self.assertRaises(ValueError):
            module.export(input_ids=self.input_ids, inputs_embeds=self.inputs_embeds)

        # Should fail with neither
        with self.assertRaises(ValueError):
            module.export()

    def test_decoder_only_lm_export(self):
        """Test TorchExportableModuleForDecoderOnlyLM export with both input types"""
        module = TorchExportableModuleForDecoderOnlyLM(self.model)

        # Test export with input_ids
        exported_program_ids = module.export(input_ids=self.input_ids, cache_position=self.cache_position)
        eager_output_ids = self.model(input_ids=self.input_ids, use_cache=False).logits
        exported_output_ids = exported_program_ids.module()(
            input_ids=self.input_ids, cache_position=self.cache_position
        )
        torch.testing.assert_close(eager_output_ids, exported_output_ids, atol=1e-4, rtol=1e-4)

        # Test export with inputs_embeds
        exported_program_embeds = module.export(inputs_embeds=self.inputs_embeds, cache_position=self.cache_position)
        eager_output_embeds = self.model(inputs_embeds=self.inputs_embeds, use_cache=False).logits
        exported_output_embeds = exported_program_embeds.module()(
            inputs_embeds=self.inputs_embeds, cache_position=self.cache_position
        )
        torch.testing.assert_close(eager_output_embeds, exported_output_embeds, atol=1e-4, rtol=1e-4)

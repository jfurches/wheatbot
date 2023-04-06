from typing import Dict as TypingDict

from gymnasium.spaces import Dict, Box

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as FCNet
from ray.rllib.models.torch.recurrent_net import LSTMWrapper
from ray.rllib.models.torch.attention_net import AttentionWrapper
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement

from ray.rllib.utils.torch_utils import FLOAT_MIN
TensorType = torch.Tensor

class TorchLSTMActionMaskModel(TorchModelV2, nn.Module):
    """Parametric action space model that uses LSTMWrapper"""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        new_obs_space = orig_space['observations']

        TorchModelV2.__init__(self, orig_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        wrapper = ModelCatalog._wrap_if_needed(FCNet, LSTMWrapper)
        wrapper._wrapped_forward = FCNet.forward

        self.internal_model: LSTMWrapper = wrapper(new_obs_space, action_space, num_outputs,
                                                   model_config, name + '_internal_model')
        self.view_requirements = self.internal_model.view_requirements
        self.view_requirements[SampleBatch.OBS] = ViewRequirement(SampleBatch.OBS, space=orig_space, shift=0)

        # for k in ('state_in_0', 'state_in_1', 'state_out_0', 'state_out_1'):
        #     self.view_requirements[k] = ViewRequirement(
        #         k,
        #         space=Box(low=-np.inf, high=np.inf, shape=(model_config.get('lstm_cell_size'),)), 
        #         shift=0
        #     )
        # print(self.view_requirements)

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get('no_masking', False)

    def forward(self, input_dict: TypingDict[str, TensorType], state, seq_lens):
        # Copy the input dict so we don't modify the original
        new_input_dict = input_dict.copy()

        obs = new_input_dict["obs"]["observations"]
        action_mask = new_input_dict["obs"]["action_mask"]

        # remove the action mask
        new_input_dict['obs'] = obs
        new_input_dict['obs_flat'] = obs

        # Compute the unmasked logits by passing through lstm
        logits, state = self.internal_model(new_input_dict, state, seq_lens)

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits
        return masked_logits, state

    def value_function(self) -> TensorType:
        return self.internal_model.value_function()

    def get_initial_state(self):
        return self.internal_model.get_initial_state()


class TorchAttnActionMaskModel(TorchModelV2, nn.Module):
    """Parametric action space model that uses AttentionWrapper"""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        new_obs_space = orig_space['observations']

        TorchModelV2.__init__(self, orig_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        wrapper = ModelCatalog._wrap_if_needed(FCNet, AttentionWrapper)
        wrapper._wrapped_forward = FCNet.forward

        self.internal_model: AttentionWrapper = wrapper(new_obs_space, action_space, num_outputs,
                                                   model_config, name + '_internal_model')
        self.view_requirements = self.internal_model.view_requirements
        self.view_requirements[SampleBatch.OBS] = ViewRequirement(SampleBatch.OBS, space=orig_space, shift=0)

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get('no_masking', False)

    def forward(self, input_dict: TypingDict[str, TensorType], state, seq_lens):
        # Copy the input dict so we don't modify the original
        new_input_dict = input_dict.copy()

        obs = new_input_dict["obs"]["observations"]
        action_mask = new_input_dict["obs"]["action_mask"]

        # remove the action mask
        new_input_dict['obs'] = obs
        new_input_dict['obs_flat'] = obs

        # Compute the unmasked logits by passing through gtrxl
        logits, state = self.internal_model(new_input_dict, state, seq_lens)

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits
        return masked_logits, state

    def value_function(self) -> TensorType:
        return self.internal_model.value_function()

    def get_initial_state(self):
        return self.internal_model.get_initial_state()
import numpy as np
import warnings
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
warnings.filterwarnings("ignore")

# --- 通用工具类：用于处理多变量切分与部分归一化 ---

class BaseBatchNormModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_splits, norm_indices):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_splits = num_splits
        self.norm_indices = norm_indices
        
        self.split_size = obs_space.shape[0] // num_splits
        
        self.bn_layers = nn.ModuleDict({
            str(i): nn.BatchNorm1d(self.split_size) for i in norm_indices
        })

        hiddens = model_config.get("fcnet_hiddens", [256, 256, 256, 256])
        activation = model_config.get("fcnet_activation", "tanh")
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        layers = []
        prev_layer_size = obs_space.shape[0]
        for size in hiddens:
            layers.append(nn.Linear(prev_layer_size, size))
            layers.append(act_fn())
            prev_layer_size = size
        
        self.hidden_layers = nn.Sequential(*layers)
        self.action_out = nn.Linear(prev_layer_size, num_outputs)
        self.value_out = nn.Linear(prev_layer_size, 1)
        self._last_flat_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        parts = list(torch.split(obs, self.split_size, dim=1))
        
        for i in self.norm_indices:
            idx = i - 1 
            parts[idx] = self.bn_layers[str(i)](parts[idx])
        
        x = torch.cat(parts, dim=1)
        
        self._last_flat_out = self.hidden_layers(x)
        logits = self.action_out(self._last_flat_out)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        return torch.reshape(self.value_out(self._last_flat_out), [-1])


# --- 具体模型实现 ---

class BatchNormModel(BaseBatchNormModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, 
                         num_splits=5, norm_indices=[2, 3, 4, 5])

class BatchNormModel2(BaseBatchNormModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, 
                         num_splits=6, norm_indices=[3, 4, 5, 6])

class BatchNormModel3(BaseBatchNormModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, 
                         num_splits=8, norm_indices=[3, 4, 5, 6, 7, 8])

class BatchNormModel4(BaseBatchNormModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, 
                         num_splits=9, norm_indices=[3, 4, 5, 6, 7, 8, 9])

class GameNormModel(BaseBatchNormModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        model_config["fcnet_hiddens"] = [64, 64]
        super().__init__(obs_space, action_space, num_outputs, model_config, name, 
                         num_splits=8, norm_indices=[2, 5, 6, 7, 8])


# --- 包装 RLlib 内置 FullyConnectedNetwork 的类 ---

class CustomModelTorchAgent1(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        model_config['fcnet_hiddens'] = [256, 256, 256, 256]
        self.model = TorchFCNet(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

class CustomModelTorchAgent2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.model = TorchFCNet(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
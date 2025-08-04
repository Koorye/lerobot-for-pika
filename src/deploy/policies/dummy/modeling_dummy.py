import builtins
import torch
from collections import deque
from pathlib import Path
from torch import Tensor
from typing import TypeVar

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_dummy import DummyConfig

T = TypeVar("T", bound="PreTrainedPolicy")


class DummyPolicy(PreTrainedPolicy):

    config_class = DummyConfig
    name = "dummy"

    def __init__(
        self,
        config: DummyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        self.num_action_steps = config.num_action_steps
        self.action = torch.tensor(config.action, dtype=torch.float32)
        self.reset()
    
    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        print('DummyPolicy does not need method from_pretrained, return a new instance directly.')
        if config is None:
            config = DummyConfig()
        return DummyPolicy(config, **kwargs)

    def get_optim_params(self):
        return {}
    
    def reset(self):
        self._action_queue = deque([], maxlen=self.num_action_steps)
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, :, self.num_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        # (7,) -> (1, 1, 7) -> (B, N, 7)
        actions = self.action.unsqueeze(0).unsqueeze(0).repeat(
            batch["observations"].shape[0], self.num_action_steps, 1
        )
        return actions

    def forward(self, batch):
        pass

    def push_model_to_hub(self, cfg):
        pass

    def generate_model_card(self, dataset_repo_id, model_type, license, tags):
        pass

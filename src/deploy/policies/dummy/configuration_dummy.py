from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("dummy")
@dataclass
class DummyConfig(PreTrainedConfig):
    num_action_steps: int = 16
    action: list[int] = field(default_factory=lambda: [1, 0, 0, 0, 0, 0, 0])

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

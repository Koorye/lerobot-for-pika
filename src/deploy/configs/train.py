from dataclasses import dataclass

from lerobot.configs.train import TrainPipelineConfig as TrainPipelineConfig_


@dataclass
class TrainPipelineConfig(TrainPipelineConfig_):
    input_keys: list[str] = None

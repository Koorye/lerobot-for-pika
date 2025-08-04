# from lerobot.policies.factory import 
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

from .dummy.configuration_dummy import DummyConfig


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "dummy":
        from .dummy.modeling_dummy import DummyPolicy

        return DummyPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    elif policy_type == "dummy":
        return DummyConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")

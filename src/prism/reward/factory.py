# registers the rewards and returns the RewardManager class

import argparse
from src.prism.reward.scorer import RewardManager
from src.prism.reward.scoring.molecular_props import QEDReward, SAScoreReward, LipinskiReward, BertzReward
from src.prism.reward.scoring.sucos import SuCOSReward
from src.prism.reward.scoring.interaction_fingerprints import InteractionFingerprintsReward
from src.prism.reward.scoring.feature_density import FeatureDensityReward
from src.prism.reward.scoring.geometry_checks import PoseBustersGeometryChecks
from src.prism.reward.scoring.flatness_checks import PoseBustersFlatnessReward
from src.prism.reward.scoring.aromatic_counter import AromaticBonus
from src.prism.reward.scoring.aromatic_feature import AromaticFeatureReward
# 1. The Registry

REWARD_REGISTRY = {
    "qed": QEDReward,
    "sa_score": SAScoreReward,
    "lipinski": LipinskiReward,
    "bertz": BertzReward,
    "sucos": SuCOSReward,
    "interaction_fingerprints": InteractionFingerprintsReward,
    "feature_density": FeatureDensityReward,
    "geometry_checks": PoseBustersGeometryChecks,
    "flatness_checks": PoseBustersFlatnessReward,
    "aromatic_counter": AromaticBonus,
    "aromatic_anchor": AromaticFeatureReward,
}

def get_reward_manager(config, dataset_info, ddpm_module=None):
    """
    Parses the namespace config, instantiates specific reward classes, 
    and returns the Orchestrator.
    
    Args:
        config: The argparse.Namespace object from train.py
        dataset_info: Dataset metadata
        ddpm_module: The diffusion model (for virtual nodes etc)
    """
    active_rewards = []
    weights = {}

    # 1. Safely access 'reward_params' from the Namespace
    # We use getattr because older configs might not have this section
    reward_params_ns = getattr(config, 'reward_params', None)

    if reward_params_ns is None:
        print("WARNING: 'reward_params' not found in config. No rewards will be calculated.")
        return RewardManager([], {}, dataset_info, ddpm_module)

    # 2. Safely access 'rewards' from the nested Namespace
    rewards_ns = getattr(reward_params_ns, 'rewards', None)

    if rewards_ns is None:
        print("WARNING: 'rewards' list is empty or missing.")
        return RewardManager([], {}, dataset_info, ddpm_module)

    # 3. Convert Namespace to Dict for iteration
    # Since dict_to_namespace works recursively, 'rewards_ns' is a Namespace.
    # vars() returns the __dict__ attribute, allowing us to iterate over it.
    if isinstance(rewards_ns, argparse.Namespace):
        rewards_dict = vars(rewards_ns)
    elif isinstance(rewards_ns, dict):
        # Fallback in case it somehow remained a dict
        rewards_dict = rewards_ns
    else:
        raise TypeError(f"Unexpected type for config.reward_params.rewards: {type(rewards_ns)}")

    reward_paths_ns = getattr(reward_params_ns, 'reward_paths', None)
    if isinstance(reward_paths_ns, argparse.Namespace):
        reward_paths = vars(reward_paths_ns)
    elif isinstance(reward_paths_ns, dict):
        reward_paths = reward_paths_ns
    else:
        reward_paths = {}
        
    print(f"Initializing Rewards: {rewards_dict}")
    print(f"Initializing Reward Paths: {reward_paths}")

    for name, weight in rewards_dict.items():
        # Skip zero-weighted rewards or internal keys
        if float(weight) <= 0:
            continue
            
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Reward '{name}' defined in config but not found in REWARD_REGISTRY.")
        
        # Instantiate the class
        reward_cls = REWARD_REGISTRY[name]

        # Only pass dataset_info to rewards that need it (like SuCOS)
        if name == 'sucos' or name == 'interaction_fingerprints':
            active_rewards.append(reward_cls(dataset_info))
        elif name == 'feature_density':
            active_rewards.append(reward_cls(reward_paths['feature_density']))
            ########## TODO: remove later:
        elif name == 'aromatic_anchor':
            active_rewards.append(reward_cls(reward_paths['aromatic_anchor']))
        else:
            active_rewards.append(reward_cls())
            
        weights[name] = float(weight)


    return RewardManager(
        reward_fns=active_rewards,
        reward_weights=weights,
        dataset_info=dataset_info,
        ddpm_module=ddpm_module
    )
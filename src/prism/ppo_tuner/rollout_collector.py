import torch
import math
import traceback
from collections import defaultdict

from tests.ppo_debug_utils import assert_same_ids, dbg_tensor


class RolloutCollector:
    """
    A self-contained class responsible for collecting experience (rollouts)
    by running the diffusion model policy.
    """
    def __init__(self, policy_network, reward_function, config):
        self.policy_network = policy_network
        self.reward_function = reward_function
        self.config = config

        # NOTE: We get the device directly from the policy network
        self.device = next(self.policy_network.parameters()).device

    @torch.no_grad()
    def collect(self, pocket_batch, current_epoch, get_ligand_and_pocket_fn):
        """
        Generates molecule rollouts for a batch of pockets.
        """
        self.policy_network.eval()
        
        rollout_data = {
            'molecules': ([], []), 
            'masks': ([], []), 
            'rewards': [], 
            'raw_score': [],
            'old_log_probs': [], 
            'z_states': [], 
            'pocket_indices': [],
            'component_scores': defaultdict(list)
        }
        
        # names is a list of IDs for the whole batch (e.g. ['1c3b...', '7e2z...'])
        _, pocket_data, names = get_ligand_and_pocket_fn(pocket_batch)

        local_batch_size = len(pocket_batch['num_pocket_nodes'])
        
        if torch.distributed.is_initialized():
            world_size, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
        else:
            world_size, rank = 1, 0

        # Determine Sampling Strategy
        # can add in a eval num steps instead but okay for now.
        
        n_steps = self.config.ppo_params.n_steps
        
        samples_per_rank = math.ceil(n_steps / world_size)
        samples_per_pocket = math.ceil(samples_per_rank / max(1, local_batch_size))
        total_target_samples = min(samples_per_rank, local_batch_size * samples_per_pocket)

        total_samples, valid_samples = 0, 0
        max_pockets_per_rank = max(1, local_batch_size)
        global_offset = rank * max_pockets_per_rank * samples_per_pocket
        
        # Main Loop: Process Pockets
        max_chunk_size = min(4, local_batch_size)
        for chunk_start_outer in range(0, local_batch_size, max_chunk_size):
            if total_samples >= total_target_samples:
                break
            
            chunk_end_outer = min(chunk_start_outer + max_chunk_size, local_batch_size)
            
            pocket_sample_counts = []
            remaining_to_gen = total_target_samples - total_samples
            for pocket_idx in range(chunk_start_outer, chunk_end_outer):
                
                samples_for_pocket = min(samples_per_pocket, remaining_to_gen)
                if samples_for_pocket <= 0:
                    continue
                pocket_sample_counts.append((pocket_idx, samples_for_pocket))
                remaining_to_gen -= samples_for_pocket

            for pocket_idx, samples_to_generate in pocket_sample_counts:
                pocket_mask_base = global_offset + (pocket_idx * samples_per_pocket)
                ligand_chunk_size = self.config.ppo_params.ligand_chunk_size
                
                # --- FIX START: Identify the specific name for this pocket ---
                current_name = names[pocket_idx]
                
                pocket_mask_idx = (pocket_data['mask'] == pocket_idx)                
                if pocket_mask_idx.sum() == 0:
                    print(f"  WARNING: No atoms found for pocket_idx={pocket_idx}!")
                    print(f"  Available mask values: {torch.unique(pocket_data['mask'])}")
                    print(f"  Skipping this pocket...")
                    continue  # Skip to next pocket instead of crashing
                
                for lig_chunk_start in range(0, samples_to_generate, ligand_chunk_size):
                    try:
                        samples_in_chunk = min(ligand_chunk_size, samples_to_generate - lig_chunk_start)
                        chunk_mask_base = pocket_mask_base + lig_chunk_start
                        
                        single_pocket_data = {
                            'x': pocket_data['x'][pocket_mask_idx],
                            'one_hot': pocket_data['one_hot'][pocket_mask_idx],
                            'size': pocket_data['size'][pocket_idx:pocket_idx+1],
                            'mask': torch.zeros_like(pocket_data['mask'][pocket_mask_idx])
                        }

                        # --- FIX CONTINUE: Create corrected name list ---
                        # We repeat the correct name for every sample in this chunk
                        chunk_names = [current_name] * samples_in_chunk

                        chunk_results = self._generate_and_evaluate_chunk(
                            single_pocket_data, 
                            samples_in_chunk, 
                            chunk_mask_base, 
                            chunk_names,
                            current_epoch
                        )
                        # ------------------------------------------------

                        # Append results
                        rollout_data['molecules'][0].append(chunk_results['xh_lig'])
                        rollout_data['molecules'][1].append(chunk_results['xh_pocket'])
                        rollout_data['masks'][0].append(chunk_results['global_lig_mask'])
                        rollout_data['masks'][1].append(chunk_results['global_pocket_mask'])
                        rollout_data['rewards'].append(chunk_results['rewards'])
                        rollout_data['raw_score'].append(chunk_results['raw_score'])
                        rollout_data['old_log_probs'].append(chunk_results['old_log_probs'])
                        rollout_data['z_states'].append(chunk_results['z_states'])
                        rollout_data['pocket_indices'].append(
                            torch.full((samples_in_chunk,), pocket_idx, device=self.device, dtype=torch.long)
                        )
                        if 'component_scores' in chunk_results:
                            for key, value in chunk_results['component_scores'].items():
                                rollout_data['component_scores'][key].append(value)
                        
                        valid_samples += chunk_results['rewards'].shape[0]
                        total_samples += samples_in_chunk

                    except Exception as e:
                        print(f"[ERROR] RollooutCollector failed for pocket {pocket_idx}: {str(e)}")
                        traceback.print_exc()
                        continue
        
        # Final Aggregation and Processing
        return self._aggregate_rollouts(rollout_data, valid_samples, rank)


    def _generate_and_evaluate_chunk(self, single_pocket_data, samples_in_chunk, chunk_mask_base, names, current_epoch):
        """
        Generates, remaps, and calculates rewards for a single chunk of molecules.
        """
    
        for key, tensor in single_pocket_data.items():
            if torch.is_tensor(tensor):
                single_pocket_data[key] = tensor.to(self.device) 
                
        num_nodes_lig_config = self.config.ppo_params.num_nodes_lig
        if num_nodes_lig_config is None:
            # leads to instability for training with size distribution 
            # num_nodes_lig = self.policy_network.size_distribution.sample_conditional(
            #     n1=None, n2=single_pocket_data['size']
            # ).repeat(samples_in_chunk)
            num_nodes_lig = torch.randint(15, 40, (samples_in_chunk,), dtype=torch.long)
            # print(f"num_nodes_lig sampled from range [15, 55]: {num_nodes_lig}")
        else:
            num_nodes_lig = torch.randint(
                num_nodes_lig_config - 10,
                num_nodes_lig_config + 10,
                (samples_in_chunk,),
                dtype=torch.long
            )
        
        num_nodes_lig = num_nodes_lig.to(self.device)
        print(f"num_nodes_lig: {num_nodes_lig}")


        local_mask_base = torch.arange(samples_in_chunk, device=self.device)
        batched_pocket_data = {
            'x': single_pocket_data['x'].repeat(samples_in_chunk, 1),
            'one_hot': single_pocket_data['one_hot'].repeat(samples_in_chunk, 1),
            'size': single_pocket_data['size'].repeat(samples_in_chunk),
            'mask': local_mask_base.repeat_interleave(single_pocket_data['x'].shape[0]).to(self.device)
        }

        xh_lig, xh_pocket, lig_mask, pocket_mask, mol_log_probs, z_states = \
            self.policy_network.sample_given_pocket(batched_pocket_data, num_nodes_lig)

        sample_mask = torch.arange(samples_in_chunk, device=self.device) + chunk_mask_base
        global_lig_mask = sample_mask[lig_mask]
        global_pocket_mask = sample_mask[pocket_mask]
        
        ######################### DEBUGGING #########################
        assert_same_ids("collect_rollouts/post_sample", global_lig_mask, global_pocket_mask)
        # dbg_tensor("collect_rollouts/global_lig_mask", global_lig_mask)
        # dbg_tensor("collect_rollouts/global_pocket_mask", global_pocket_mask)
        ######################### DEBUGGING #########################

        rewards, component_scores = self.reward_function(
            xh_lig, 
            global_lig_mask,
            current_epoch=current_epoch,
            # kwargs:
            xh_pocket=xh_pocket,
            global_pocket_mask=global_pocket_mask,
            names=names # carries the pocket names - important for docking etc.
        )

        raw_score = rewards.clone() 
        
        return {
            'xh_lig': xh_lig, 
            'xh_pocket': xh_pocket,
            'global_lig_mask': global_lig_mask, 
            'global_pocket_mask': global_pocket_mask,
            'rewards': rewards, 
            'raw_score': raw_score,
            'old_log_probs': torch.stack(mol_log_probs, dim=1),
            'z_states': torch.stack(z_states, dim=1),
            'component_scores': component_scores
        }
        
        
    def _aggregate_rollouts(self, rollout_data, valid_samples, rank):
        """
        Takes rollout data with lists of tensors and aggregates them into
        final, concatenated tensors. Also handles final timestep processing.
        """
        if valid_samples > 0:
            rollout_data['molecules'] = (torch.cat(rollout_data['molecules'][0]), torch.cat(rollout_data['molecules'][1]))
            rollout_data['masks'] = (torch.cat(rollout_data['masks'][0]), torch.cat(rollout_data['masks'][1]))
            rollout_data['rewards'] = torch.cat(rollout_data['rewards'])
            rollout_data['raw_score'] = torch.cat(rollout_data['raw_score'])
            rollout_data['old_log_probs'] = torch.cat(rollout_data['old_log_probs'])
            rollout_data['z_states'] = torch.cat(rollout_data['z_states'])
            rollout_data['pocket_indices'] = torch.cat(rollout_data['pocket_indices'])
            
            final_component_scores = {}
            for key, tensor_list in rollout_data['component_scores'].items():
                if tensor_list:
                    final_component_scores[key] = torch.cat(tensor_list)
            rollout_data['component_scores'] = final_component_scores
                
            z_states = rollout_data['z_states']
            rollout_data['latents'] = z_states[:, :-1]
            rollout_data['next_latents'] = z_states[:, 1:]
            
            num_molecules = rollout_data['rewards'].shape[0]
            seq_length_minus1 = rollout_data['z_states'].shape[1] - 1

            # Determine timesteps
            diffusion_steps_config = self.config.diffusion_params.diffusion_steps
            
            if isinstance(diffusion_steps_config, int):
                diffusion_steps = diffusion_steps_config
                timesteps_1d = torch.arange(
                    diffusion_steps - 1, -1, -1, device=self.device
                )[:seq_length_minus1]
            else:
                timesteps_1d = diffusion_steps_config.flip(0)[:seq_length_minus1]

            rollout_data['timesteps'] = timesteps_1d.unsqueeze(0).repeat(num_molecules, 1)
        else:
            print(f"[DEBUG] Rank {rank} collected no valid samples, returning empty rollout_data.")
            empty_shape = (0, self.policy_network.atom_nf + self.policy_network.n_dims)
            rollout_data['molecules'] = (torch.empty(empty_shape, device=self.device), torch.empty(empty_shape, device=self.device))
            rollout_data['masks'] = (torch.empty(0, device=self.device, dtype=torch.long), torch.empty(0, device=self.device, dtype=torch.long))
            for key in ['rewards', 'raw_score', 'old_log_probs', 'z_states', 'pocket_indices']:
                rollout_data[key] = torch.empty(0, device=self.device, dtype=torch.long)
        
        return rollout_data
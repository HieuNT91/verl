# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        
        # Initialize per-question statistics tracking
        self.per_question_statistics = {}
        self.current_epoch = 0
        
        # EMA decay for std score estimation (higher = more weight on recent epochs)
        self.ema_decay = getattr(self.config.algorithm, 'ema_decay', 0.7)
        
        # Minimum repeat times for each question (to avoid cases when std = 0)
        self.min_repeat_times = getattr(self.config.algorithm, 'min_repeat_times', 4)
        
        # Estimated std score for each question (used for adaptive repeat times)
        self.estimated_std_score_per_question = {}

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            self.current_epoch = epoch
            for batch_dict in self.train_dataloader:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                # Calculate adaptive repeat times for each question
                repeat_times_per_question = self._calculate_adaptive_repeat_times(batch_dict)
                
                # Log repeat times for monitoring
                if self.current_epoch > 0 and self.global_steps % 10 == 0:  # Log every 10 steps after first epoch
                    total_repeats = sum(repeat_times_per_question.values())
                    avg_repeats = total_repeats / len(repeat_times_per_question) if repeat_times_per_question else 0
                    print(f"Epoch {self.current_epoch}, Step {self.global_steps}: "
                          f"Adaptive repeats - Total: {total_repeats}, Avg: {avg_repeats:.2f}, "
                          f"Questions: {len(repeat_times_per_question)}")
                
                # Apply adaptive repeating
                gen_batch = self._apply_adaptive_repeat(gen_batch, batch_dict, repeat_times_per_question)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout using adaptive repeat times
                    new_batch = self._apply_adaptive_repeat(new_batch, batch_dict, repeat_times_per_question)
                    new_batch = new_batch.union(gen_batch_output)

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            
                        # Update per-question statistics
                        self._update_per_question_statistics(batch_dict, reward_extra_infos_dict)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.gen_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch - calculate expected trajectory batch size based on adaptive repeats
                            expected_total_trajectories = sum(repeat_times_per_question.values())
                            batch = batch[:expected_total_trajectories]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    
                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        
                        # Calculate effective number of repeats for this batch
                        effective_num_repeat = self._get_effective_num_repeat(batch_dict, repeat_times_per_question)
                        
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=effective_num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # Add per-question statistics to metrics
                per_question_summary = self.get_per_question_statistics_summary()
                extreme_stats = self._analyze_extreme_question_stats()
                
                metrics.update({
                    "per_question/total_questions": per_question_summary["total_questions"],
                    "per_question/avg_mean_acc": per_question_summary["avg_mean_acc"],
                    "per_question/avg_mean_score": per_question_summary["avg_mean_score"],
                    "per_question/avg_std_acc": per_question_summary["avg_std_acc"],
                    "per_question/avg_std_score": per_question_summary["avg_std_score"],
                    "extreme_stats/all_zero_acc": extreme_stats["all_zero_acc"],
                    "extreme_stats/all_one_acc": extreme_stats["all_one_acc"],
                    "extreme_stats/middle_acc": extreme_stats["middle_acc"],
                    "extreme_stats/zero_std_acc": extreme_stats["zero_std_acc"],
                    "extreme_stats/zero_std_score": extreme_stats["zero_std_score"],
                    "extreme_stats/high_std_acc": extreme_stats["high_std_acc"],
                    "extreme_stats/high_std_score": extreme_stats["high_std_score"]
                })

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
            
            # Finalize epoch statistics at the end of each epoch
            self._finalize_epoch_statistics()
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
    
    def _update_per_question_statistics(self, batch_dict, reward_extra_infos_dict):
        """
        Update per-question statistics with acc and scores from reward_extra_infos_dict.
        
        Args:
            batch_dict: The original batch dictionary containing 'index' field with question UUIDs
            reward_extra_infos_dict: Dictionary containing 'acc' and 'score' arrays
        """
        if 'index' not in batch_dict:
            print("Warning: 'index' field not found in batch_dict, skipping per-question statistics update")
            return
            
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
            
        # Extract acc and scores from reward_extra_infos_dict
        acc_values = reward_extra_infos_dict.get('acc', [])
        score_values = reward_extra_infos_dict.get('score', [])
        
        # Handle case where we have different numbers of entries
        max_entries = max(len(question_uuids), len(acc_values), len(score_values))
        
        for i in range(max_entries):
            # Get question UUID (cycle through if fewer UUIDs than entries)
            question_uuid = question_uuids[i % len(question_uuids)] if question_uuids else f"unknown_{i}"
            
            # Get acc and score values (use 0.0 as default if not available)
            acc = acc_values[i] if i < len(acc_values) else 0.0
            score = score_values[i] if i < len(score_values) else 0.0
            
            # Initialize question statistics if not exists
            if question_uuid not in self.per_question_statistics:
                self.per_question_statistics[question_uuid] = {
                    'current_epoch_acc_values': [],  # Accumulator for current epoch
                    'current_epoch_score_values': [],  # Accumulator for current epoch
                    'mean_acc_per_epoch': [],  # List of mean acc for each epoch
                    'mean_score_per_epoch': [],  # List of mean score for each epoch
                    'std_acc_per_epoch': [],  # List of std acc for each epoch
                    'std_score_per_epoch': [],  # List of std score for each epoch
                    'count_per_epoch': [],  # List of counts for each epoch
                    'last_processed_epoch': -1
                }
            
            # Update statistics for current epoch
            stats = self.per_question_statistics[question_uuid]
            stats['current_epoch_acc_values'].append(acc)
            stats['current_epoch_score_values'].append(score)
    
    def _finalize_epoch_statistics(self):
        """
        Finalize statistics for the current epoch and save to file.
        """
        import json
        import os
        
        for question_uuid, stats in self.per_question_statistics.items():
            # Only process if we haven't processed this epoch for this question yet
            if stats['last_processed_epoch'] < self.current_epoch:
                current_acc_values = stats['current_epoch_acc_values']
                current_score_values = stats['current_epoch_score_values']
                
                if current_acc_values or current_score_values:
                    # Calculate statistics for this epoch
                    mean_acc = np.mean(current_acc_values) if current_acc_values else 0.0
                    mean_score = np.mean(current_score_values) if current_score_values else 0.0
                    std_acc = np.std(current_acc_values) if len(current_acc_values) > 1 else 0.0
                    std_score = np.std(current_score_values) if len(current_score_values) > 1 else 0.0
                    count = len(current_acc_values)
                    
                    # Append to epoch lists
                    stats['mean_acc_per_epoch'].append(mean_acc)
                    stats['mean_score_per_epoch'].append(mean_score)
                    stats['std_acc_per_epoch'].append(std_acc)
                    stats['std_score_per_epoch'].append(std_score)
                    stats['count_per_epoch'].append(count)
                    
                    # Clear current epoch accumulators
                    stats['current_epoch_acc_values'] = []
                    stats['current_epoch_score_values'] = []
                    stats['last_processed_epoch'] = self.current_epoch
                    
                    # Update EMA estimation for std_score
                    if question_uuid not in self.estimated_std_score_per_question:
                        # First epoch for this question
                        self.estimated_std_score_per_question[question_uuid] = std_score
                    else:
                        # EMA update: new_estimate = decay * current_std + (1 - decay) * old_estimate
                        old_estimate = self.estimated_std_score_per_question[question_uuid]
                        self.estimated_std_score_per_question[question_uuid] = (
                            self.ema_decay * std_score + (1 - self.ema_decay) * old_estimate
                        )
        
        # Save to file
        self._save_per_question_statistics_to_file()
        
        # Log extreme statistics at the end of each epoch
        extreme_stats = self._analyze_extreme_question_stats()
        self._log_extreme_stats(extreme_stats)
    
    def _save_per_question_statistics_to_file(self):
        """
        Save per-question statistics to a JSON file.
        """
        import json
        import os
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self.config.trainer.default_local_dir, "per_question_stats")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for saving (exclude current epoch accumulators)
        save_data = {}
        for question_uuid, stats in self.per_question_statistics.items():
            save_data[question_uuid] = {
                'mean_acc_per_epoch': stats['mean_acc_per_epoch'],
                'mean_score_per_epoch': stats['mean_score_per_epoch'],
                'std_acc_per_epoch': stats['std_acc_per_epoch'],
                'std_score_per_epoch': stats['std_score_per_epoch'],
                'count_per_epoch': stats['count_per_epoch'],
                'total_epochs_seen': len(stats['mean_acc_per_epoch'])
            }
        
        # Save to file
        filename = f"per_question_statistics_epoch_{self.current_epoch}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Saved per-question statistics to {filepath}")
        
        # Also save the latest as a separate file for easy access
        latest_filepath = os.path.join(output_dir, "per_question_statistics_latest.json")
        with open(latest_filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def get_per_question_statistics_summary(self):
        """
        Get a summary of per-question statistics.
        
        Returns:
            dict: Summary containing total questions tracked and average metrics across all questions
        """
        if not self.per_question_statistics:
            return {"total_questions": 0, "avg_mean_acc": 0.0, "avg_mean_score": 0.0, "avg_std_acc": 0.0, "avg_std_score": 0.0}
        
        total_questions = len(self.per_question_statistics)
        
        # Get latest epoch statistics for each question
        current_mean_accs = []
        current_mean_scores = []
        current_std_accs = []
        current_std_scores = []
        
        for stats in self.per_question_statistics.values():
            if stats['mean_acc_per_epoch']:
                current_mean_accs.append(stats['mean_acc_per_epoch'][-1])
                current_mean_scores.append(stats['mean_score_per_epoch'][-1])
                current_std_accs.append(stats['std_acc_per_epoch'][-1])
                current_std_scores.append(stats['std_score_per_epoch'][-1])
        
        return {
            "total_questions": total_questions,
            "avg_mean_acc": np.mean(current_mean_accs) if current_mean_accs else 0.0,
            "avg_mean_score": np.mean(current_mean_scores) if current_mean_scores else 0.0,
            "avg_std_acc": np.mean(current_std_accs) if current_std_accs else 0.0,
            "avg_std_score": np.mean(current_std_scores) if current_std_scores else 0.0,
        }
    
    def _save_checkpoint(self):
        """
        Override parent method to include per-question statistics in checkpoints.
        """
        # Call parent save method first
        super()._save_checkpoint()
        
        # Save per-question statistics
        import json
        import os
        
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        
        # Save per-question statistics
        stats_data = {
            'per_question_statistics': self.per_question_statistics,
            'estimated_std_score_per_question': self.estimated_std_score_per_question,
            'current_epoch': self.current_epoch,
            'ema_decay': self.ema_decay,
            'min_repeat_times': self.min_repeat_times
        }
        
        stats_path = os.path.join(local_global_step_folder, "per_question_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        print(f"Saved per-question statistics to {stats_path}")
    
    def _load_checkpoint(self):
        """
        Override parent method to restore per-question statistics from checkpoints.
        """
        # Get the resumed global steps from parent method
        resumed_steps = super()._load_checkpoint()
        
        if self.config.trainer.resume_mode != "disable":
            import json
            import os
            
            # Find the checkpoint folder
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            
            # Construct the global step folder path
            if hasattr(self, 'global_steps') and self.global_steps > 0:
                global_step_folder = os.path.join(checkpoint_folder, f"global_step_{self.global_steps}")
                stats_path = os.path.join(global_step_folder, "per_question_stats.json")
                
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path, 'r') as f:
                            stats_data = json.load(f)
                        
                        self.per_question_statistics = stats_data.get('per_question_statistics', {})
                        self.estimated_std_score_per_question = stats_data.get('estimated_std_score_per_question', {})
                        self.current_epoch = stats_data.get('current_epoch', 0)
                        self.ema_decay = stats_data.get('ema_decay', self.ema_decay)
                        # Only restore min_repeat_times if it was saved, otherwise keep config value
                        if 'min_repeat_times' in stats_data:
                            self.min_repeat_times = stats_data['min_repeat_times']
                        
                        print(f"Loaded per-question statistics from {stats_path}")
                        print(f"Restored {len(self.per_question_statistics)} question statistics")
                        print(f"Resumed from epoch {self.current_epoch}")
                    except Exception as e:
                        print(f"Warning: Failed to load per-question statistics: {e}")
                        # Initialize empty if loading fails
                        self.per_question_statistics = {}
                        self.estimated_std_score_per_question = {}
                else:
                    print(f"No per-question statistics found at {stats_path}, starting fresh")
        
        return resumed_steps
    
    def _calculate_adaptive_repeat_times(self, batch_dict):
        """
        Calculate adaptive repeat times for each question based on estimated std scores.
        
        Args:
            batch_dict: The original batch dictionary containing 'index' field with question UUIDs
            
        Returns:
            dict: Mapping from question UUID to repeat times
        """
        if 'index' not in batch_dict:
            # Fallback to default repeat times if no index available
            default_repeat = self.config.actor_rollout_ref.rollout.n
            return {f"unknown_{i}": default_repeat for i in range(len(batch_dict.get('input_ids', [])))}
        
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
        
        # For first epoch, use default repeat times
        if self.current_epoch == 0:
            default_repeat = self.config.actor_rollout_ref.rollout.n
            return {uuid: default_repeat for uuid in question_uuids}
        
        # Calculate adaptive repeat times for subsequent epochs
        batch_size = len(question_uuids)
        total_budget = self.config.actor_rollout_ref.rollout.n * batch_size
        
        # Get estimated std scores for questions in this batch
        batch_std_scores = []
        for uuid in question_uuids:
            if uuid in self.estimated_std_score_per_question:
                batch_std_scores.append(self.estimated_std_score_per_question[uuid])
            else:
                # Use a default std score for new questions
                batch_std_scores.append(0.1)  # Default moderate variance
        
        # Avoid division by zero
        total_std_score = sum(batch_std_scores)
        if total_std_score == 0:
            # If all std scores are 0, use minimum repeat times
            return {uuid: self.min_repeat_times for uuid in question_uuids}
        
        # Calculate repeat times proportional to std scores
        repeat_times = {}
        for i, uuid in enumerate(question_uuids):
            proportion = batch_std_scores[i] / total_std_score
            repeat_time = max(self.min_repeat_times, int(total_budget * proportion))  # At least min_repeat_times
            repeat_times[uuid] = repeat_time
        
        # Ensure total doesn't exceed budget (due to rounding)
        total_assigned = sum(repeat_times.values())
        if total_assigned > total_budget:
            # Proportionally reduce but respect minimum repeat times
            factor = total_budget / total_assigned
            for uuid in repeat_times:
                repeat_times[uuid] = max(self.min_repeat_times, int(repeat_times[uuid] * factor))
        
        return repeat_times
    
    def _apply_adaptive_repeat(self, gen_batch, batch_dict, repeat_times_per_question):
        """
        Apply adaptive repeat times to generate batch based on per-question requirements.
        
        Args:
            gen_batch: The generation batch (DataProto)
            batch_dict: Original batch dictionary with question UUIDs
            repeat_times_per_question: Dict mapping question UUID to repeat times
            
        Returns:
            DataProto: Batch with adaptive repeating applied
        """
        if 'index' not in batch_dict:
            # Fallback to original repeat logic if no index available
            return gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
        
        # Create a list to hold repeated batches
        repeated_batches = []
        
        for i, uuid in enumerate(question_uuids):
            # Get the repeat times for this question
            repeat_times = repeat_times_per_question.get(uuid, 1)
            
            # Extract the single item for this question
            single_item = gen_batch[i:i+1]  # DataProto slice
            
            # Repeat this single item
            repeated_item = single_item.repeat(repeat_times=repeat_times, interleave=False)
            
            repeated_batches.append(repeated_item)
        
        # Concatenate all repeated batches
        if repeated_batches:
            final_batch = DataProto.concat(repeated_batches)
        else:
            # Fallback to original if no batches created
            final_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        
        return final_batch
    
    def _get_effective_num_repeat(self, batch_dict, repeat_times_per_question):
        """
        Calculate the effective number of repeats for advantage computation.
        Since different questions may have different repeat times, we use the average.
        
        Args:
            batch_dict: Original batch dictionary with question UUIDs
            repeat_times_per_question: Dict mapping question UUID to repeat times
            
        Returns:
            float: Effective number of repeats (average across questions)
        """
        if 'index' not in batch_dict or not repeat_times_per_question:
            return self.config.actor_rollout_ref.rollout.n
        
        question_uuids = batch_dict['index']
        if not isinstance(question_uuids, (list, tuple, np.ndarray)):
            question_uuids = [question_uuids]
        
        # Calculate average repeat times
        total_repeats = sum(repeat_times_per_question.get(uuid, 1) for uuid in question_uuids)
        avg_repeats = total_repeats / len(question_uuids) if question_uuids else 1
        
        return avg_repeats
    
    def _analyze_extreme_question_stats(self):
        """
        Analyze and log questions with extreme accuracy/std patterns.
        
        Returns:
            dict: Statistics about extreme cases
        """
        if not self.per_question_statistics:
            return {"all_zero_acc": 0, "all_one_acc": 0, "middle_acc": 0, "zero_std": 0, "high_std": 0}
        
        all_zero_acc = 0
        all_one_acc = 0
        middle_acc = 0
        zero_std_acc = 0
        zero_std_score = 0
        high_std_acc = 0  # std > 0.3
        high_std_score = 0  # std > 0.3
        
        for uuid, stats in self.per_question_statistics.items():
            if not stats['mean_acc_per_epoch']:
                continue
                
            # Get latest epoch statistics
            latest_mean_acc = stats['mean_acc_per_epoch'][-1]
            latest_std_acc = stats['std_acc_per_epoch'][-1] if stats['std_acc_per_epoch'] else 0.0
            latest_std_score = stats['std_score_per_epoch'][-1] if stats['std_score_per_epoch'] else 0.0
            
            # Categorize by accuracy
            if latest_mean_acc == 0.0:
                all_zero_acc += 1
            elif latest_mean_acc == 1.0:
                all_one_acc += 1
            else:
                middle_acc += 1
            
            # Categorize by std
            if latest_std_acc == 0.0:
                zero_std_acc += 1
            elif latest_std_acc > 0.3:
                high_std_acc += 1
                
            if latest_std_score == 0.0:
                zero_std_score += 1
            elif latest_std_score > 0.3:
                high_std_score += 1
        
        total_questions = len(self.per_question_statistics)
        
        return {
            "total_questions": total_questions,
            "all_zero_acc": all_zero_acc,
            "all_one_acc": all_one_acc,
            "middle_acc": middle_acc,
            "zero_std_acc": zero_std_acc,
            "zero_std_score": zero_std_score,
            "high_std_acc": high_std_acc,
            "high_std_score": high_std_score,
        }
    
    def _log_extreme_stats(self, extreme_stats):
        """
        Log extreme statistics in a readable format.
        """
        total = extreme_stats["total_questions"]
        if total == 0:
            print("No questions tracked yet.")
            return
        
        print(f"\n=== Extreme Question Statistics (Epoch {self.current_epoch}) ===")
        print(f"Total Questions: {total}")
        
        # Accuracy distribution
        print(f"\nAccuracy Distribution:")
        print(f"  All Zero (0.0):   {extreme_stats['all_zero_acc']:3d} ({extreme_stats['all_zero_acc']/total*100:5.1f}%)")
        print(f"  All One (1.0):    {extreme_stats['all_one_acc']:3d} ({extreme_stats['all_one_acc']/total*100:5.1f}%)")
        print(f"  Middle (0.0-1.0): {extreme_stats['middle_acc']:3d} ({extreme_stats['middle_acc']/total*100:5.1f}%)")
        
        # Standard deviation distribution
        print(f"\nStandard Deviation Distribution:")
        print(f"  Zero Std Acc:     {extreme_stats['zero_std_acc']:3d} ({extreme_stats['zero_std_acc']/total*100:5.1f}%)")
        print(f"  High Std Acc:     {extreme_stats['high_std_acc']:3d} ({extreme_stats['high_std_acc']/total*100:5.1f}%)")
        print(f"  Zero Std Score:   {extreme_stats['zero_std_score']:3d} ({extreme_stats['zero_std_score']/total*100:5.1f}%)")
        print(f"  High Std Score:   {extreme_stats['high_std_score']:3d} ({extreme_stats['high_std_score']/total*100:5.1f}%)")
        print("=" * 60)

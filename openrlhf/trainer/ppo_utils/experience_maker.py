import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from time import sleep

from openrlhf.models.actor import Actor
from openrlhf.models.ring_attn_utils import pad_sequences, unpad_sequences
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray, remote_rm_fn_ray_refinement

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    labels: list[str]
    pad_len: Optional[int]
    rank: Optional[int]
    step: Optional[int]
    rewards: Optional[List[torch.Tensor]] = None


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: Union[list[str], str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        remote_rm_url = [remote_rm_url] if isinstance(remote_rm_url, str) else remote_rm_url
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts, labels)` from {remote_rm_url[0]}")
            import importlib.util

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    # def make_experience_list(
    #     self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    # ) -> List[Experience]:
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, step: Optional[int] = None, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # print(f"[Experience Maker] Running make_experience_list")

        # generate responses
        if self.strategy.ring_attn_group is not None:
            # Only rank 0 in the ring attention group executes the generation function, and then broadcasts it to all other ranks.
            if self.strategy.ring_attn_rank == 0:
                samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
                dist.broadcast_object_list(samples_list, src=dist.get_rank(), group=self.strategy.ring_attn_group)
            else:
                world_size = torch.distributed.get_world_size() // args.ring_attn_size
                samples_list = [None] * (
                    args.rollout_batch_size * args.n_samples_per_prompt // world_size // args.micro_rollout_batch_size
                )
                dist.broadcast_object_list(
                    samples_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                )
        else:
            # samples_list = self.generate_samples(all_prompts, all_labels, **generate_kwargs)
            samples_list = self.generate_samples(all_prompts, all_labels, step=step, **generate_kwargs)

        print(f"[Experience Maker] Running make_experience")
        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu")) # CHANGE: pass in rank & step as metadata

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo", "reinforce_baseline", "group_norm", "group_norm_refinement"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        # print(f"[Experience Maker] Finished making experience_list")
        return experiences

    @torch.no_grad()
    # def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
    def generate_samples(self, all_prompts: List[str], all_labels, step: Optional[int] = None, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            labels = all_labels[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompts,
                labels=labels,
                pad_len=None,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
        else:
            base_action_log_probs = None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.labels).to(
                    device=action_log_probs.device
                )
            else:
                r = remote_rm_fn(
                    self.remote_rm_url, queries=queries, prompts=samples.prompts, labels=samples.labels
                ).to(device=action_log_probs.device)
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for rloo and reinforce_baseline
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            rewards = rewards - rewards.mean(-1, keepdim=True)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        elif args.advantage_estimator == "group_norm":
            assert len(experiences) == 1, (
                   "Group norm advantage estimator only supports one task. "
            )
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            # print(f"Rewards before normalization: {rewards}")
            
            # Create mask for non-nan values
            mask = ~torch.isnan(rewards)
            # print(f"Mask: {mask}")
            for experience in experiences:
                experience_mask = ~torch.isnan(experience.info["reward"])
                experience.info["reward"] = experience.info["reward"][experience_mask]
                # print(f"[process_experiences] experience.info['reward']: {experience.info['reward']}, experience_mask: {experience_mask}")
            
            # Calculate mean using nanmean
            mean = torch.nanmean(rewards, dim=-1, keepdim=True)
            
            # Calculate std manually with masking
            n = mask.sum(dim=-1, keepdim=True)
            masked_diff = torch.where(mask, (rewards - mean) ** 2, torch.tensor(0.0, device=rewards.device))
            # Clamp n-1 to minimum of 1 to avoid division by zero
            std = torch.sqrt(masked_diff.sum(dim=-1, keepdim=True) / (n - 1).clamp(min=1.0))
            
            # print(f"Mean: {mean}, Std: {std}")
            rewards = (rewards - mean) / (std + 1e-9)
            
            # print(f"Rewards after normalization: {rewards}")
            rewards = rewards[mask].reshape(-1)
            rewards = rewards.to(device="cpu").chunk(len(experiences))
            
            # rank = torch.distributed.get_rank()
            # print(f"DEDUP: [Rank {rank}] Rewards after removing nan: {rewards}")
            
            return experiences, rewards
        elif args.advantage_estimator == "group_norm_refinement":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt // args.num_passes, args.num_passes).to(device="cuda")
            rewards = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + 1e-9)
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            return experiences, rewards

        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    # def make_experience_list(
    #     self, all_prompts: Union[str, List[str]], all_labels, **generate_kwargs
    # ) -> List[Experience]:
    def make_experience_list(
        self, all_prompts: Union[str, List[str]], all_labels, step: Optional[int] = None, **generate_kwargs
    ) -> List[Experience]:
        print(f"[Step {step}] Making experience list")
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        # experiences = super().make_experience_list(all_prompts, all_labels, **generate_kwargs)
        experiences = super().make_experience_list(all_prompts, all_labels, step=step, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    # def generate_samples(self, all_prompts: List[str], all_labels, **generate_kwargs) -> List[Samples]:
    def generate_samples(self, all_prompts: List[str], all_labels, step: Optional[int] = None, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, all_labels, **generate_kwargs)
        args = self.strategy.args
        # vLLM generation
        if args.refinement:
            samples = self._generate_vllm_refinement(all_prompts, all_labels, step=step, **generate_kwargs)
        else:
            samples = self._generate_vllm(all_prompts, all_labels, step=step, **generate_kwargs)
        return samples

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        rank = samples.rank
        step = samples.step
        metadata = {"rank": rank, "step": step}
        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.initial_model is not None:
            torch.distributed.barrier()
            print(f"[Rank {rank}] [Step {step}] Running initial model")
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, logps_allgather=True, packed_seq_lens=packed_seq_lens
            )

            if args.colocate_actor_ref or args.colocate_all_models:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])
                torch.distributed.barrier()
            print(f"[Rank {rank}] [Step {step}] Finished running initial model")
        else:
            base_action_log_probs_ref = ray.put(None)
        
        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        # rewards handling - use pre-calculated rewards from _generate_vllm
        if samples.rewards is not None:
            rewards = samples.rewards
            # print(f"[Rank {rank}] [Step {step}] Using pre-calculated rewards: {rewards}")
        else:
            if not self.remote_rm_url:
                r_refs = []
                for rm in self.reward_model:
                    r_refs.append(
                        rm.forward.remote(
                            sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens, pad_sequence=True
                        )
                    )
                if args.colocate_all_models:
                    ray.get(r_refs)
                    ray.get([self.reward_model[0].empty_cache.remote()])
                rewards = ray.get(r_refs)
            else:
                print(f"[Rank {rank}] [Step {step}] WARNING: No pre-calculated rewards available")
    
        actor_value_rm_time = time.time() - start
        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        wait_time = time.time() - start

        base_action_log_probs, value = ref_values[0], ref_values[1]
        if base_action_log_probs is not None:
            base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        if self.packing_samples:
            action_log_probs = torch.zeros_like(sequences, dtype=torch.float)
        else:
            action_log_probs = torch.zeros_like(action_mask, dtype=torch.float)
        
        # Process rewards
        rewards = [r.to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        if (self.initial_model is not None) and (not args.use_kl_loss):
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                kl_estimator=self.strategy.args.kl_estimator,
            )
        else:
            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            if self.strategy.ring_attn_group is not None:
                assert samples.pad_len is not None
                sequences, attention_mask, num_actions, packed_seq_lens, _, _, kl = unpad_sequences(
                    pad_len=samples.pad_len,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    num_actions=num_actions,
                    packed_seq_lens=packed_seq_lens,
                    ring_attn_group=self.strategy.ring_attn_group,
                    action_log_probs=action_log_probs,
                    values=value,
                    kl=kl,
                )
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)
            if base_action_log_probs is not None:
                base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        if not args.use_kl_loss:
            base_action_log_probs = None

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience
    
    def _get_chat_template(self):
        with open(self.strategy.args.chat_template, 'r') as f:
            return f.read()

    def _generate_vllm_refinement(self, all_prompts: List[str], all_labels, step: Optional[int] = None, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        overall_start_time = time.time()
        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        world_size = torch.distributed.get_world_size()

        # No need to distribute requests to engines if num_actors == num_engines
        llm = self.vllm_engines[rank]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        self.tokenizer.chat_template = self._get_chat_template()

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * (args.n_samples_per_prompt//args.num_passes) for prompt in all_prompts], [])
        all_labels = sum([[label] * (args.n_samples_per_prompt//args.num_passes) for label in all_labels], [])

        # Initialize messages with chat template and think tag
        all_messages = [
            [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], 
                    tokenize=False, 
                    add_generation_prompt=True
                ) + "<think>\n"
            ] for prompt in all_prompts
        ]

        all_outputs = [["" for _ in range(args.num_passes)] for _ in range(len(all_prompts))]
        all_rewards = torch.zeros(len(all_prompts), args.num_passes)
        all_feedbacks = [["" for _ in range(args.num_passes)] for _ in range(len(all_prompts))]
        all_feedbacks_lengths = torch.zeros(len(all_prompts), args.num_passes)

        # Remove examples from prompts and apply chat template
        example_indices = [prompt.find("Here's an example") for prompt in all_prompts]
        all_prompts = [
            self.tokenizer.apply_chat_template(
                [{
                    "role": "user", 
                    "content": (prompt[:example_indices[i]] if example_indices[i] != -1 else prompt) + "Here are your previous attempts:\n"
                }],
                tokenize=False,
                add_generation_prompt=True
            ) for i, prompt in enumerate(all_prompts)
        ]
        all_prompts_lengths = [len(self.tokenize_fn(prompt, self.prompt_max_len, padding=False)["input_ids"]) for prompt in all_prompts]
        
        for i in range(args.num_passes):
            print(f"[Rank {rank}] [Step {step}] [Refinement Step {i+1}] Generating samples")

            # wake up vLLM engine
            ref = llm.wake_up.remote()
            ray.get(ref)
            torch.cuda.synchronize()

            # Generate samples
            current_messages = [trajectory_message[-1] for trajectory_message in all_messages]
            prompt_token_ids = self.tokenize_fn(current_messages, self.prompt_max_len, padding=False)["input_ids"]
            ref = llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            current_outputs = ray.get(ref)
            current_texts = [output.outputs[0].text.strip() for output in current_outputs]

            # vLLM offload when vllm_enable_sleep
            ref = llm.sleep.remote()
            ray.get(ref)
            torch.cuda.synchronize()
            
            # send to reward server
            metadata = {"rank": rank, "step": step, "refinement_step": i + 1, "sample_indices": list(range(len(all_prompts))), "train": True}
            ref = remote_rm_fn_ray_refinement.remote(self.remote_rm_url[0], queries=current_texts, prompts=current_messages, labels=all_labels, metadata=metadata)
            rewards, infos = ray.get(ref)

            # process eval feedback and add to prompt, truncating from the left if it exceeds the max prompt length
            for j in range(len(all_prompts)):
                all_feedbacks[j][i] = (current_texts[j].split("</think>")[-1].lstrip('\n') if "</think>" in current_texts[j] and self.tokenizer.eos_token in current_texts[j] else self.tokenizer.eos_token) + \
                                self.tokenizer.apply_chat_template([{"role": "user", "content": infos[j]}], tokenize=False, add_generation_prompt=True)
                all_feedbacks_lengths[j][i] = len(self.tokenize_fn(all_feedbacks[j][i], self.prompt_max_len, padding=False)["input_ids"])
                new_prompt = ""
                new_prompt_length = 0
                
                for k in range(i, -1, -1):
                    new_prompt_length += all_feedbacks_lengths[j][k]
                    if new_prompt_length + all_prompts_lengths[j] >= self.prompt_max_len - 32: # Buffer since len(tokenize(A)) + len(tokenize(B)) might not equal len(tokenize(A + B)).
                        break
                    new_prompt = all_feedbacks[j][k] + new_prompt
                new_prompt = all_prompts[j] + new_prompt + "<think>\n"
                if i < args.num_passes - 1:
                    all_messages[j].append(new_prompt)
                all_rewards[j][i] = rewards[j]
                all_outputs[j][i] = current_outputs[j]

        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        print(f"[Rank {rank}] [Step {step}] Total refinement process completed in {overall_duration:.2f} seconds")

        # apply gamma to all_rewards
        for i in reversed(range(args.num_passes - 1)):
            all_rewards[:, i] += args.refinement_gamma * all_rewards[:, i + 1] 
            # all_rewards[:, i] = torch.maximum(all_rewards[:, i], args.refinement_gamma * all_rewards[:, i + 1])
        
        all_messages = sum(all_messages, [])
        all_labels = sum([[label] * args.num_passes for label in all_labels], [])
        all_outputs = sum(all_outputs, [])
        all_rewards = [all_rewards.reshape(-1)]
        
        if args.overlong_filtering:
            mask = all_rewards[0] > -10.0
        else:
            mask = torch.ones_like(all_rewards[0], dtype=torch.bool)
        rewards = [all_rewards[0].masked_fill(~mask, float('nan'))]

        print(f"DEDUP: [Rank {rank}] Mask: {mask}, Rewards: {rewards}")

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_messages[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]

            # Apply mask to filter out invalid samples
            mask_slice = mask[i:i + self.strategy.args.micro_rollout_batch_size]
            outputs = [out for out, m in zip(outputs, mask_slice) if m]
            prompts = [p for p, m in zip(prompts, mask_slice) if m] 
            labels = [l for l, m in zip(labels, mask_slice) if m]
            
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        labels=labels,
                        pad_len=None,
                        rank=rank,
                        step=step,
                        rewards=rewards
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=pad_len,
                        rank=rank,
                        step=step,
                        rewards=rewards
                    )
                )
        return samples_list

    def _generate_vllm(self, all_prompts: List[str], all_labels, step: Optional[int] = None, **kwargs) -> List[Samples]:
        from vllm import SamplingParams
        args = self.strategy.args

        # round-robin load balance
        rank = torch.distributed.get_rank() // self.strategy.ring_attn_size
        print(f"[Rank {rank}] [Step {step}] Generating samples")
        world_size = torch.distributed.get_world_size() // self.strategy.ring_attn_size

        if args.vllm_enable_sleep:
            if args.colocate_all_models:
                # No need to synchronize if each actor corresponds to a single vLLM engine
                # print(f"[Experience Maker {rank}] waking up vLLM engine")
                ref = self.vllm_engines[rank].wake_up.remote()
                ray.get(ref)
            else:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

                batch_vllm_engine_call(self.vllm_engines, "wake_up")
                torch.distributed.barrier()
            torch.cuda.synchronize()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            # logprobs=1
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_labels = sum([[label] * args.n_samples_per_prompt for label in all_labels], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        
        if args.colocate_all_models:
            # No need to distribute requests to engines if num_actors == num_engines
            llm = self.vllm_engines[rank]
            ref = llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=all_prompt_token_ids)
            all_outputs = ray.get(ref)
        else:
            # Distribute requests to engines and collect responses to outputs
            refs = []
            batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
            for i, llm in enumerate(llms):
                prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
                refs.append(
                    llm.add_requests.remote(rank, sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )
            ray.get(refs)

            # Make sure all requests are sent.
            if self.strategy.ring_attn_group is None:
                torch.distributed.barrier()
            else:
                time.sleep(3)
    
            # Retrieve and combine results from all outputs
            all_output_refs = []
            for i, llm in enumerate(llms):
                all_output_refs.append(llm.get_responses.remote(rank))
            all_outputs = sum(ray.get(all_output_refs), [])
        
        # vLLM offload when vllm_enable_sleep
        if args.colocate_all_models:
            print(f"[Experience Maker {rank}] sleeping vLLM engine")
            ref = self.vllm_engines[rank].sleep.remote()
            ray.get(ref)
        else:
            if args.vllm_enable_sleep:
                from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
                batch_vllm_engine_call(self.vllm_engines, "sleep")
            torch.distributed.barrier()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
            
        # Calculate rewards for all outputs at once if remote reward model is available
        all_rewards = None
        if self.remote_rm_url:
            if self.strategy.ring_attn_group is None or self.strategy.ring_attn_rank == 0:
                # Prepare all sequences for decoding
                metadata = {"rank": rank, "step": step}
                
                # Prepare all sequences for batch decoding
                all_sequences = []
                for output in all_outputs:
                    all_sequences.append(output.prompt_token_ids + list(output.outputs[0].token_ids))
                
                all_decoded_texts = self.tokenizer.batch_decode(all_sequences, skip_special_tokens=False)
                # if rank == 0:
                    # print(f"[Rank {rank}] [Step {step}] len(all_decoded_texts): {len(all_decoded_texts)}")
                
                print(f"[Rank {rank}] [Step {step}] Calculating rewards for all {len(all_decoded_texts)} outputs at once")
                r_refs = []
                
                if self.custom_reward_func:
                    r = self.custom_reward_func.remote(all_decoded_texts, all_prompts, all_labels)
                    r_refs.append(r)
                else:
                    for rm in self.remote_rm_url:
                        r = remote_rm_fn_ray.remote(
                            rm, 
                            queries=all_decoded_texts, 
                            prompts=all_prompts, 
                            labels=all_labels, 
                            metadata=metadata
                        )
                        r_refs.append(r)
                
                print(f"[Rank {rank}] [Step {step}] Waiting for all rewards")
                all_rewards = ray.get(r_refs)
                print(f"[Rank {rank}] [Step {step}] Received all rewards")
            
            # Broadcast rewards to all ring attention ranks
            if self.strategy.ring_attn_group is not None:
                if self.strategy.ring_attn_rank == 0:
                    dist.broadcast_object_list([all_rewards], src=dist.get_rank(), group=self.strategy.ring_attn_group)
                else:
                    all_rewards_list = [None]
                    dist.broadcast_object_list(
                        all_rewards_list, src=self.strategy.ring_attn_ranks[0], group=self.strategy.ring_attn_group
                    )
                    all_rewards = all_rewards_list[0]

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            labels = all_labels[i : i + self.strategy.args.micro_rollout_batch_size]
            
            # Extract relevant rewards for this batch
            batch_rewards = None
            if all_rewards is not None:
                batch_rewards = [reward[i:i + self.strategy.args.micro_rollout_batch_size] for reward in all_rewards]
            
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        labels=labels,
                        pad_len=None,
                        rank=rank,
                        step=step,
                        rewards=batch_rewards
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))
                    num_actions.append(max(1, output_len))

                # pad seq makes the sequence a multiple of ring_attention_size.
                pad_len = None
                if self.strategy.ring_attn_group is not None:
                    pad_len, sequences, attention_mask, num_actions, packed_seq_lens = pad_sequences(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        ring_attn_group=self.strategy.ring_attn_group,
                        pad_token_id=pad_token_id,
                    )

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        labels=labels,
                        pad_len=pad_len,
                        rank=rank,
                        step=step,
                        rewards=batch_rewards
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None

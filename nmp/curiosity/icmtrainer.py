from collections import OrderedDict

import numpy as np

import rlkit.torch.pytorch_util as ptu
import torch
import torch.optim as optim
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.sac import SACTrainer
from torch import nn as nn


class ICMTrainer(SACTrainer):
    def __init__(
        self,
        env,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        icm,
        discount=0.99,
        reward_scale=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        optimizer_class=optim.Adam,
        soft_target_tau=1e-2,
        target_update_period=1,
        render_eval_paths=False,
        alpha=1,
        lmbda=0.1,
        beta=0.2,
        use_automatic_entropy_tuning=True,
        target_entropy=None,
        policy_optimizer=None,
        qf1_optimizer=None,
        qf2_optimizer=None,
    ):
        self.icm = icm

        self.forward_criterion = nn.MSELoss()
        self.inverse_criterion = nn.MSELoss()

        self.beta = beta
        self.lmbda = lmbda

        super().__init__(
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            discount,
            reward_scale,
            policy_lr,
            qf_lr,
            optimizer_class,
            soft_target_tau,
            target_update_period,
            render_eval_paths,
            alpha,
            use_automatic_entropy_tuning,
            target_entropy,
            policy_optimizer,
            qf1_optimizer,
            qf2_optimizer,
        )

        self.policy_optimizer.add_param_group(self.icm.parameters())

        # if add_param_group doesnt work, try this:
        #
        # self.policy_optimizer = optimizer_class(
        #   list(self.policy.parameters()) + list(self.icm.parameters()),
        #   lr=policy_lr,
        # )

    def train_from_torch(self, batch):
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )

        '''
        Forward and Inverse Loss
        '''

        next_h, next_h_pred, actions_pred = self.icm(actions, obs, next_obs)
        forward_loss = self.forward_criterion(next_h_pred, next_h)
        inverse_loss = self.inverse_criterion(actions, actions_pred)

        """
        Policy and Alpha Loss
        """
        # TODO: should the alpha_loss be modified to include inverse_loss and forward_loss ?

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        policy_icm_loss = self.lmbda*policy_loss + (1-self.beta)*inverse_loss + self.beta*forward_loss
        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = (
            torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            )
            - alpha * new_log_pi
        )

        q_target = (
            self.reward_scale * rewards
            + (1.0 - terminals) * self.discount * target_q_values
        )
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_icm_loss.backward()
        norm_policy = nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
        norm_icm = nn.utils.clip_grad_norm_(self.icm.parameters(), 100)
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        norm_qf1 = nn.utils.clip_grad_norm_(self.qf1.parameters(), 100)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        norm_qf2 = nn.utils.clip_grad_norm_(self.qf2.parameters(), 100)
        self.qf2_optimizer.step()
        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics["QF1 Grad Norm"] = ptu.get_numpy(norm_qf1)
            self.eval_statistics["QF2 Grad Norm"] = ptu.get_numpy(norm_qf2)
            self.eval_statistics["Policy Grad Norm"] = ptu.get_numpy(norm_policy)

            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q1 Predictions", ptu.get_numpy(q1_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q2 Predictions", ptu.get_numpy(q2_pred),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Q Targets", ptu.get_numpy(q_target),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Log Pis", ptu.get_numpy(log_pi),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("Policy mu", ptu.get_numpy(policy_mean),)
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std", ptu.get_numpy(policy_log_std),
                )
            )
            # ICM statistics
            self.eval_statistics["Forward Loss"] = np.mean(ptu.get_numpy(forward_loss))
            self.eval_statistics["Inverse Loss"] = np.mean(ptu.get_numpy(inverse_loss))
            self.eval_statistics["Policy+ICM Loss"] = np.mean(ptu.get_numpy(policy_icm_loss))
            self.eval_statistics["ICM Grad Norm"] = ptu.get_numpy(norm_icm)

            self.eval_statistics.update(
                create_stats_ordered_dict("ICM next_h_pred", ptu.get_numpy(next_h_pred), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("ICM next_h", ptu.get_numpy(next_h), )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict("ICM actions_pred", ptu.get_numpy(actions_pred), )
            )
            ###

            self.eval_statistics["Alpha"] = alpha.item()
            if self.use_automatic_entropy_tuning:
                self.eval_statistics["Alpha Loss"] = alpha_loss.item()
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    @networks.setter
    def networks(self, nets):
        self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2 = nets

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            policy_optimizer=self.policy_optimizer,
            qf1_optimizer=self.qf1_optimizer,
            qf2_optimizer=self.qf2_optimizer,
        )
        for k, v in snapshot.items():
            if "optimizer" not in k:
                if hasattr(v, "module"):
                    snapshot[k] = v.module
        return snapshot
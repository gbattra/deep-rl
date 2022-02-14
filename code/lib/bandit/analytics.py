# Greg Attra
# 01/28/2022

from typing import Tuple
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt


class BanditAnalytics:
    def __init__(self, n_problems: int, n_steps: int, n_actions: int) -> None:
        self.n_problems = n_problems
        self.n_steps = n_steps
        self.n_actions = n_actions
        self.ar_hist = np.zeros((n_problems, n_steps, n_actions, 2))
        self.q_stars = np.zeros((n_problems, n_actions))

    def update_qstar(self, p: int, q_star: np.array) -> None:
        """
        Set the qstar values for problem p
        """
        self.q_stars[p] = q_star

    def update_ar_hist(self, p: int, s: int, a: int, r: float) -> None:
        """
        Insert new sample into ar history for problem p
        """
        self.ar_hist[p, s, a] = (r, 1)

    def splot(self, ax: plt.Axes, rgb: Tuple[float, float, float]) -> None:
        p = 0
        action_samples = []
        for a in range(self.ar_hist.shape[2]):
            samples = self.ar_hist[p, :, a, 0][self.ar_hist[p, :, a, 1] == 1]
            action_samples.append(samples)
        faces = ax.violinplot(action_samples, range(1, 11), showmeans=True)
        faces['cmeans'].set(color='blue')

        q_means = []
        for a in range(self.q_stars.shape[1]):
            m = self.q_stars[p, a]
            x_start = a + 0.5
            x_end = a + 1.5
            q_means.append([[x_start, x_end], [m, m]])
        
        for q in q_means:
            ax.plot(q[0], q[1], 'r--')
        
        plt.xlabel('Actions')
        plt.ylabel('Reward Distribution')

        ax.legend(
            [Line2D([0], [0], linestyle='--', c='r'), Line2D([0], [0], c='b')],
            ['q* mean', 'Q mean'])

    def plot(self, rgb: Tuple[float, float, float]) -> None:
        fig, ax = plt.subplots()

        self.splot(ax, rgb)

        plt.show()


class EpsilonGreedyBanditAnalytics(BanditAnalytics):
    def __init__(
            self,
            eps: float,
            n_problems: int,
            n_steps: int,
            n_actions: int) -> None:
        super().__init__(n_problems, n_steps, n_actions)
        self.eps = eps

    def splot(self, ax: plt.Axes, rgb: Tuple[float, float, float], style: str) -> None:
        sum_rwd_per_step = self.ar_hist.sum(axis=2)
        step_avg_rwd = sum_rwd_per_step.mean(axis=0)

        stde_avg_rwd = 1.96 * (np.std(sum_rwd_per_step[:, :, 0], axis=0) / np.sqrt(self.n_problems))
        y_neg = step_avg_rwd[:, 0] - stde_avg_rwd
        y_pos = step_avg_rwd[:, 0] + stde_avg_rwd
        ax[0].fill_between(
            range(self.n_steps),
            y_neg,
            y_pos,
            alpha=0.2,
            color=rgb)
        
        ax[0].plot(
            step_avg_rwd[:, 0],
            label='Eps: ' + str(self.eps),
            color=rgb)

        optimal_actions = np.argmax(self.q_stars, axis=1)
        chosen_actions = np.argmax(self.ar_hist[:, :, :, 1], axis=2)
        mask = chosen_actions == optimal_actions.reshape(optimal_actions.shape[0], 1)
        pct_optimal = mask.sum(axis=0) / self.n_problems
        ax[1].plot(
            pct_optimal,
            label='Eps: ' + str(self.eps),
            color=rgb)

        ax.flat[0].set(xlabel='Steps', ylabel='Avg Reward')
        ax.flat[1].set(xlabel='Steps', ylabel='Pct. Optimal')

        ax[0].legend()
        ax[1].legend()

class OptimisticGreedyBanditAnalytics(BanditAnalytics):
    def __init__(
            self,
            q_init: float,
            eps: float,
            n_problems: int,
            n_steps: int,
            n_actions: int) -> None:
        super().__init__(n_problems, n_steps, n_actions)
        self.eps = eps
        self.q_init = q_init

    def splot(self, ax: plt.Axes, rgb: Tuple[float, float, float], style: str) -> None:
        sum_rwd_per_step = self.ar_hist.sum(axis=2)
        step_avg_rwd = sum_rwd_per_step.mean(axis=0)

        stde_avg_rwd = 1.96 * (np.std(sum_rwd_per_step[:, :, 0], axis=0) / np.sqrt(self.n_problems))
        y_neg = step_avg_rwd[:, 0] - stde_avg_rwd
        y_pos = step_avg_rwd[:, 0] + stde_avg_rwd
        ax[0].fill_between(
            range(self.n_steps),
            y_neg,
            y_pos,
            alpha=0.2,
            color=rgb)

        ax[0].plot(
            step_avg_rwd[:, 0],
            label='Q init: ' + str(self.q_init) + ', Eps: ' + str(self.eps),
            color=rgb)

        optimal_actions = np.argmax(self.q_stars, axis=1)
        chosen_actions = np.argmax(self.ar_hist[:, :, :, 1], axis=2)
        mask = chosen_actions == optimal_actions.reshape(optimal_actions.shape[0], 1)
        pct_optimal = mask.sum(axis=0) / self.n_problems
        ax[1].plot(
            pct_optimal,
            label='Q init: ' + str(self.q_init) + ', Eps: ' + str(self.eps),
            color=rgb)

        ax.flat[0].set(xlabel='Steps', ylabel='Avg Reward')
        ax.flat[1].set(xlabel='Steps', ylabel='Pct. Optimal')


class UcbBanditAnalytics(BanditAnalytics):
    def __init__(self, c: float, n_problems: int, n_steps: int, n_actions: int) -> None:
        super().__init__(n_problems, n_steps, n_actions)
        self.c = c

    def splot(self, ax: plt.Axes, rgb: Tuple[float, float, float], style: str) -> None:
        sum_rwd_per_step = self.ar_hist.sum(axis=2)
        step_avg_rwd = sum_rwd_per_step.mean(axis=0)
        ax[0].plot(step_avg_rwd[:, 0], label='UCB, c = ' + str(self.c), color=rgb)

        stde_avg_rwd = 1.96 * (np.std(sum_rwd_per_step[:, :, 0], axis=0) / np.sqrt(self.n_problems))
        y_neg = step_avg_rwd[:, 0] - stde_avg_rwd
        y_pos = step_avg_rwd[:, 0] + stde_avg_rwd
        ax[0].fill_between(
            range(self.n_steps),
            y_neg,
            y_pos,
            alpha=0.2,
            color=rgb)

        optimal_actions = np.argmax(self.q_stars, axis=1)
        chosen_actions = np.argmax(self.ar_hist[:, :, :, 1], axis=2)
        mask = chosen_actions == optimal_actions.reshape(optimal_actions.shape[0], 1)
        pct_optimal = mask.sum(axis=0) / self.n_problems
        ax[1].plot(pct_optimal, label='UCB, c = ' + str(self.c), color=rgb)

        ax.flat[0].set(xlabel='Steps', ylabel='Avg Reward')
        ax.flat[1].set(xlabel='Steps', ylabel='Pct. Optimal')

"""
階層的 Predictive Coding モデル (ground truth dynamics)
=======================================================

Rao & Ballard (1999) / Friston (2005, 2009) の free-energy 最小化に基づく
離散時間の階層的予測符号化ネットワーク。

各層 l=1..L は
  - 状態            x_l  ∈ R^{d_l}
  - 下層への予測    μ_l  = g(W_l · x_l)                 (g: nonlinearity)
  - 予測誤差        ε_l  = x_{l-1} - μ_l                (層 l-1 の observation を説明)
  - 精度            π_l  (スカラー or 対角, 動的更新可)
を保持する。x_0 は sensory drive（刺激）によって直接駆動される。

状態更新 (Euler, gradient descent on variational free energy):
    Δx_l =  η_l · ( W_l^T diag(π_{l-1}) ε_{l-1}         # bottom-up PE
                  - diag(π_l) ε_l                         # top-down correction
                  - λ_l · x_l )                          # leak
    ε_l   = x_{l-1} - g(W_l x_l)   (x_{-1} := sensory drive for l=1)

精度の動的更新（volatility 追従）:
    π_l ← π_l + η_π · ( 1/(ε_l·ε_l + κ) - π_l )
（surprise が低く安定した予測誤差分散下で π が上昇、高分散下で低下）

PV-RNN で再構成すべき latent 変数は prediction μ_l, error ε_l, precision π_l。
これらの時系列を `PCState` として保持し、後段の forward model に渡す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


@dataclass
class PCState:
    """シミュレーション全期間の latent 変数時系列（ground truth）。"""
    x:         list[np.ndarray]   # 各層状態 x_l,        shape=(T, d_l)
    mu:        list[np.ndarray]   # 下層への予測 μ_l,    shape=(T, d_{l-1})
    epsilon:   list[np.ndarray]   # 予測誤差 ε_l,         shape=(T, d_{l-1} or d_l)
    precision: list[np.ndarray]   # 精度 π_l,             shape=(T, d_{l-1})
    drive:     np.ndarray         # sensory drive,        shape=(T, d_0)
    layer_dims: list[int]
    fs: float


@dataclass
class HierarchicalPCModel:
    """
    階層的 predictive coding ネットワーク。

    Parameters
    ----------
    layer_dims : Sequence[int]
        各層の次元 [d_0, d_1, ..., d_L]。d_0 は sensory drive 次元。
    fs : float
        サンプリング周波数 [Hz]。
    time_constants_ms : Sequence[float]
        各層 (l=1..L) の時定数 [ms]。上位層ほど長い時定数を推奨。
    leak : Sequence[float]
        各層の leak 係数 λ_l。
    precision_init : Sequence[float]
        各層 l=1..L の初期精度（スカラー, broadcastで対角）。
    precision_lr : float
        精度の学習率 η_π。0 なら固定精度。
    precision_kappa : float
        精度更新の安定化項 κ。
    nonlinearity : Callable
        予測写像 g。tanh 推奨。
    weight_scale : float
        生成重み W_l の初期化スケール。
    seed : Optional[int]
        乱数シード（重み初期化）。
    """
    layer_dims: Sequence[int] = (3, 6, 4)
    fs: float = 1000.0
    time_constants_ms: Sequence[float] = (20.0, 80.0)
    leak: Sequence[float] = (0.05, 0.02)
    precision_init: Sequence[float] = (1.0, 0.5)
    precision_lr: float = 0.01
    precision_kappa: float = 1e-2
    nonlinearity: Callable[[np.ndarray], np.ndarray] = _tanh
    weight_scale: float = 0.6
    seed: Optional[int] = 1

    def __post_init__(self):
        self.L = len(self.layer_dims) - 1
        assert self.L >= 1, "階層は 1 層以上必要"
        assert len(self.time_constants_ms) == self.L
        assert len(self.leak) == self.L
        assert len(self.precision_init) == self.L

        rng = np.random.default_rng(self.seed)
        # W_l: (d_{l-1}, d_l)  -- 上層 l が下層 l-1 を予測する生成行列
        self.W = []
        for l in range(1, self.L + 1):
            d_lower = self.layer_dims[l - 1]
            d_upper = self.layer_dims[l]
            W = rng.standard_normal((d_lower, d_upper)) * (
                self.weight_scale / np.sqrt(d_upper)
            )
            self.W.append(W)

        self.dt = 1.0 / self.fs
        # 離散化された積分ゲイン η_l = dt / τ_l
        self.eta = np.array(
            [self.dt / (tc * 1e-3) for tc in self.time_constants_ms]
        )

    # ------------------------------------------------------------------
    def simulate(self, drive: np.ndarray) -> PCState:
        """
        sensory drive を入力として階層ダイナミクスを積分する。

        Parameters
        ----------
        drive : np.ndarray
            shape = (T, d_0)

        Returns
        -------
        PCState
        """
        T, d0 = drive.shape
        assert d0 == self.layer_dims[0], (
            f"drive 次元 {d0} != d_0={self.layer_dims[0]}"
        )

        # 状態初期化
        x = [np.zeros(self.layer_dims[l]) for l in range(1, self.L + 1)]
        pi = [
            np.ones(self.layer_dims[l - 1]) * self.precision_init[l - 1]
            for l in range(1, self.L + 1)
        ]

        # 履歴保持
        x_hist = [np.zeros((T, self.layer_dims[l])) for l in range(1, self.L + 1)]
        mu_hist = [
            np.zeros((T, self.layer_dims[l - 1])) for l in range(1, self.L + 1)
        ]
        eps_hist = [
            np.zeros((T, self.layer_dims[l - 1])) for l in range(1, self.L + 1)
        ]
        pi_hist = [
            np.zeros((T, self.layer_dims[l - 1])) for l in range(1, self.L + 1)
        ]

        g = self.nonlinearity

        for t in range(T):
            # 下層 observation（l=1 は sensory drive）
            lower_obs = [drive[t]] + x  # [x_0, x_1, ..., x_L]
            # 予測と誤差
            mu_t, eps_t = [], []
            for l in range(self.L):
                mu_l = g(self.W[l] @ x[l])          # predicts lower_obs[l]
                eps_l = lower_obs[l] - mu_l
                mu_t.append(mu_l)
                eps_t.append(eps_l)

            # 状態更新（Euler）
            new_x = []
            for l in range(self.L):
                # bottom-up PE (weighted by π of below layer)
                bu = self.W[l].T @ (pi[l] * eps_t[l])
                # top-down PE (from above layer): ε_{l+1} は x_l を説明する誤差
                if l + 1 < self.L:
                    td = pi[l + 1] * eps_t[l + 1]  # 上層の ε, 次元 = d_l
                else:
                    # 最上層は prior 0 からの偏差を収縮項として扱う
                    td = self.precision_init[l] * 0.1 * x[l]
                dx = self.eta[l] * (bu - td - self.leak[l] * x[l])
                new_x.append(x[l] + dx)

            # 精度更新（動的）
            if self.precision_lr > 0:
                new_pi = []
                for l in range(self.L):
                    target = 1.0 / (eps_t[l] ** 2 + self.precision_kappa)
                    # クリップして発散防止
                    target = np.clip(target, 0.05, 50.0)
                    new_pi_l = pi[l] + self.precision_lr * (target - pi[l])
                    new_pi.append(new_pi_l)
                pi = new_pi

            # 履歴記録
            for l in range(self.L):
                x_hist[l][t] = x[l]
                mu_hist[l][t] = mu_t[l]
                eps_hist[l][t] = eps_t[l]
                pi_hist[l][t] = pi[l]

            x = new_x

        return PCState(
            x=x_hist,
            mu=mu_hist,
            epsilon=eps_hist,
            precision=pi_hist,
            drive=drive,
            layer_dims=list(self.layer_dims),
            fs=self.fs,
        )

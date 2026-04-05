"""
Oddball 刺激系列ジェネレータ
============================

標準刺激 (standard) を高頻度で、逸脱刺激 (deviant) を低頻度で
呈示するオドボール課題の刺激列を生成する。

各刺激は連続値特徴ベクトル（例：音高・音強度・持続時間の符号）を持ち、
階層的 predictive coding モデルのボトム層への駆動入力となる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class StimulusFeature:
    """単一刺激の特徴ベクトル表現。"""
    feature: np.ndarray          # shape = (d_feature,)
    is_deviant: bool
    stim_type: int               # 0 = standard, 1,2,... = deviant variants
    onset_sample: int            # サンプル単位の刺激オンセット
    duration_samples: int


@dataclass
class OddballSequence:
    """
    オドボール刺激系列を生成するクラス。

    Parameters
    ----------
    n_trials : int
        試行（刺激）数。
    deviant_prob : float
        逸脱刺激の出現確率 (0 < p < 0.5 推奨)。
    fs : float
        サンプリング周波数 [Hz]。
    isi_sec : float
        刺激間インターバル（オンセット間隔） [秒]。
    stim_duration_sec : float
        刺激持続 [秒]。
    standard_feature : np.ndarray
        標準刺激の特徴ベクトル。
    deviant_features : Sequence[np.ndarray]
        逸脱刺激の特徴ベクトル（複数種類可）。
    min_standards_before_deviant : int
        逸脱間の最小標準連続数（連続逸脱の抑制）。
    seed : Optional[int]
        乱数シード。
    """
    n_trials: int = 400
    deviant_prob: float = 0.15
    fs: float = 1000.0
    isi_sec: float = 0.6
    stim_duration_sec: float = 0.1
    standard_feature: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )
    deviant_features: Sequence[np.ndarray] = field(
        default_factory=lambda: (np.array([0.0, 1.0, 0.0]),)
    )
    min_standards_before_deviant: int = 2
    seed: Optional[int] = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self.d_feature = self.standard_feature.shape[0]
        for f in self.deviant_features:
            assert f.shape[0] == self.d_feature, "feature 次元不一致"
        self.isi_samples = int(round(self.isi_sec * self.fs))
        self.dur_samples = int(round(self.stim_duration_sec * self.fs))
        self.total_samples = self.isi_samples * self.n_trials
        self.stimuli = self._generate_sequence()

    def _generate_sequence(self) -> list[StimulusFeature]:
        seq: list[StimulusFeature] = []
        standards_since_deviant = self.min_standards_before_deviant
        for i in range(self.n_trials):
            if (
                standards_since_deviant >= self.min_standards_before_deviant
                and self._rng.random() < self.deviant_prob
                and i < self.n_trials - 1
            ):
                dev_idx = self._rng.integers(0, len(self.deviant_features))
                feat = self.deviant_features[dev_idx]
                stim = StimulusFeature(
                    feature=feat.copy(),
                    is_deviant=True,
                    stim_type=int(dev_idx) + 1,
                    onset_sample=i * self.isi_samples,
                    duration_samples=self.dur_samples,
                )
                standards_since_deviant = 0
            else:
                stim = StimulusFeature(
                    feature=self.standard_feature.copy(),
                    is_deviant=False,
                    stim_type=0,
                    onset_sample=i * self.isi_samples,
                    duration_samples=self.dur_samples,
                )
                standards_since_deviant += 1
            seq.append(stim)
        return seq

    def drive_signal(self) -> np.ndarray:
        """
        連続時間の sensory drive を返す。

        Returns
        -------
        drive : np.ndarray
            shape = (total_samples, d_feature)
            刺激区間のみ特徴ベクトルが立ち上がる（矩形＋指数減衰）。
        """
        T = self.total_samples
        drive = np.zeros((T, self.d_feature), dtype=float)
        # 指数減衰カーネル（刺激誘発 envelope）
        tau = max(self.dur_samples * 0.5, 1.0)
        t_k = np.arange(self.dur_samples * 3)
        kernel = np.exp(-t_k / tau)
        for stim in self.stimuli:
            start = stim.onset_sample
            end = min(start + kernel.shape[0], T)
            k_len = end - start
            drive[start:end] += stim.feature[None, :] * kernel[:k_len, None]
        return drive

    def event_table(self) -> dict:
        """解析時に使う event 情報を dict で返す。"""
        return {
            "onset_sample": np.array([s.onset_sample for s in self.stimuli]),
            "is_deviant": np.array([s.is_deviant for s in self.stimuli]),
            "stim_type": np.array([s.stim_type for s in self.stimuli]),
            "fs": self.fs,
            "isi_samples": self.isi_samples,
        }

"""
pcsim: Predictive-Coding simulation for PV-RNN validation
=========================================================

階層的 predictive coding を ground truth とした仮想 ECoG 信号生成器。
オドボール課題下で prediction / prediction-error / precision の
latent dynamics を既知量として保持したまま多チャネル ECoG を合成する。

主要コンポーネント
------------------
- OddballSequence         : 標準/逸脱刺激系列の生成
- HierarchicalPCModel     : 階層的予測符号化の ground-truth dynamics
- ECoGForwardModel        : latent source → ECoG 電極への順モデル
- PCSimulator             : 上記を統合したトップレベル API
"""

from .oddball import OddballSequence, StimulusFeature
from .hierarchical_pc import HierarchicalPCModel, PCState
from .forward_model import ECoGForwardModel
from .simulator import PCSimulator, SimulationResult

__all__ = [
    "OddballSequence",
    "StimulusFeature",
    "HierarchicalPCModel",
    "PCState",
    "ECoGForwardModel",
    "PCSimulator",
    "SimulationResult",
]

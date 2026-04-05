# sleepsim: Synthetic PSG Data Generator for Sleep Digital Twin

睡眠デジタルツインモデル（HyperNet-Sleep）のテスト・開発用に、
仮想的なPSG（ポリソムノグラフィ）データを生成するPythonパッケージです。

## 背景と構想

### 最終目標: HyperNet-Sleep Digital Twin

安静時fMRI（rsfMRI）の機能的結合（FC）行列を**個人特性**として入力し、
HyperNetworkがMain-Net（RNN / Variational RNN）の重みを出力、
そのMain-Netが**睡眠中のPSG生理信号時系列**を生成する
——というアーキテクチャのsleep digital twinモデルを構築することが最終目標です。

```
rsfMRI FC matrix ──→ HyperNet ──→ Main-Net weights (RNN/VRNN)
                                        │
                                        ▼
                                  PSG時系列生成
                                  (EEG, EMG, EOG, ECG)
```

### 本パッケージの役割

実データなしでもモデルのend-to-endテストを可能にするため、
**制御可能な因果関係**を持つ仮想データを生成します:

```
個人特性パラメータ ──→ 仮想FC matrix（HyperNetの入力として使用）
        │
        ▼
   睡眠ステージ遷移 ──→ PSG時系列（Main-Netのターゲットとして使用）
```

- 個人特性パラメータ（δ波パワー、紡錘波密度、REM潜時など）が既知
- それらがFC matrixとPSG信号の両方に因果的に影響
- モデルが「FC → 個人特性 → PSG」を正しく学習できているかを検証可能

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/y66yamay/sleepsim.git
cd sleepsim

# 依存パッケージのインストール
pip install numpy scipy

# 可視化も使う場合
pip install matplotlib
```

## クイックスタート

### 基本的なデータ生成

```python
from sleepsim import SleepDataGenerator

# 50被験者, 256Hz, 8時間の合成PSGデータを生成
gen = SleepDataGenerator(
    n_subjects=50,
    sampling_rate=256,
    duration_hours=8.0,
    n_roi=20,       # FC matrixのROI数
    seed=42,        # 再現性のためのシード
)

# メモリ効率的な逐次生成（推奨）
for subject_data in gen.generate_subject_iter():
    fc_matrix = subject_data["fc_matrix"]      # (20, 20) - HyperNetの入力
    psg_data  = subject_data["psg_data"]        # (8, ~7.4M) - Main-Netのターゲット
    hypnogram = subject_data["hypnogram"]       # (960,) - 睡眠ステージラベル
    traits    = subject_data["traits"]          # SubjectTraits - 個人特性

    # ... モデルの学習に使用 ...
```

### 疾患群データの生成

```python
from sleepsim import SleepDataGenerator

# 各条件を個別に生成（condition引数で指定）
healthy_gen  = SleepDataGenerator(n_subjects=20, condition="healthy",  seed=42)
rbd_gen      = SleepDataGenerator(n_subjects=20, condition="rbd",      seed=43)
osa_gen      = SleepDataGenerator(n_subjects=20, condition="osa",      seed=44)
insomnia_gen = SleepDataGenerator(n_subjects=20, condition="insomnia", seed=45)

# 群間比較用にデータを統合
for gen in [healthy_gen, rbd_gen, osa_gen, insomnia_gen]:
    for data in gen.generate_subject_iter():
        condition = data["traits"].condition  # "healthy", "rbd", etc.
        fc = data["fc_matrix"]               # 疾患特異的なFC異常を反映
        psg = data["psg_data"]               # 疾患特異的な信号特徴を含む
```

### RNN学習用のエポックバッチ取得

```python
# 特定の被験者・エポックのみを効率的に取得
batch = gen.generate_epoch_batch(
    subject_indices=[0, 1, 2, 3, 4],
    epoch_indices=[0, 10, 20, 50, 100, 200],
)
# batch["psg_epochs"]   : (5, 6, 8, 7680)  - (subjects, epochs, channels, samples)
# batch["stages"]       : (5, 6)            - ステージラベル
# batch["fc_matrices"]  : (5, 20, 20)       - FC行列
```

### 高速プロトタイピング用（ダウンサンプル）

```python
# 64Hzにダウンサンプルしてメモリ・計算コストを1/4に
gen = SleepDataGenerator(
    n_subjects=10,
    sampling_rate=256,
    downsample_factor=4,  # 実効64Hz
)
```

## アーキテクチャ

### パッケージ構成

```
sleepsim/
├── __init__.py        # パッケージエクスポート
├── traits.py          # 個人特性パラメータの定義・生成
├── conditions.py      # 疾患条件プロファイル（RBD, OSA, 不眠症）
├── stages.py          # 睡眠ステージ遷移エンジン（マルコフ連鎖）
├── channels.py        # PSG信号生成（EEG, EMG, EOG, ECG, 呼吸, SpO2）
├── fc_matrix.py       # 個人特性→FC行列の埋め込み
├── generator.py       # 統合オーケストレータ
└── utils.py           # フィルタ、ノイズ生成等のユーティリティ

examples/
├── generate_cohort.py      # 基本的な使用例（5被験者, 1時間）
├── validate_data.py        # 健常群の検証可視化（10種プロット）
└── validate_conditions.py  # 疾患間比較の検証可視化（12種プロット）
```

### データフロー

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  traits.py  │────→│  stages.py   │────→│  channels.py    │
│             │     │              │     │                 │
│ 12個の個人  │     │ 時間非均一   │     │ ステージ依存の  │
│ 特性パラメ  │     │ マルコフ連鎖 │     │ 信号合成        │
│ ータを定義  │     │ →ヒプノグラム│     │ →PSG時系列      │
└──────┬──────┘     └──────────────┘     └─────────────────┘
       │                    ↑                      ↑
       │            ┌───────┴──────┐       ┌───────┴───────┐
       │            │conditions.py │       │ conditions.py │
       │            │ ステージ遷移 │       │ 信号特性変調  │
       │            │ の疾患変調   │       │ (RBD/OSA/不眠)│
       │            └──────────────┘       └───────────────┘
       │
       │            ┌──────────────┐
       └───────────→│ fc_matrix.py │
                    │ ROI重み行列  │
                    │ + 疾患FC摂動 │
                    │ →FC行列      │
                    └──────────────┘
```

## 個人特性パラメータ（12次元）

睡眠生理学で個人差が知られている特徴量をパラメータ化しています:

| パラメータ | 範囲 | 生理学的意味 |
|---|---|---|
| `delta_power` | 0.5 - 2.0 | N3でのδ波（0.5-4 Hz）振幅の強さ |
| `spindle_density` | 0.3 - 1.5 | N2での紡錘波の出現頻度（回/30秒エポック） |
| `spindle_frequency` | 11.0 - 16.0 Hz | 紡錘波の中心周波数（slow/fast spindle） |
| `alpha_power` | 0.5 - 1.5 | 覚醒時のα波（8-12 Hz）振幅 |
| `rem_latency` | 60 - 120 分 | 入眠から最初のREM出現までの時間 |
| `sleep_cycle_duration` | 80 - 110 分 | NREM-REMサイクルの周期長 |
| `rem_density` | 0.3 - 1.0 | REM中の急速眼球運動の密度 |
| `muscle_atonia_depth` | 0.5 - 1.0 | REM中の筋弛緩の深さ（1.0=完全弛緩） |
| `sleep_efficiency` | 0.80 - 0.98 | 記録時間中の睡眠割合 |
| `n3_fraction` | 0.10 - 0.25 | 総睡眠時間中のN3（徐波睡眠）割合 |
| `heart_rate_mean` | 55 - 75 bpm | 睡眠中の平均心拍数 |
| `hrv_amplitude` | 0.3 - 1.0 | 心拍変動の大きさ（迷走神経緊張の指標） |

パラメータ間には生理学的に妥当な**相関構造**があります:
- `delta_power` ↔ `n3_fraction`: 正の相関（δ波が強い人はN3が多い）
- `sleep_efficiency` ↔ `rem_latency`: 負の相関（睡眠効率が高い人はREM潜時が短い）
- `rem_density` ↔ `muscle_atonia_depth`: 正の相関
- `heart_rate_mean` ↔ `hrv_amplitude`: 負の相関

## 睡眠ステージモデル

30秒エポック単位の**時間非均一マルコフ連鎖**で睡眠ステージを生成します。

**遷移確率は以下の要因で変調されます:**

1. **ウルトラディアンサイクル位相**: サイクル前半はN3優位、後半はREM優位
2. **サイクル番号**: 夜の前半のサイクルはN3が多く、後半はREMが増加
3. **個人特性**: `n3_fraction`、`sleep_efficiency`、`rem_density`等が遷移確率に影響
4. **REM潜時制約**: `rem_latency`分が経過するまでREM遷移を抑制
5. **禁止遷移**: Wake→N3直接、Wake→REM直接、N3→REM直接は不可

**生成されるステージ分布（8時間、10被験者平均）:**
- N2: 45-53% / N3: 13-22% / REM: 17-25% / Wake: 5-8% / N1: 3-8%

## 臨床条件（疾患群シミュレーション）

4つの条件をサポートしています。各条件は特性パラメータ、ステージ遷移、
信号特性、FC行列の全レイヤーに影響します。

### 対応条件

| 条件 | 説明 | `condition=` |
|---|---|---|
| **健常対照** | 正常な睡眠パターン | `"healthy"` |
| **RBD** | REM睡眠行動障害 | `"rbd"` |
| **OSA** | 閉塞性睡眠時無呼吸 | `"osa"` |
| **不眠症** | 原発性不眠症 | `"insomnia"` |

### 各疾患のPSG特徴

#### RBD（REM睡眠行動障害）

α-synuclein病変（パーキンソン病、レビー小体型認知症の前駆症状）に関連する睡眠障害。

| 層 | 変調内容 |
|---|---|
| **特性** | `muscle_atonia_depth` ↓↓（0.35-0.49 vs 健常0.70-0.84） |
| **ステージ** | 睡眠構造はほぼ保持。REM微小断片化のみ |
| **EMG** | **REM without atonia**: REM中のtonic EMG持続（floor=0.5）+ 大振幅の相動性バースト + 夢内容行動化アーチファクト |
| **FC** | 脳幹-皮質間FC低下、皮質下ネットワーク内FC低下 |

#### OSA（閉塞性睡眠時無呼吸）

上気道閉塞による反復的な呼吸停止と覚醒を特徴とする睡眠障害。

| 層 | 変調内容 |
|---|---|
| **特性** | `sleep_efficiency` ↓, `n3_fraction` ↓↓, `heart_rate_mean` ↑, `hrv_amplitude` ↓ |
| **ステージ** | AHI≈30の頻回覚醒（30回/時）、N3ほぼ消失（1-2%）、Wake 40%超 |
| **呼吸** | Resp_effort: 無呼吸（10-40秒）→回復呼吸。SpO2: V字型脱飽和（最大8%低下） |
| **EEG** | 無呼吸後の覚醒バースト（α+β帯域） |
| **EMG** | いびき振動アーチファクト（N1/N2中） |
| **FC** | 前頭-頭頂DMN FC低下、島皮質FC低下 |

#### 不眠症（Primary Insomnia）

入眠困難・中途覚醒・早朝覚醒を特徴とする睡眠障害。

| 層 | 変調内容 |
|---|---|
| **特性** | `sleep_efficiency` ↓↓（0.67-0.74）, `alpha_power` ↑, `delta_power` ↓ |
| **ステージ** | 入眠潜時2.5倍延長、WASO増加（3倍）、5.5時間以降の早朝覚醒 |
| **EEG** | NREM中のβパワー1.8倍増（皮質過覚醒）、α波混入（alpha intrusion） |
| **FC** | 前頭覚醒系ネットワーク過結合、前頭-皮質下FC増加 |

### 生成されるステージ分布の比較（8時間平均）

| | Wake | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| **Healthy** | 6-8% | 5-7% | 45-53% | 13-22% | 17-26% |
| **RBD** | 9-13% | 5-7% | 48-53% | 16-20% | 14-18% |
| **OSA** | 40-44% | 10-15% | 34-35% | 0-2% | 8-14% |
| **Insomnia** | 27-30% | 7-12% | 38-42% | 7-10% | 13-14% |

## EEGチャンネル構成（設定可能）

EEGチャンネルは国際10-20法の標準位置から自由に選択できます。デフォルトは
`["C3", "C4"]` の2チャンネルで、非EEGチャンネル（EOG×2, EMG, ECG, 呼吸, SpO2）
と合わせて計8チャンネルとなります。

### 使用例

```python
from sleepsim import SleepDataGenerator

# デフォルト（2ch EEG + 6ch 非EEG = 計8ch）
gen = SleepDataGenerator(n_subjects=10)

# 臨床標準の6ch EEG構成
gen = SleepDataGenerator(
    n_subjects=10,
    eeg_channels=["F3", "F4", "C3", "C4", "O1", "O2"],
)  # 計12チャンネル

# フル10-20法（19ch EEG）
full_1020 = ["Fp1","Fp2","F3","F4","F7","F8","Fz",
             "C3","C4","Cz","T3","T4","T5","T6",
             "P3","P4","Pz","O1","O2"]
gen = SleepDataGenerator(n_subjects=10, eeg_channels=full_1020)  # 計25チャンネル

# 単一チャンネル（Cz のみ）
gen = SleepDataGenerator(n_subjects=10, eeg_channels=["Cz"])
```

### 利用可能なEEGチャンネル

| 領域 | チャンネル名 |
|---|---|
| 前頭極 | `Fp1`, `Fp2` |
| 前頭 | `F3`, `F4`, `F7`, `F8`, `Fz` |
| 中心 | `C3`, `C4`, `Cz` |
| 側頭 | `T3`, `T4`, `T5`, `T6` |
| 頭頂 | `P3`, `P4`, `Pz` |
| 後頭 | `O1`, `O2`, `Oz` |
| 耳介/乳様突起 | `A1`, `A2`, `M1`, `M2` |

### トポグラフィによる信号特性の変調

各チャンネルは**生理学的に妥当な空間分布**で信号が生成されます（`EEG_TOPOGRAPHY`）:

| 帯域 | 空間分布の特徴 |
|---|---|
| **δ波（0.5-4 Hz）** | 前頭部で最大（Fz, F3/F4）、後頭部で最小 |
| **α波（8-12 Hz）** | 後頭部で最大（O1/O2, 1.5×）、前頭部で最小（Fp1/2, 0.3×） |
| **紡錘波** | 中心部（C3/C4, Cz）で最大、末梢で減弱 |
| **β波（16-30 Hz）** | 前頭側頭で優位（F7/F8） |

さらに、チャンネル間には**空間的距離に応じた相関構造**が導入されます
（近接チャンネルほど高相関）。

### カスタマイズ

```python
from sleepsim.channels import EEG_TOPOGRAPHY, AVAILABLE_EEG_CHANNELS

print("Available channels:", AVAILABLE_EEG_CHANNELS)
print("O1 topography:", EEG_TOPOGRAPHY["O1"])
# {'delta': 0.7, 'theta': 0.9, 'alpha': 1.5, 'spindle': 0.45, 'beta': 0.75}
```

## PSG信号チャンネル（デフォルト8チャンネル）

| チャンネル | 生成方法 |
|---|---|
| **EEG_C3, EEG_C4** | 帯域別（δ/θ/α/σ/β）フィルタノイズの重み付き合成。N2では紡錘波バースト・K複合を重畳。1/fノイズフロア付き。不眠症ではβ増強+α混入、OSAでは覚醒バースト |
| **EOG_L, EOG_R** | 覚醒時: サッカード。NREM: 緩徐眼球運動。REM: 急速共役眼球運動バースト（`rem_density`依存） |
| **EMG_chin** | 広帯域ノイズ（20-100 Hz）。振幅: Wake > N1 > N2 > N3 >> REM。RBDではREM中のtonic EMG持続+夢内容行動化。OSAではいびきアーチファクト |
| **ECG** | QRS波形テンプレート配置。R-R間隔は`heart_rate_mean`ベースにステージ依存変調 + `hrv_amplitude`による呼吸性変動 |
| **Resp_effort** | 呼吸努力信号（正弦波ベース）。OSAでは無呼吸イベント（呼吸停止→回復呼吸）を含む |
| **SpO2** | 酸素飽和度（0-1スケール）。健常≈0.97。OSAでは無呼吸に伴うV字型脱飽和イベント |

## 仮想FC行列

個人特性の12次元ベクトルを、20×20の対称FC行列（相関行列様）に埋め込みます。

**方法:**
1. 20個の仮想ROI（前頭葉4, 中心/頭頂4, 側頭4, 後頭4, 皮質下4）を定義
2. 各ROIは特定の特性パラメータに重み付け（例: 前頭葉ROIは`delta_power`と`spindle_density`に強く関連）
3. 重み行列 × 正規化特性ベクトル → ROI活性ベクトル → 外積 → FC行列
4. 小さなノイズを加算し、対角を1.0に固定

**HyperNetにとっての意味:**
- FC行列は12次元多様体上にあり（12特性 → rank-12の構造）
- 特性の小さな変化 → FC行列の小さな変化（滑らかな写像）
- HyperNetはこの12次元構造を学習して特性を復元可能

## データの保存と読み込み

生成した合成データをディスクに保存するためのIO関数が用意されています。

### 一括保存（推奨）

```python
from sleepsim import SleepDataGenerator

gen = SleepDataGenerator(n_subjects=10, condition="healthy", seed=42)

# NPZ形式で保存（デフォルト）
gen.save_to_disk("output/healthy_dataset", fmt="npz")

# EDF形式で保存（pyedflib必要: pip install pyedflib）
gen.save_to_disk("output/healthy_dataset_edf", fmt="edf")
```

出力されるディレクトリ構造:

```
output/healthy_dataset/
├── metadata.json                      # データセット全体のメタデータ
├── traits.csv                         # 全被験者の特性パラメータ一覧
└── subjects/
    ├── subject_0000.npz               # PSG+FC+hypnogram+traits（NPZ）
    ├── subject_0000_hypnogram.csv     # イベント形式のヒプノグラム
    ├── subject_0001.npz
    ├── subject_0001_hypnogram.csv
    └── ...
```

### 個別のIO関数

```python
from sleepsim import (
    save_subject_npz, load_subject_npz,
    save_hypnogram_csv, save_traits_csv,
)

# 1被験者だけ保存
for data in gen.generate_subject_iter():
    save_subject_npz(data, f"subject_{data['traits'].subject_id:03d}.npz")
    save_hypnogram_csv(data["hypnogram"], f"hypnogram_{data['traits'].subject_id:03d}.csv")

# 読み込み
loaded = load_subject_npz("subject_000.npz")
# loaded["psg_data"], loaded["hypnogram"], loaded["fc_matrix"], loaded["traits"]

# 特性テーブルのみCSV保存
save_traits_csv(gen.subjects, "traits.csv")
```

### NPZ ファイルの内容

| キー | 形状 | 型 | 説明 |
|---|---|---|---|
| `psg_data` | (n_channels, n_samples) | float32 | PSG時系列 |
| `hypnogram` | (n_epochs,) | int8 | 睡眠ステージラベル |
| `fc_matrix` | (n_roi, n_roi) | float32 | FC行列 |
| `trait_vector` | (12,) | float64 | 個人特性パラメータ |
| `channel_names` | (n_channels,) | str | チャンネル名 |
| `sampling_rate` | scalar | int | サンプリングレート |

### EDF 形式

EDF（European Data Format）はPSG研究の標準形式です。EDGE Browser、EDFbrowser、MNE-Pythonなど各種ツールで開けます。

- `pyedflib` が必要: `pip install pyedflib`
- ヒプノグラムはEDFアノテーションとして各ステージ遷移に埋め込まれます
- SpO2は保存時に%スケール（0-100）に自動変換されます

```bash
python examples/export_data.py   # 全フォーマットのデモを実行
```

## 検証可視化

```bash
# 基本的な使用例（5被験者, 1時間, 4プロット）
python examples/generate_cohort.py

# 健常群の包括的な検証（10被験者, 8時間, 10プロット）
python examples/validate_data.py

# 疾患間比較の検証（4条件 x 8被験者, 8時間, 12プロット）
python examples/validate_conditions.py
```

### `validate_data.py`（健常群の検証、10プロット）

| プロット | 検証内容 |
|---|---|
| 01_hypnograms | 全被験者のヒプノグラム比較 |
| 02_stage_distribution | 被験者別ステージ比率（正常範囲の帯付き） |
| 03_eeg_psd | ステージ別EEGパワースペクトル密度 + 帯域パワー |
| 04_per_stage_signals | 5ステージ × 8チャンネルの波形スニペット |
| 05_fc_matrices | 被験者間FC行列比較 + 被験者間変動 |
| 06_trait_distributions | 12パラメータのヒストグラム |
| 07_trait_correlations | パラメータ間相関行列（設定した相関構造の検証） |
| 08_sleep_cycle_dynamics | 夜間を通じたN3/REM比率の推移 |
| 09_emg_by_stage | ステージ別EMG振幅（REM atonia検証） |
| 10_ecg_by_stage | ステージ別推定心拍数（自律神経変調検証） |

### `validate_conditions.py`（疾患間比較、12プロット）

| プロット | 検証内容 |
|---|---|
| 01_hypnograms_by_condition | 4条件の代表的ヒプノグラム比較 |
| 02_stage_distribution | 条件別ステージ比率（群間比較） |
| 03_sleep_architecture | 睡眠効率・N3%・REM%・WASO・入眠潜時の箱ひげ図 |
| 04_eeg_psd_by_condition | N2/REM時のEEG PSD条件間比較 |
| 05_band_power_heatmap | 帯域パワー × ステージ × 条件のヒートマップ |
| 06_emg_by_condition | **条件別×ステージ別EMG振幅（RBD: REM without atonia検証）** |
| 07_respiratory_spo2 | **呼吸努力+SpO2の条件間比較（OSA: 無呼吸+脱飽和検証）** |
| 08_fc_by_condition | 条件別平均FC行列 |
| 09_fc_diff_from_healthy | **疾患FC - 健常FCの差分行列（ROIグループ境界付き）** |
| 10_trait_shift | 健常からの特性パラメータシフト量 |
| 11_onset_waso | **入眠潜時+WASO比較（不眠症検証）** |
| 12_condition_signal_gallery | REM中の4チャンネル × 4条件の信号比較 |

## API リファレンス

### `SleepDataGenerator`（メインクラス）

```python
SleepDataGenerator(
    n_subjects=50,        # 被験者数
    sampling_rate=256,    # サンプリングレート (Hz)
    duration_hours=8.0,   # 記録時間
    epoch_sec=30.0,       # エポック長 (秒)
    n_roi=20,             # FC行列のROI数
    seed=42,              # 乱数シード
    downsample_factor=1,  # ダウンサンプル倍率
    condition="healthy",  # "healthy", "rbd", "osa", "insomnia"
)
```

| メソッド | 戻り値 | 用途 |
|---|---|---|
| `generate_subject_iter()` | `Iterator[dict]` | 1被験者ずつ逐次生成（メモリ効率的） |
| `generate_dataset()` | `dict` | 全被験者を一括生成（小規模データセット向け） |
| `generate_epoch_batch(subject_indices, epoch_indices)` | `dict` | 指定エポックのみ生成（RNN学習用） |
| `generate_subject(traits)` | `dict` | 1被験者のデータを生成 |

### `SubjectTraits`（個人特性）

```python
from sleepsim import generate_subjects

# 相関構造付きで被験者を生成
subjects = generate_subjects(n=100, condition="healthy", seed=42)

# 特性ベクトルへの変換
trait_vector = subjects[0].to_vector()            # (12,) 生の値
trait_norm   = subjects[0].to_normalized_vector()  # (12,) [0,1]正規化
```

### `FCMatrixGenerator`（FC行列生成）

```python
from sleepsim import FCMatrixGenerator

fc_gen = FCMatrixGenerator(n_roi=20, noise_scale=0.05)
fc_matrix = fc_gen.generate(subjects[0])        # (20, 20)
fc_batch  = fc_gen.generate_batch(subjects[:10]) # (10, 20, 20)
```

### メモリの目安

| 設定 | 1被験者あたり | 50被験者 |
|---|---|---|
| 256 Hz, 8h, 8ch, float32 | ~235 MB | ~11.7 GB |
| 128 Hz, 8h, 8ch, float32 | ~118 MB | ~5.9 GB |
| 256 Hz, 8h, downsample=4, float32 | ~59 MB | ~2.9 GB |

大規模データセットでは `generate_subject_iter()` を使い、
1被験者ずつ処理（保存）することを推奨します。

## 依存パッケージ

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4（可視化のみ）

## ライセンス

TBD

## 参考文献

**睡眠生理・PSGスコアリング:**
- Berry, R. B., et al. (2017). *The AASM Manual for the Scoring of Sleep and Associated Events*. — 睡眠ステージ判定基準
- Carskadon, M. A., & Dement, W. C. (2011). Normal human sleep: an overview. — 正常睡眠構造
- De Gennaro, L., & Ferrara, M. (2003). Sleep spindles: an overview. *Sleep Medicine Reviews*. — 紡錘波の個人差

**RBD:**
- Postuma, R. B., et al. (2019). Risk and predictors of dementia and parkinsonism in RBD. *Neurology*. — RBDの臨床的意義
- Iranzo, A., et al. (2006). REM sleep behavior disorder and neurodegeneration. *Current Neurology and Neuroscience Reports*. — RBDとFC異常

**OSA:**
- Park, B., et al. (2016). Aberrant resting-state functional connectivity in OSA. *Sleep*. — OSAのFC異常
- Dempsey, J. A., et al. (2010). Pathophysiology of sleep apnea. *Physiological Reviews*. — 無呼吸の生理メカニズム

**不眠症:**
- Riemann, D., et al. (2010). The hyperarousal model of insomnia. *Sleep Medicine Reviews*. — 過覚醒モデル
- Li, Y., et al. (2014). Altered resting-state FC in insomnia. *Sleep*. — 不眠症のFC異常

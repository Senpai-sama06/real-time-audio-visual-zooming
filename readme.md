# Real-Time Audio Visual Zooming

This repository contains a comprehensive pipeline for simulating and evaluating 2-microphone "Audio Zoom" systems. The project investigates various beamforming techniques to overcome the traditional $M-1$ Degree of Freedom (DoF) limit, enabling separation of multiple sound sources using only 2 microphones.

## Overview

The repository is organized into several experiment folders, each exploring different beamforming approaches:

1. **`loaded_mvdr/`** - MVDR with diagonal loading for zoom control
2. **`masked_mvdr_exp/`** - Mask-driven MVDR with oracle masks
3. **`max_snr/`** - Maximum SNR (GEV) beamformer with field-of-view control
4. **`neural_model+mvdr/`** - Neural network mask estimation combined with MVDR

---

## Experiment 1: Loaded MVDR (`loaded_mvdr/`)

### What it does
Implements MVDR beamforming with diagonal loading (sigma parameter) to control beam width and aggressiveness. Lower sigma values create narrow, aggressive beams (zoom in), while higher sigma values create wider, gentler beams (zoom out).

### Key Files
- `world_builder.py` - Generates 2-source mixture from LJ Speech dataset
- `zoomable_mvdr.py` - Applies MVDR beamforming with configurable sigma
- `sigma_vs_bw.py` - Analyzes relationship between sigma and beamwidth

### How to Run

1. **Generate test mixture:**
   ```bash
   cd experiments/loaded_mvdr
   python world_builder.py
   ```
   Output: `mixture_2_sources.wav`

2. **Run MVDR beamforming:**
   ```bash
   python zoomable_mvdr.py
   ```
   Edit `SIGMA` in the script to control zoom:
   - `SIGMA = 1e-9` → Narrow beam (Zoom In)
   - `SIGMA = 1.0` → Wide beam (Zoom Out)
   
   Output: `output_2src_sigma_{SIGMA:.1e}.wav` and beam pattern plot

3. **Analyze sigma vs beamwidth:**
   ```bash
   python sigma_vs_bw.py
   ```
   Output: `sigma_vs_beamwidth.png` showing null depth vs sigma relationship

### Results
- Beam pattern visualizations saved as PNG files
- Separated audio output for different sigma values
- Null depth analysis showing trade-off between aggressiveness and stability

---

## Experiment 2: Masked MVDR (`masked_mvdr_exp/`)

### What it does
Implements mask-driven MVDR beamforming. Uses a time-frequency mask to build a clean noise covariance matrix before applying MVDR. Tests both hard geometric masks (phase-based) and oracle masks (ground truth).

### Key Files
- `world.py` - Generates 3-source mixture with reference files
- `masked_mvdr.py` - Applies hard geometric mask (phase-based)
- `oracle_debug.py` - Uses perfect oracle mask (ground truth)
- `validate.py` - Calculates SIR/SDR metrics

### How to Run

1. **Generate test data:**
   ```bash
   cd experiments/masked_mvdr_exp
   python world.py
   ```
   Outputs: `mixture_3_sources.wav`, `target_reference.wav`, `interference_reference.wav`

2. **Run masked MVDR (hard mask):**
   ```bash
   python masked_mvdr.py
   ```
   Output: `output_masked_mvdr.wav`

3. **Run oracle MVDR (best case):**
   ```bash
   python oracle_debug.py
   ```
   Output: `output_oracle.wav`

4. **Validate results:**
   ```bash
   python validate.py
   ```
   Reports SIR and SDR improvements

### Results
- Oracle mask achieves ~36dB SIR (upper bound)
- Demonstrates that beamforming engine works when mask is accurate
- Hard geometric masks show limitations due to phase ambiguities

---

## Experiment 3: Maximum SNR Beamformer (`max_snr/`)

### What it does
Implements Maximum SNR (Generalized Eigenvalue) beamformer with Field-of-View (FOV) control. Scans all angles, builds separate covariance matrices for sources inside and outside the FOV, then solves a generalized eigenvalue problem.

### Key Files
- `test_world.py` - Generates test mixture
- `test_maxsnr.py` - Implements Max-SNR (GEV) beamformer
- `oracle.py` - Oracle version using ground truth masks
- `test_validate.py` - Validation script

### How to Run

1. **Generate test data:**
   ```bash
   cd experiments/max_snr
   python test_world.py
   ```
   Outputs: `test_mixture.wav`, `test_target_ref.wav`, `test_interferer_ref.wav`

2. **Run Max-SNR beamformer:**
   ```bash
   python test_maxsnr.py
   ```
   Edit `FOV_WIDTH_DEG` in the script to control FOV:
   - `FOV_WIDTH_DEG = 3.0` → Narrow FOV (Zoom In)
   - `FOV_WIDTH_DEG = 20.0` → Wide FOV (Zoom Out)
   
   Output: `output_maxsnr_fov_{FOV_WIDTH_DEG}deg.wav`

3. **Run oracle version:**
   ```bash
   python oracle.py
   ```
   Output: `output_oracle_gev.wav`

4. **Validate:**
   ```bash
   python test_validate.py
   ```
   Reports SIR/SDR metrics

### Results
- FOV-based control for spatial selectivity
- GEV solver maximizes SNR ratio between FOV and non-FOV regions
- Comparison with oracle shows potential performance bounds

---

## Experiment 4: Neural Model + MVDR (`neural_model+mvdr/`)

### What it does
Trains a shallow CNN to predict time-frequency masks from spatial-spectral features, then uses these predicted masks in an MVDR beamformer. This replaces the oracle mask with a learned mask estimator.

### Key Files
- `world_builder.py` - Data generation utilities
- `neural_model_train.py` - Trains CNN mask estimator
- `inference_and_deployment.py` - Deploys trained model + MVDR

### How to Run

1. **Train the neural mask estimator:**
   ```bash
   cd experiments/neural_model+mvdr
   python neural_model_train.py
   ```
   - Downloads LJ Speech dataset automatically
   - Trains for 50 epochs (adjustable)
   - Output: `mask_estimator.pth`

2. **Deploy and validate:**
   ```bash
   # First, generate test data using world.py from masked_mvdr_exp
   # Copy mixture_3_sources.wav, target_reference.wav, interference_reference.wav to this folder
   
   python inference_and_deployment.py
   ```
   Output: `output_neural_mvdr.wav` with SIR/SDR metrics printed

### Results
- Neural network learns to estimate masks from log-magnitude and IPD features
- Combines learning with classical beamforming
- Shows bridge between oracle performance and practical mask estimation

---

## Dependencies

Install required packages:

```bash
pip install numpy scipy soundfile librosa matplotlib torch torchaudio kagglehub mir-eval tqdm
```

## Dataset

All experiments use the **LJ Speech Dataset**, which is automatically downloaded via Kaggle Hub when needed. No manual download required.

## Common Workflow

1. **Data Generation**: Run the `world.py` or `world_builder.py` script to create test mixtures
2. **Beamforming**: Run the main beamforming script (e.g., `zoomable_mvdr.py`, `masked_mvdr.py`)
3. **Validation**: Run validation scripts to measure SIR/SDR improvements
4. **Analysis**: Check output audio files and generated plots

## Notes

- All audio processing uses 16kHz sample rate
- STFT parameters: 512 FFT size, 256 hop (50% overlap)
- Mic spacing: 0.04m (unless otherwise specified in individual experiments)
- Target angle: 90° (broadside) in most experiments

## Key Findings

- **Oracle masks** demonstrate ~36dB SIR (upper performance bound)
- **Diagonal loading** (sigma) provides intuitive zoom control
- **Hard geometric masks** fail due to phase ambiguities (~4dB SIR)
- **Neural masks** bridge gap between oracle and geometric approaches
- **Max-SNR** beamformer offers FOV-based spatial control

---

## License

This project is provided as-is for research and educational purposes.

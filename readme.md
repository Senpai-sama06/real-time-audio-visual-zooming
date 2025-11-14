## Mask-based Minimum Variance Distortionless Response Audio Beamforming

This repository contains a Python pipeline for simulating and evaluating a 2-microphone "Audio Zoom" system. The project's goal is to overcome the $M-1$ Degree of Freedom (DoF) limit, which traditionally prevents a 2-mic array from separating multiple (N) sound sources. We investigate a Mask-Driven MVDR (Minimum Variance Distortionless Response) beamformer, where a time-frequency mask is used to generate a clean Noise Covariance Matrix ($\mathbf{R}_{noise}$) before the beamforming math is applied.

### 1. How to Run the Simulation

This pipeline uses a 3-script process to validate the "Oracle" scenario (the theoretical best-case performance).

world.py: Generates the test data. It downloads the LJ Speech dataset, selects N+1 random files, and mixes them into a 2-channel file (mixture_...wav) based on 3D far-field physics. Crucially, it also saves the clean "answer key" files: target_reference.wav and interference_reference.wav.

oracle_debug.py: This is the core solver. It loads the mixture_...wav and "cheats" by using the reference files to create a perfect Ideal Binary Mask (IBM). It then uses this mask to build a clean $\mathbf{R}_{noise}$ and runs the MVDR beamformer.

validate.py: This script scores the result by comparing the output of the beamformer against the reference files to calculate the true Signal-to-Interference Ratio (SIR).

To run the experiment, ensure all dependencies from requirements.txt are installed, then execute the scripts in order.

### 2. Project Progress & Key Findings

Our initial experiments with a "blind" (standard) MVDR beamformer failed when faced with 2+ interferers, confirming the $M-1$ DoF limit. This led us to the Mask-Driven approach.

Our first attempt used a "Heuristic Mask" based on simple geometric phase-difference (i.e., "Target = 0 phase diff"). This method failed, yielding a minimal ~4.4dB SIR improvement due to phase ambiguities (spatial aliasing) and signal overlaps.

The breakthrough came from the "Oracle Test." By using the ground-truth files to create a perfect mask, the exact same MVDR pipeline achieved a massive 36.24dB SIR. This proves that the beamforming "engine" is sound and that the entire problem is now reduced to a high-fidelity mask estimation task.

###  3. Future Work

The 36dB Oracle result provides a clear upper bound for system performance. The logical next step is to replace the "Oracle" with a predictive model.

Future work will involve designing and training a Deep Learning model (e.g., a U-Net or Bi-LSTM) to predict the Ideal Binary Mask. This model will be trained on a "spatial-spectral" feature set, combining the Log-Magnitude Spectrogram (to identify what a sound is) with the Inter-channel Phase Difference (IPD) (to identify where a sound is).
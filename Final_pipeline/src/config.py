# src/config.py
import os

# --- PATHS ---
# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SIM_DIR = os.path.join(DATA_DIR, "simulated")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# --- AUDIO PHYSICS ---
FS = 16000
C_SPEED = 343.0
N_FFT = 1024
HOP_LEN = 512
WIN_SIZE = 32000 

# --- SIMULATION DEFAULTS ---
ROOM_DIM = [4.9, 4.9, 4.9]
RT60_TARGET = 0.5
SIR_TARGET_DB = 0

# --- GEOMETRY ---
# Mic locations (for simulation)
MIC_LOCS_SIM = [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]]
# Mic distance (for inference physics)
MIC_DIST = 0.08
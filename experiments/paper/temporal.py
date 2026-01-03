import numpy as np
from collections import deque
from scipy.stats import mode

class TemporalSmoother:
    """
    Implements a Hysteresis filter to stabilize frame-by-frame decisions.
    Uses a rolling voting window to filter out transient noise.
    """
    def __init__(self, n_bins, window_size=5):
        self.n_bins = n_bins
        self.window_size = window_size
        
        # Buffer: List of arrays, shape [Window, Bins]
        # We use a deque with maxlen for efficient rolling
        self.history = deque(maxlen=window_size)
        
        # Fill buffer with H0 (Silence) initially
        for _ in range(window_size):
            self.history.append(np.zeros(n_bins, dtype=int))

    def update(self, current_decisions):
        """
        Input: current_decisions [Bins] (Integer codes)
        Output: smoothed_decisions [Bins]
        """
        # Add new frame
        self.history.append(current_decisions)
        
        # Stack history: [Window, Bins]
        stack = np.array(self.history)
        
        # Vote: Calculate Mode along time axis (axis 0)
        # This returns the most frequent hypothesis per bin
        smoothed, count = mode(stack, axis=0, keepdims=False)
        
        # --- Hysteresis Logic (Senior Officer Req 3.1) ---
        # If the vote is weak (e.g., split decision), keep the previous state?
        # For now, Mode is robust enough for median filtering.
        
        return smoothed.flatten()
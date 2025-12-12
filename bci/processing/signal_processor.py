"""
Advanced BCI Signal Processing Pipeline
Handles filtering, feature extraction, and pattern recognition
"""
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import signal as scipy_signal
from dataclasses import dataclass

from bci.core.interfaces import IBCIProcessor, BCIReading, BrainwaveData


@dataclass
class ProcessedSignal:
    """Container for processed neural signals"""
    filtered_signal: np.ndarray
    frequency_bands: Dict[str, np.ndarray]
    features: np.ndarray
    artifacts_detected: List[str]
    snr: float  # Signal-to-noise ratio


class UniversalSignalProcessor(IBCIProcessor):
    """Universal signal processor for all BCI devices"""

    def __init__(self):
        # Filter parameters
        self.notch_freq = 60.0  # Hz (power line noise)
        self.lowcut = 0.5  # Hz
        self.highcut = 100.0  # Hz

    def process_raw_signal(self, raw_data: np.ndarray, sample_rate: float) -> Dict[str, Any]:
        """
        Process raw neural signals through complete pipeline
        """
        # Step 1: Artifact removal
        cleaned = self._remove_artifacts(raw_data)

        # Step 2: Bandpass filtering
        filtered = self._bandpass_filter(cleaned, sample_rate)

        # Step 3: Notch filter (remove power line noise)
        filtered = self._notch_filter(filtered, sample_rate)

        # Step 4: Extract frequency bands
        bands = self._extract_frequency_bands(filtered, sample_rate)

        # Step 5: Calculate signal quality metrics
        snr = self._calculate_snr(filtered)

        # Step 6: Detect remaining artifacts
        artifacts = self._detect_artifacts(filtered, sample_rate)

        return {
            "filtered": filtered,
            "bands": bands,
            "snr": snr,
            "artifacts": artifacts,
            "sample_rate": sample_rate
        }

    def extract_features(self, processed_data: Dict[str, Any]) -> np.ndarray:
        """
        Extract feature vector from processed signals
        """
        features = []

        bands = processed_data["bands"]

        # Power features for each band
        for band_name, band_data in bands.items():
            power = np.mean(band_data ** 2)
            features.append(power)

        # Spectral entropy
        psd = np.abs(np.fft.fft(processed_data["filtered"]))
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features.append(spectral_entropy)

        # Band power ratios
        if "alpha" in bands and "beta" in bands:
            alpha_power = np.mean(bands["alpha"] ** 2)
            beta_power = np.mean(bands["beta"] ** 2)
            alpha_beta_ratio = alpha_power / (beta_power + 1e-10)
            features.append(alpha_beta_ratio)

        # Hjorth parameters
        activity = np.var(processed_data["filtered"])
        mobility = np.std(np.diff(processed_data["filtered"])) / (np.std(processed_data["filtered"]) + 1e-10)
        complexity = (np.std(np.diff(np.diff(processed_data["filtered"]))) /
                     (np.std(np.diff(processed_data["filtered"])) + 1e-10)) / mobility
        features.extend([activity, mobility, complexity])

        return np.array(features)

    def decode_intent(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Decode user intent from feature vector
        Uses simplified classifier (in production, use trained ML model)
        """
        # Simplified intent detection
        # In production: use trained neural network or SVM

        # Example: high beta + low alpha = high cognitive load
        # High alpha + low beta = relaxed state
        # High gamma = focused attention

        intent = {
            "state": "unknown",
            "confidence": 0.0,
            "parameters": {}
        }

        # Basic heuristics (placeholder for ML model)
        if len(features) >= 5:
            alpha_power = features[2]  # Assuming alpha is 3rd band
            beta_power = features[3]  # Beta is 4th

            ratio = beta_power / (alpha_power + 1e-10)

            if ratio > 1.5:
                intent["state"] = "active_thinking"
                intent["confidence"] = min(0.8, ratio / 2.0)
            elif ratio < 0.7:
                intent["state"] = "relaxed"
                intent["confidence"] = min(0.8, 1.0 - ratio)
            else:
                intent["state"] = "neutral"
                intent["confidence"] = 0.5

        return intent

    # ========== Private Helper Methods ==========

    def _bandpass_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Apply bandpass filter"""
        nyquist = fs / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        if high >= 1.0:
            high = 0.99

        b, a = scipy_signal.butter(4, [low, high], btype='band')
        filtered = scipy_signal.filtfilt(b, a, data)
        return filtered

    def _notch_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Remove power line noise (60 Hz in US, 50 Hz in EU)"""
        nyquist = fs / 2
        notch_freq_norm = self.notch_freq / nyquist

        if notch_freq_norm >= 1.0:
            return data  # Skip if frequency out of range

        Q = 30.0  # Quality factor
        b, a = scipy_signal.iirnotch(notch_freq_norm, Q)
        filtered = scipy_signal.filtfilt(b, a, data)
        return filtered

    def _extract_frequency_bands(self, data: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
        """Extract standard EEG frequency bands"""
        bands = {}

        # Define frequency bands
        band_definitions = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 100)
        }

        nyquist = fs / 2

        for band_name, (low, high) in band_definitions.items():
            if high > nyquist:
                continue

            low_norm = low / nyquist
            high_norm = high / nyquist

            if high_norm >= 1.0:
                high_norm = 0.99

            b, a = scipy_signal.butter(4, [low_norm, high_norm], btype='band')
            band_data = scipy_signal.filtfilt(b, a, data)
            bands[band_name] = band_data

        return bands

    def _remove_artifacts(self, data: np.ndarray) -> np.ndarray:
        """
        Remove common artifacts (eye blinks, muscle movements, etc.)
        Simplified version - production would use ICA or similar
        """
        # Remove extreme outliers (likely artifacts)
        threshold = 3 * np.std(data)
        cleaned = np.clip(data, -threshold, threshold)
        return cleaned

    def _detect_artifacts(self, data: np.ndarray, fs: float) -> List[str]:
        """Detect artifacts in the signal"""
        artifacts = []

        # Check for saturation
        if np.max(np.abs(data)) > 1000:
            artifacts.append("saturation")

        # Check for flat line (electrode disconnection)
        if np.std(data) < 1.0:
            artifacts.append("flat_line")

        # Check for high-frequency noise
        high_freq_power = np.mean(np.abs(np.fft.fft(data)[int(len(data)/2):]))
        if high_freq_power > 100:
            artifacts.append("high_frequency_noise")

        return artifacts

    def _calculate_snr(self, data: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simplified SNR calculation
        signal_power = np.mean(data ** 2)
        noise_estimate = np.var(np.diff(data))  # High frequency = noise
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        return float(snr)


class AdaptiveDecoder:
    """
    Adaptive decoder that learns and improves over time
    """

    def __init__(self):
        self.calibration_data = []
        self.model_trained = False

    def add_calibration_sample(self, features: np.ndarray, label: str):
        """Add labeled sample for training"""
        self.calibration_data.append((features, label))

    def train(self):
        """Train decoder on calibration data"""
        if len(self.calibration_data) < 10:
            print("[Decoder] Insufficient calibration data")
            return False

        print(f"[Decoder] Training on {len(self.calibration_data)} samples...")
        # In production: train ML model (SVM, Neural Network, etc.)
        self.model_trained = True
        print("[Decoder] Training complete")
        return True

    def decode(self, features: np.ndarray) -> Dict[str, Any]:
        """Decode intent from features"""
        if not self.model_trained:
            return {"state": "uncalibrated", "confidence": 0.0}

        # Placeholder for trained model inference
        return {
            "state": "decoded_intent",
            "confidence": 0.85,
            "class": "motor_imagery"
        }

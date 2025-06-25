import librosa
import numpy as np
import parselmouth


def extract_all_features(file_path):
    try:
        snd = parselmouth.Sound(file_path)
        pitch = snd.to_pitch()
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        # === Pitch-based features ===
        fo_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")
        fo_min = parselmouth.praat.call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        fo_max = parselmouth.praat.call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

        # === Jitter DDP (3 × RAP) ===
        try:
            jitter_rap = parselmouth.praat.call([snd, point_process], "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ddp = 3 * jitter_rap
        except:
            jitter_ddp = 0.0
        

        # === NHR ===
        try:
            nhr = parselmouth.praat.call(snd, "Get harmonics-to-noise ratio", 0.0, 0.0)
        except:
            nhr = 0.0
        

        # === Librosa features ===
        y, sr = librosa.load(file_path)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spread1 = np.mean(spectral_centroid)
        spread2 = np.std(spectral_centroid)

        # PPE approximation (spectral flatness)
        ppe = np.mean(librosa.feature.spectral_flatness(y=y))

        # Final 9 features in correct order:
        features = np.array([
            ppe,           # PPE
            fo_mean,       # MDVP:Fo(Hz)
            spread1,       # spread1
            fo_min,        # MDVP:Flo(Hz)
            jitter_ddp,    # Jitter:DDP
            fo_max,        # MDVP:Fhi(Hz) ← ✅ Now correctly added
            spread2,       # spread2
            nhr            # NHR
        ])

        return features.reshape(1, -1)

    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

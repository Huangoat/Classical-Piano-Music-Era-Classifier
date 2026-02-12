import numpy as np
import pandas as pd
import pretty_midi
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALER_DIR = (PROJECT_ROOT / "data" / "scaler" / "scaler.pkl").as_posix()
scaler = joblib.load(SCALER_DIR)

def extract_features_from_midi(file_path):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        notes = [n for instr in pm.instruments for n in instr.notes]

        if not notes:
            return None

        pitches = np.array([n.pitch for n in notes])
        velocities = np.array([n.velocity for n in notes])
        durations = np.array([n.end - n.start for n in notes])
        total_time = pm.get_end_time()
        if total_time == 0:
            return None

        features = {
            "pitch_range": pitches.max() - pitches.min(),
            "avg_pitch": pitches.mean(),
            "std_pitch": pitches.std(),
            "note_density": len(notes) / total_time,
            "polyphony_level": np.count_nonzero(pm.get_piano_roll(fs=10), axis=0).mean(),
            "vel_mean": velocities.mean(),
            "vel_var": velocities.var(),
            "articulation": durations.mean(),
            "tempo": pm.estimate_tempo(),
            "highest_note": pitches.max(),
            "lowest_note": pitches.min(),
            "chromaticism": -np.sum((pm.get_pitch_class_histogram()/ (pm.get_pitch_class_histogram().sum()+1e-6)) * np.log2((pm.get_pitch_class_histogram()/ (pm.get_pitch_class_histogram().sum()+1e-6))+1e-6))
        }

        pc_histogram = pm.get_pitch_class_histogram()
        for i, val in enumerate(pc_histogram):
            features[f"pc_{i}"] = val
        df_features = pd.DataFrame([features])
        df_features = df_features.reindex(columns=scaler.feature_names_in_, fill_value=0)
        df_scaled = pd.DataFrame(scaler.transform(df_features), columns=scaler.feature_names_in_)

        return df_scaled

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


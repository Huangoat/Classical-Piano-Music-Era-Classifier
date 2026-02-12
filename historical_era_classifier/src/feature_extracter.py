import os
import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "Segmented Classical Dataset"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "features_unnormalized.csv"

def extract_midi_features(file_path):
    try:
        pm = pretty_midi.PrettyMIDI(file_path)
        notes = []
        for instrument in pm.instruments:
            notes.extend(instrument.notes)
            
        if not notes:
            return None

        pitches = [n.pitch for n in notes]
        velocities = [n.velocity for n in notes]
        durations = [n.end - n.start for n in notes]
        
        # Timbral Features
        pitch_range = max(pitches) - min(pitches)
        avg_pitch = np.mean(pitches)
        std_pitch = np.std(pitches)
        note_density = len(notes) / pm.get_end_time()
        piano_roll = pm.get_piano_roll(fs=10) 
        notes_per_step = np.count_nonzero(piano_roll, axis=0)
        polyphony_level = np.mean(notes_per_step)

        # Rhythmic Features
        vel_mean = np.mean(velocities)
        vel_var = np.var(velocities)
        avg_articulation = np.mean(durations)
        tempo = pm.estimate_tempo()

        #Pitch Content Features
        highest_note = max(pitches)
        lowest_note = min(pitches)

        pc_histogram = pm.get_pitch_class_histogram()
        hist_norm = pc_histogram / (np.sum(pc_histogram) + 1e-6)
        chromaticism_index = -np.sum(hist_norm * np.log2(hist_norm + 1e-6))

        # Feature dictionary
        features = {
            "pitch_range": pitch_range,
            "avg_pitch": avg_pitch,
            "std_pitch": std_pitch,
            "note_density": note_density,
            "polyphony_level": polyphony_level,
            "vel_mean": vel_mean,
            "vel_var": vel_var,
            "articulaftion": avg_articulation,
            "tempo": tempo,
            "highest_note": highest_note,
            "lowest_note": lowest_note,
            "chromaticism": chromaticism_index
        }
        
        for i, val in enumerate(pc_histogram):
            features[f"pc_{i}"] = val

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
data = []
files = list(Path(SOURCE_DIR).rglob("*.mid")) + list(Path(SOURCE_DIR).rglob("*.midi"))

for file_path in tqdm(files):
    features = extract_midi_features(str(file_path))
    
    if features:
        features["filename"] = file_path.name

        rel_path = file_path.relative_to(SOURCE_DIR)
        features["label"] = rel_path.parts[0]

        data.append(features)

df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Completed Feature Extracting")

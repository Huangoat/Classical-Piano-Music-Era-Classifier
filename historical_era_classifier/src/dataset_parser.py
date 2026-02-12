import os
import pretty_midi
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "Raw Classical Dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "Segmented Classical Dataset"
SAMPLE_LENGTH = 20 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_count = 0
segment_count = 0

for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith((".mid", ".midi")):
            file_path = os.path.join(root, file)
            
            # Get the relative path to maintain structure
            rel_path = os.path.relpath(root, SOURCE_DIR)
            dest_dir = os.path.join(OUTPUT_DIR, rel_path)
            
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            try:
                # Loading MIDI file
                midi_data = pretty_midi.PrettyMIDI(file_path)
                total_duration = midi_data.get_end_time()
                
                # Calculating full number of segments
                num_segments = int(total_duration // SAMPLE_LENGTH)
                
                for i in range(num_segments):
                    start_time = i * SAMPLE_LENGTH
                    end_time = (i + 1) * SAMPLE_LENGTH
                    total_notes_in_chunk = 0
                    chunk = pretty_midi.PrettyMIDI()
                    for instrument in midi_data.instruments:
                        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
                        
                        for note in instrument.notes:
                            if note.start >= start_time and note.end <= end_time:
                                new_note = pretty_midi.Note(
                                    velocity=note.velocity,
                                    pitch=note.pitch,
                                    # Normalizing time
                                    start=note.start - start_time,
                                    end=note.end - start_time
                                )
                                new_instrument.notes.append(new_note)
                                total_notes_in_chunk += 1
                        
                        chunk.instruments.append(new_instrument)

                    # Only save if the segment has above 8 notes
                    if total_notes_in_chunk > 8: 
                        chunk_name = f"{Path(file).stem}_chunk{i}.mid"
                        chunk.write(os.path.join(dest_dir, chunk_name))
                        segment_count += 1

                file_count += 1
                print(f"{file} ({num_segments} segments)")

            except Exception as e:
                print(f"Error processing file")

print(f"Completed file segmenting")
print(f"Original files processed: {file_count}")
print(f"Total 20s segments created: {segment_count}")
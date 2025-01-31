from pydub import AudioSegment

def mix_wav_files(file1_path, file2_path, output_path):
    # Load audio files
    audio1 = AudioSegment.from_wav(file1_path)
    audio2 = AudioSegment.from_wav(file2_path)
    
    # Overlay the tracks (mix them together)
    combined = audio1.overlay(audio2)
    
    # Export the result
    combined.export(output_path, format='wav')


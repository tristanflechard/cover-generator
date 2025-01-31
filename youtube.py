from pytubefix import YouTube
from pydub import AudioSegment
import os
import re


def sanitize_filename(filename):
    """Replace invalid characters in filenames with underscores."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def youtube_to_wav(youtube_url, output_folder="yt_audios"):
    # Download audio from YouTube
    yt = YouTube(youtube_url, "WEB")
    title = yt.title
    sanitized_title = sanitize_filename(yt.title)
    stream = yt.streams.filter(only_audio=True).first()
    audio_file = stream.download(filename="temp_audio")
    output_path = os.path.join(output_folder, f"{sanitized_title}")
    print(f"Output path: {output_path}")

    # Convert to WAV using pydub
    audio = AudioSegment.from_file(audio_file)
    audio.export(output_path, format="wav")

    print(f"Converted to {output_path}")
    os.remove(audio_file)
    return sanitized_title

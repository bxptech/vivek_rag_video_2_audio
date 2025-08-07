# src/video_to_audio.py
import subprocess
import glob
import os

VIDEO_FOLDER = "data/VIDEO/*.mp4"  # Videos outside project folder
AUDIO_FOLDER = "data/audio"

def transcribe_video2_audio():
    """
    Extract audio from all videos in VIDEO_FOLDER and save in AUDIO_FOLDER.
    """
    os.makedirs(AUDIO_FOLDER, exist_ok=True)

    video_files = glob.glob(VIDEO_FOLDER)
    if not video_files:
        print("❌ No video files found.")
        return

    for video_file in video_files:
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        audio_file = os.path.join(AUDIO_FOLDER, f"{base_name}.mp3")

        if os.path.exists(audio_file):
            print(f"⏭️ Skipping {video_file} (already converted).")
            continue

        print(f"🎥 Processing {video_file} → 🎵 {audio_file}")
        subprocess.run(
            ["ffmpeg", "-i", video_file, "-q:a", "0", "-map", "a", audio_file],
            check=True
        )

    print("✅ All videos converted to audio.")

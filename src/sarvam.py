# import os
# import ffmpeg
# import math
# from sarvamai import SarvamAI
# from dotenv import load_dotenv

# load_dotenv()

# API_KEY = os.getenv("SARVAM_API_KEY")
# MODEL = os.getenv("MODEL", "saarika:v2.5")
# LANGUAGE_CODE = os.getenv("LANGUAGE_CODE", "te-IN")
# CHUNK_DURATION_SECONDS = int(os.getenv("CHUNK_DURATION_SECONDS", 30))

# client = SarvamAI(api_subscription_key=API_KEY)

# def get_audio_duration(audio_path):
#     probe = ffmpeg.probe(audio_path)
#     return float(probe['format']['duration'])

# def split_audio(audio_path, chunk_duration=CHUNK_DURATION_SECONDS):
#     duration = get_audio_duration(audio_path)
#     num_chunks = math.ceil(duration / chunk_duration)
#     chunk_paths = []
#     for i in range(num_chunks):
#         start_time = i * chunk_duration
#         output_chunk = f"chunk_{i}.wav"
#         (
#             ffmpeg
#             .input(audio_path, ss=start_time, t=chunk_duration)
#             .output(output_chunk, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
#             .overwrite_output()
#             .run(quiet=True)
#         )
#         chunk_paths.append(output_chunk)
#     return chunk_paths

# def transcribe_chunk(chunk_path):
#     with open(chunk_path, "rb") as f:
#         response = client.speech_to_text.transcribe(
#             file=f,
#             model=MODEL,
#             language_code=LANGUAGE_CODE
#         )
#     return getattr(response, "transcript", "").strip()
# def transcribe_audio(audio_path, save_to_file=True):
#     chunk_files = split_audio(audio_path)
#     all_texts = []

#     for chunk in chunk_files:
#         print(f"🎙️ Transcribing {chunk}")
#         try:
#             text = transcribe_chunk(chunk)
#             print(f"✅ {text}")
#             all_texts.append(text)
#         except Exception as e:
#             print(f"❌ Error: {e}")
#             all_texts.append("")
#         finally:
#             if os.path.exists(chunk):
#                 os.remove(chunk)

#     # Save transcript to file
#     if save_to_file:
#         transcript_path = os.path.splitext(audio_path)[0] + "_transcript.txt"
#         with open(transcript_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(all_texts))
#         print(f"💾 Transcript saved to: {transcript_path}")

#     return all_texts
import os
import ffmpeg
import math
from sarvamai import SarvamAI
from dotenv import load_dotenv
from googletrans import Translator  # For translation
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")
MODEL = os.getenv("MODEL", "saarika:v2.5")
CHUNK_DURATION_SECONDS = int(os.getenv("CHUNK_DURATION_SECONDS", 30))

# SarvamAI client
client = SarvamAI(api_subscription_key=API_KEY)

# Google Translator client
translator = Translator()

def get_audio_duration(audio_path):
    """Get the total duration of the audio file in seconds."""
    probe = ffmpeg.probe(audio_path)
    return float(probe['format']['duration'])

def split_audio(audio_path, chunk_duration=CHUNK_DURATION_SECONDS):
    """Split audio file into smaller chunks."""
    duration = get_audio_duration(audio_path)
    num_chunks = math.ceil(duration / chunk_duration)
    chunk_paths = []
    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_chunk = f"chunk_{i}.wav"
        (
            ffmpeg
            .input(audio_path, ss=start_time, t=chunk_duration)
            .output(output_chunk, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        chunk_paths.append(output_chunk)
    return chunk_paths

def transcribe_chunk(chunk_path):
    """Transcribe a single chunk and return original text."""
    with open(chunk_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model=MODEL,
            language_code="unknown"  # Let SarvamAI detect supported language
        )
    return getattr(response, "transcript", "").strip()


def translate_to_english(text):
    """Translate any text to English."""
    if not text.strip():
        return ""
    return GoogleTranslator(source="auto", target="en").translate(text)
 
def transcribe_audio(audio_path, save_to_file=True):
    """
    Transcribe audio to text (any language) → translate to English → return English text.
    This English text can then be used for chunking + embeddings.
    """
    chunk_files = split_audio(audio_path)
    all_english_texts = []

    for chunk in chunk_files:
        print(f"🎙️ Transcribing {chunk}")
        try:
            original_text = transcribe_chunk(chunk)
            english_text = translate_to_english(original_text)

            print(f"✅ Original: {original_text}")
            print(f"🌐 English: {english_text}")

            all_english_texts.append(english_text)
        except Exception as e:
            print(f"❌ Error: {e}")
            all_english_texts.append("")
        finally:
            if os.path.exists(chunk):
                os.remove(chunk)

    # Ensure no empty embeddings
    if not any(all_english_texts):
        all_english_texts = ["No transcription available"]

    # Save transcript if needed
    if save_to_file:
        transcript_path = os.path.splitext(audio_path)[0] + "_english_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_english_texts))
        print(f"💾 Transcript saved to: {transcript_path}")

    return all_english_texts  # Send this to chunking + embeddings

if __name__ == "__main__":
    input_audio = "data/audio/your_audio_file.mp3"
    english_texts = transcribe_audio(input_audio)

    # Example: send to embeddings pipeline
    # chunks = chunk_text("\n".join(english_texts))
    # vector_store.save_embeddings(embed(chunks), chunks)

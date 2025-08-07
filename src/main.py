# main.py (or your existing RAG script)
import os
import glob

from video2audio import transcribe_video2_audio
import sarvam
import embedder
import vector_store
import rag_query

AUDIO_FOLDER = "data/audio/*.mp3"

def build_kb_from_all_files():
    all_chunks = []
    files = glob.glob(AUDIO_FOLDER)
    if not files:
        print("❌ No audio files found.")
        return

    for audio_file in files:
        print(f"🎙️ Processing {audio_file}")
        texts = sarvam.transcribe_audio(audio_file)
        chunks = [t for t in texts if t.strip()]
        all_chunks.extend(chunks)

    vectors = embedder.embed_chunks(all_chunks)
    vector_store.save_embeddings(vectors, all_chunks)
    print(f"✅ Knowledge base built from {len(files)} audio files with Gemini embeddings.")

def interactive_chat():
    index, chunks = vector_store.load_embeddings()
    print("\n💬 Interactive RAG Audio Chat (type 'exit' to quit)\n")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = rag_query.rag_query(q, embedder.embed_query, index, chunks)
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    # Step 1: Extract audio from videos
    transcribe_video2_audio()

    # Step 2: Build KB if not already built
    if not os.path.exists("data/embeddings/faiss.index"):
        build_kb_from_all_files()

    # Step 3: Start chat
    interactive_chat()

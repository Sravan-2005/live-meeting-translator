"""
Real-time Multilingual Speech Translation with Speaker Identification
Enhanced version with speaker recognition using voice embeddings

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

import os, sys, time, json, queue, pathlib
import numpy as np
import sounddevice as sd
import webrtcvad
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from vosk import Model as VoskModel, KaldiRecognizer
from scipy.spatial.distance import cosine
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables for Hugging Face authentication
load_dotenv()

# Authenticate with Hugging Face if token is available
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    print("🔐 Authenticating with Hugging Face...")
    login(token=hf_token)
    print("✅ Authentication successful!\n")
else:
    print("⚠️  No HUGGINGFACE_TOKEN found in .env file.")
    print("    Speaker identification may require authentication.")
    print("    Create a .env file with: HUGGINGFACE_TOKEN=your_token_here\n")


# ██████████████████████████████████████████████████████████████████████████████
#                              CONSTANTS & CONFIG
# ██████████████████████████████████████████████████████████████████████████████

# --- Audio Settings ---
SR = 16_000
DTYPE = "int16"
CHANNELS = 1
FRAME_MS = 20
FRAME_SAMP = SR * FRAME_MS // 1000
VAD_MODE = 2
WHISPER_BUF_SEC = 4.0
WHISPER_BUF_SAMP = int(WHISPER_BUF_SEC * SR)
LOG_PATH = pathlib.Path("captions_translated.jsonl")

# --- Speaker Identification Settings ---
SIMILARITY_THRESHOLD = 0.60  # Cosine similarity threshold for speaker matching
MIN_AUDIO_LENGTH = 2.0  # Minimum audio length (seconds) for reliable speaker ID
SPEAKER_UPDATE_THRESHOLD = 0.70  # Update existing speaker embedding if similarity is high
MAX_SILENCE_FOR_SAME_SPEAKER = 5.0  # Seconds - assume same speaker if gap is short
CONSERVATIVE_NEW_SPEAKER_THRESHOLD = 0.50  # Only create new speaker if ALL similarities below this

# --- Source Language (STT) Config ---
STT_MENU = {
    1: {"name": "Telugu",    "code": "te", "engine": "whisper"},
    2: {"name": "Hindi",     "code": "hi", "engine": "whisper"},
    3: {"name": "Malayalam", "code": "ml", "engine": "whisper"},
    4: {"name": "Thai",      "code": "th", "engine": "whisper"},
    5: {"name": "English",   "code": "en", "engine": "vosk"},
    6: {"name": "Spanish",   "code": "es", "engine": "vosk"},
    7: {"name": "French",    "code": "fr", "engine": "vosk"},
    8: {"name": "Chinese",   "code": "zh", "engine": "vosk"},
}

WHISPER_MODELS = {
    "te": "vasista22/whisper-telugu-tiny",
    "hi": "openai/whisper-small",
    "ml": "openai/whisper-small",
    "th": "openai/whisper-small",
}

VOSK_MODELS = {
    "en": "stt/vosk/models/en/vosk-model-small-en-us-0.15",
    "es": "stt/vosk/models/es/vosk-model-small-es-0.42",
    "fr": "stt/vosk/models/fr/vosk-model-small-fr-0.22",
    "zh": "stt/vosk/models/cn/vosk-model-small-cn-0.22",
}

# --- Target Language (Translation) Config ---
TRANSLATION_MENU = {
    1: {"name": "English",   "code": "eng_Latn"},
    2: {"name": "Hindi",     "code": "hin_Deva"},
    3: {"name": "Telugu",    "code": "tel_Telu"},
    4: {"name": "Spanish",   "code": "spa_Latn"},
    5: {"name": "French",    "code": "fra_Latn"},
    6: {"name": "Thai",      "code": "tha_Thai"},
    7: {"name": "Chinese (Simplified)", "code": "zho_Hans"},
    8: {"name": "Tamil",     "code": "tam_Taml"},
    9: {"name": "Malayalam", "code": "mal_Mlym"},
}

# --- Mapping from STT codes to NLLB Translation codes ---
NLLB_LANG_MAP = {
    "te": "tel_Telu",
    "hi": "hin_Deva",
    "ml": "mal_Mlym",
    "th": "tha_Thai",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
}

def utc_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int(time.time()*1e3)%1000:03d}"


# ██████████████████████████████████████████████████████████████████████████████
#                          SPEAKER IDENTIFICATION
# ██████████████████████████████████████████████████████████████████████████████

class SpeakerIdentifier:
    """
    Identifies speakers using voice embeddings from pyannote/embedding model.
    Maintains a database of known speakers and enrolls new speakers automatically.
    Uses adaptive embedding updates to improve recognition over time.
    """
    
    def __init__(self, similarity_threshold=SIMILARITY_THRESHOLD):
        """
        Initialize the speaker identification system.
        
        Args:
            similarity_threshold: Minimum cosine similarity to match a known speaker
        """
        print("🔊 Loading speaker embedding model (pyannote/embedding)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load the pyannote embedding model
            from pyannote.audio import Model, Inference
            model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=hf_token
            )
            # Use Inference wrapper for better preprocessing
            self.inference = Inference(model, window="whole")
            print(f"✅ Speaker embedding model loaded (device: {self.device})\n")
            self.enabled = True
        except Exception as e:
            print(f"❌ Failed to load speaker embedding model: {e}")
            print("    Continuing without speaker identification...\n")
            self.inference = None
            self.enabled = False
        
        self.similarity_threshold = similarity_threshold
        self.known_speakers = []  # List of (speaker_name, embedding_list) tuples
        self.speaker_count = 0
        self.last_speaker = None  # Track last identified speaker
        self.last_speaker_time = 0  # Timestamp of last identification
    
    def _generate_embedding(self, audio_data):
        """
        Generate a speaker embedding from audio data.
        
        Args:
            audio_data: numpy array of audio samples (int16, 16kHz, mono)
        
        Returns:
            numpy array: Speaker embedding vector
        """
        if not self.enabled:
            return None
        
        try:
            # Convert int16 to float32 and normalize to [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # pyannote Inference expects dict with 'waveform' and 'sample_rate'
            # Use the Inference wrapper which handles preprocessing
            embedding = self.inference({
                'waveform': torch.from_numpy(audio_float).unsqueeze(0).float(),
                'sample_rate': SR
            })
            
            # Inference already returns numpy array, just flatten and normalize
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            embedding = embedding.flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
        except Exception as e:
            print(f"⚠️  Embedding generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _compute_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1, emb2: Numpy arrays of embeddings
        
        Returns:
            float: Similarity score between 0 and 1
        """
        return 1 - cosine(emb1, emb2)
    
    def _update_speaker_embedding(self, speaker_idx, new_embedding, similarity):
        """
        Update speaker's embedding with exponential moving average.
        
        Args:
            speaker_idx: Index of speaker in known_speakers list
            new_embedding: New embedding to incorporate
            similarity: Similarity score with existing embedding
        """
        speaker_name, embeddings_list = self.known_speakers[speaker_idx]
        
        # Add new embedding to list (keep last 5 embeddings)
        embeddings_list.append(new_embedding)
        if len(embeddings_list) > 5:
            embeddings_list.pop(0)
        
        # Update the stored embeddings
        self.known_speakers[speaker_idx] = (speaker_name, embeddings_list)
    
    def _get_average_embedding(self, embeddings_list):
        """
        Get average embedding from list of embeddings.
        
        Args:
            embeddings_list: List of numpy arrays
        
        Returns:
            numpy array: Average embedding
        """
        if len(embeddings_list) == 1:
            return embeddings_list[0]
        
        avg_embedding = np.mean(embeddings_list, axis=0)
        # Normalize
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        return avg_embedding
    
    def identify_or_enroll(self, audio_chunk):
        """
        Identify speaker from audio or enroll as new speaker.
        Uses conservative approach to minimize false new speaker creation.
        
        Args:
            audio_chunk: numpy array of audio samples (int16, 16kHz, mono)
        
        Returns:
            str: Speaker name (e.g., "Speaker 1", "Speaker 2")
        """
        if not self.enabled:
            return "Speaker 1"
        
        current_time = time.time()
        
        # Check minimum audio length for reliable identification
        audio_duration = len(audio_chunk) / SR
        
        # If audio is too short AND we recently identified a speaker, assume it's the same
        if audio_duration < MIN_AUDIO_LENGTH:
            if self.last_speaker and (current_time - self.last_speaker_time) < MAX_SILENCE_FOR_SAME_SPEAKER:
                print(f"👤 Short audio, assuming: {self.last_speaker}")
                return self.last_speaker
            elif self.known_speakers:
                # Default to most recent speaker for very short audio
                return self.known_speakers[-1][0]
            return "Speaker 1"
        
        # Generate embedding for this audio
        new_embedding = self._generate_embedding(audio_chunk)
        
        if new_embedding is None:
            return self.last_speaker if self.last_speaker else "Speaker 1"
        
        # If no known speakers, enroll this as first speaker
        if not self.known_speakers:
            self.speaker_count += 1
            speaker_name = f"Speaker {self.speaker_count}"
            self.known_speakers.append((speaker_name, [new_embedding]))
            self.last_speaker = speaker_name
            self.last_speaker_time = current_time
            print(f"👤 [NEW] {speaker_name}")
            return speaker_name
        
        # If only 1 known speaker exists and we're still early in the conversation
        # Be more conservative about creating a second speaker
        if len(self.known_speakers) == 1:
            speaker_name, embeddings_list = self.known_speakers[0]
            avg_embedding = self._get_average_embedding(embeddings_list)
            similarity = self._compute_similarity(new_embedding, avg_embedding)
            
            # Very conservative threshold for creating second speaker
            if similarity > 0.50:
                self._update_speaker_embedding(0, new_embedding, similarity)
                self.last_speaker = speaker_name
                self.last_speaker_time = current_time
                print(f"👤 {speaker_name} (sim: {similarity:.3f})")
                return speaker_name
        
        # Compare against all known speakers using their average embeddings
        best_match_idx = None
        best_similarity = -1
        second_best_similarity = -1
        all_similarities = []
        
        for idx, (speaker_name, embeddings_list) in enumerate(self.known_speakers):
            # Get average embedding for this speaker
            avg_embedding = self._get_average_embedding(embeddings_list)
            
            # Calculate cosine similarity
            similarity = self._compute_similarity(new_embedding, avg_embedding)
            all_similarities.append((idx, speaker_name, similarity))
            
            if similarity > best_similarity:
                second_best_similarity = best_similarity
                best_similarity = similarity
                best_match_idx = idx
            elif similarity > second_best_similarity:
                second_best_similarity = similarity
        
        # Sort for display
        all_similarities.sort(key=lambda x: x[2], reverse=True)
        debug_info = " | ".join([f"{name}: {sim:.2f}" for _, name, sim in all_similarities[:min(3, len(all_similarities))]])
        
        # Calculate separation: how much better is the best match vs second best?
        separation = best_similarity - second_best_similarity if second_best_similarity > 0 else 1.0
        
        # Decision logic: Use best match if above threshold
        if best_similarity >= SIMILARITY_THRESHOLD:
            speaker_name = self.known_speakers[best_match_idx][0]
            
            # Update speaker's embedding collection if similarity is good
            if best_similarity >= SPEAKER_UPDATE_THRESHOLD:
                self._update_speaker_embedding(best_match_idx, new_embedding, best_similarity)
            
            self.last_speaker = speaker_name
            self.last_speaker_time = current_time
            print(f"👤 {speaker_name} ({debug_info})")
            return speaker_name
        
        # Check temporal context - if recent utterance, be very conservative
        if self.last_speaker and (current_time - self.last_speaker_time) < MAX_SILENCE_FOR_SAME_SPEAKER:
            # Find last speaker and check if similarity is reasonable
            for idx, (speaker_name, embeddings_list) in enumerate(self.known_speakers):
                if speaker_name == self.last_speaker:
                    # If similarity is at least above conservative threshold, assume same speaker
                    if best_match_idx == idx and best_similarity > CONSERVATIVE_NEW_SPEAKER_THRESHOLD:
                        self._update_speaker_embedding(idx, new_embedding, best_similarity)
                        self.last_speaker_time = current_time
                        print(f"👤 {speaker_name} [time-based] ({debug_info})")
                        return speaker_name
        
        # CONSERVATIVE: Only create new speaker if ALL similarities are very low
        if best_similarity < CONSERVATIVE_NEW_SPEAKER_THRESHOLD:
            # AND we haven't created a new speaker very recently
            if not hasattr(self, '_last_new_speaker_time') or (current_time - self._last_new_speaker_time) > 10.0:
                self.speaker_count += 1
                speaker_name = f"Speaker {self.speaker_count}"
                self.known_speakers.append((speaker_name, [new_embedding]))
                self.last_speaker = speaker_name
                self.last_speaker_time = current_time
                self._last_new_speaker_time = current_time
                print(f"👤 [NEW] {speaker_name} ({debug_info})")
                return speaker_name
        
        # Fallback: assign to best match even if below threshold
        speaker_name = self.known_speakers[best_match_idx][0]
        self._update_speaker_embedding(best_match_idx, new_embedding, best_similarity)
        self.last_speaker = speaker_name
        self.last_speaker_time = current_time
        print(f"👤 {speaker_name} [fallback] ({debug_info})")
        return speaker_name


# ██████████████████████████████████████████████████████████████████████████████
#                          TRANSLATION COMPONENT
# ██████████████████████████████████████████████████████████████████████████████

class Translator:
    """Translation component using NLLB-200 model."""
    
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🌍 TRANSLATION: Using device: {self.device}")
        print("🌍 TRANSLATION: Loading model... (This may take a moment)")
        self.translator = pipeline(
            "translation",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        print("✅ TRANSLATION: Model loaded successfully!\n")

    def translate(self, text, src_lang_code, tgt_lang_code):
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            src_lang_code: Source language code (NLLB format)
            tgt_lang_code: Target language code (NLLB format)
        
        Returns:
            str: Translated text
        """
        if not text or not src_lang_code or not tgt_lang_code:
            return ""
        try:
            result = self.translator(text, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
            return result[0]['translation_text']
        except Exception as e:
            print(f"⚠️  Translation Error: {e}")
            return "[Translation failed]"


# ██████████████████████████████████████████████████████████████████████████████
#                    MAIN STT + SPEAKER ID + TRANSLATION PIPELINE
# ██████████████████████████████████████████████████████████████████████████████

class SpeechTranslatorPipeline:
    """
    Unified pipeline that handles:
    1. Audio capture and VAD
    2. Speaker identification
    3. Speech-to-text (Whisper or Vosk)
    4. Translation
    """
    
    def __init__(self, src_config, target_config):
        # STT Config
        self.src_config = src_config
        self.src_lang_name = src_config["name"]
        self.src_lang_code = src_config["code"]
        self.engine = src_config["engine"]
        
        # Translation Config
        self.target_config = target_config
        self.target_lang_name = target_config["name"]
        self.target_lang_code = target_config["code"]

        # STT Whisper/Vosk variables
        self.processor = None
        self.model = None
        self.recognizer = None
        
        torch.set_grad_enabled(False)

        # Initialize components
        self.speaker_identifier = SpeakerIdentifier()
        self.translator = Translator()

    def load_whisper(self):
        """Load Whisper model for STT."""
        model_id = WHISPER_MODELS[self.src_lang_code]
        print(f"🎤 STT: Loading Whisper model: {model_id} for {self.src_lang_name.upper()}")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to("cpu").eval()
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.src_lang_code, task="transcribe"
        )
        print(f"✅ Whisper model loaded\n")
        return model_id

    def load_vosk(self):
        """Load Vosk model for STT."""
        path = VOSK_MODELS[self.src_lang_code]
        if not os.path.exists(path):
            print(f"❌ Vosk model not found: {path}")
            sys.exit(1)
        print(f"🎤 STT: Loading Vosk model: {path}")
        model = VoskModel(path)
        self.recognizer = KaldiRecognizer(model, SR)
        self.recognizer.SetWords(True)
        self.recognizer.SetPartialWords(True)
        print(f"✅ Vosk model loaded\n")
        return path

    def whisper_transcribe(self, pcm32):
        """
        Transcribe audio using Whisper.
        
        Args:
            pcm32: Audio as float32 normalized to [-1, 1]
        
        Returns:
            str: Transcribed text
        """
        feats = self.processor(pcm32, sampling_rate=SR, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = self.model.generate(feats, max_new_tokens=128, do_sample=False)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def process_segment(self, audio_int16, model_id="vosk"):
        """
        Unified processing: Speaker ID → STT → Translation → Display
        
        This is the core pipeline that processes each audio segment in order:
        1. Identify/enroll speaker
        2. Transcribe speech
        3. Translate text
        4. Display formatted output
        
        Args:
            audio_int16: Audio buffer as int16 numpy array
            model_id: Model identifier for logging
        """
        # STEP 1: Identify or enroll speaker using the SAME audio segment
        speaker_name = self.speaker_identifier.identify_or_enroll(audio_int16)
        
        # STEP 2: Transcribe the SAME audio segment
        if self.engine == "whisper":
            pcm32 = audio_int16.astype(np.float32) / 32768.0
            text = self.whisper_transcribe(pcm32)
        else:  # Vosk transcription happens incrementally in run_vosk
            text = ""  # Text comes from Vosk's AcceptWaveform result
        
        # For Vosk, text is passed separately, so only process if we have text
        if not text and self.engine == "vosk":
            return speaker_name  # Return speaker for Vosk to use later
        
        # STEP 3: Translate the transcribed text
        if text:
            self._translate_and_display(speaker_name, text, model_id)
        
        return speaker_name

    def _translate_and_display(self, speaker_name, text, model_id):
        """
        Translate text and display formatted output with speaker identification.
        
        Args:
            speaker_name: Identified speaker name
            text: Transcribed text in source language
            model_id: Model identifier for logging
        """
        if not text:
            return
        
        ts = utc_iso()
        nllb_src_code = NLLB_LANG_MAP.get(self.src_lang_code)
        
        # Translate the text
        translated_text = self.translator.translate(text, nllb_src_code, self.target_lang_code)

        # Display formatted output with speaker identification
        print(f"\n{'='*70}")
        print(f"⏰ [{ts}]")
        print(f"👤 {speaker_name}")
        print(f"🎙️  {self.src_lang_name}: {text}")
        if self.src_lang_code != self.target_lang_code[:3]:  # Basic check if different langs
            print(f"🌍 {self.target_lang_name}: {translated_text}")
        print(f"{'='*70}")

        # Log to file
        log_data = {
            "speaker": speaker_name,
            "transcribed_text": text,
            "translated_text": translated_text,
        }
        with LOG_PATH.open("a", encoding="utf-8") as log_fh:
            log_fh.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            log_fh.flush()

    def run_whisper(self):
        """
        Run Whisper-based pipeline with speaker identification.
        Audio flow: Capture → VAD → Buffer → Speaker ID → STT → Translate
        """
        model_id = self.load_whisper()
        vad = webrtcvad.Vad(VAD_MODE)
        q_in = queue.Queue()
        buf = np.empty(0, dtype=np.int16)

        print(f"📝 Logging to: {LOG_PATH.absolute()}")
        
        def audio_cb(indata, frames, t, status):
            if status:
                print("Audio:", status, file=sys.stderr)
            q_in.put(bytes(indata))
        
        print(f"\n{'='*70}")
        print(f"🎤 LISTENING... ({self.src_lang_name} → {self.target_lang_name})")
        print(f"{'='*70}")
        print("Press Ctrl+C to stop.\n")
        
        with sd.RawInputStream(samplerate=SR, blocksize=FRAME_SAMP, channels=CHANNELS, 
                               dtype=DTYPE, callback=audio_cb):
            try:
                while True:
                    chunk = q_in.get()
                    
                    # Check if this frame contains speech
                    if vad.is_speech(chunk, SR):
                        # Accumulate speech frames
                        buf = np.concatenate((buf, np.frombuffer(chunk, np.int16)))
                        
                        # Process when buffer reaches target size
                        if buf.size >= WHISPER_BUF_SAMP:
                            # Process the complete segment: Speaker ID → STT → Translate
                            self.process_segment(buf, model_id)
                            buf = np.empty(0, dtype=np.int16)
                    
                    # Process remaining buffer after speech stops
                    elif buf.size >= int(0.8 * SR):
                        # Process the complete segment: Speaker ID → STT → Translate
                        self.process_segment(buf, model_id)
                        buf = np.empty(0, dtype=np.int16)

            except KeyboardInterrupt:
                print("\n\n🛑 Pipeline stopped.")
                print(f"📊 Total speakers identified: {self.speaker_identifier.speaker_count}")

    def run_vosk(self):
        """
        Run Vosk-based pipeline with speaker identification.
        Audio flow: Capture → Vosk (incremental) → Speaker ID → Translate
        """
        model_id = self.load_vosk()
        q_in = queue.Queue()
        
        # Buffer for speaker identification (accumulate audio until final result)
        speaker_id_buffer = np.empty(0, dtype=np.int16)
        last_process_time = time.time()  # Track when we last processed

        def audio_cb(indata, frames, t, status):
            if status:
                print("Audio:", status, file=sys.stderr)
            q_in.put(bytes(indata))

        print(f"📝 Logging to: {LOG_PATH.absolute()}")
        print(f"\n{'='*70}")
        print(f"🎤 LISTENING... ({self.src_lang_name} → {self.target_lang_name})")
        print(f"{'='*70}")
        print("Press Ctrl+C to stop.\n")
        
        with sd.RawInputStream(samplerate=SR, blocksize=FRAME_SAMP, channels=CHANNELS, 
                               dtype=DTYPE, callback=audio_cb):
            try:
                while True:
                    chunk = q_in.get()
                    chunk_array = np.frombuffer(chunk, np.int16)
                    
                    # Accumulate audio for speaker identification
                    speaker_id_buffer = np.concatenate((speaker_id_buffer, chunk_array))
                    
                    # Feed to Vosk recognizer
                    if self.recognizer.AcceptWaveform(chunk):
                        # Final result available
                        res = json.loads(self.recognizer.Result())
                        text = res.get("text", "").strip()
                        
                        if text and len(speaker_id_buffer) > 0:
                            # Process: Speaker ID → already have text from Vosk → Translate
                            speaker_name = self.speaker_identifier.identify_or_enroll(speaker_id_buffer)
                            self._translate_and_display(speaker_name, text, model_id)
                            last_process_time = time.time()
                        
                        # Clear buffer after processing
                        speaker_id_buffer = np.empty(0, dtype=np.int16)
                    else:
                        # Partial result - show progress
                        part = json.loads(self.recognizer.PartialResult())
                        ptxt = part.get("partial", "").strip()
                        if ptxt:
                            print(f"… {ptxt}", end="\r")
                        
                        # If buffer is getting too large without a result, process it
                        # This helps with very long pauses
                        current_time = time.time()
                        if len(speaker_id_buffer) > SR * 10:  # More than 10 seconds
                            if (current_time - last_process_time) > 5.0:  # 5 seconds since last
                                # Reset buffer to prevent memory issues
                                speaker_id_buffer = speaker_id_buffer[-SR*3:]  # Keep last 3 seconds
                            
            except KeyboardInterrupt:
                print("\n\n🛑 Pipeline stopped.")
                print(f"📊 Total speakers identified: {self.speaker_identifier.speaker_count}")

    def run(self):
        """Start the appropriate pipeline based on selected engine."""
        if self.engine == "whisper":
            self.run_whisper()
        else:
            self.run_vosk()


# ██████████████████████████████████████████████████████████████████████████████
#                              ENTRY POINT
# ██████████████████████████████████████████████████████████████████████████████

def choose_languages():
    """Interactive CLI to select source and target languages."""
    print("\n" + "="*70)
    print("  REAL-TIME SPEECH TRANSLATION WITH SPEAKER IDENTIFICATION")
    print("="*70 + "\n")
    
    # --- Select Source Language ---
    print("STEP 1: Select the SOURCE language you will be speaking:")
    for i, config in STT_MENU.items():
        print(f" {i}. {config['name']:12} (Engine: {config['engine']})")
    
    try:
        sel_src_idx = int(input("\nEnter number for source language: ").strip())
        if sel_src_idx not in STT_MENU:
            sel_src_idx = 5  # Default to English
    except ValueError:
        sel_src_idx = 5
    
    src_config = STT_MENU[sel_src_idx]
    print(f"✅ Source selected: {src_config['name']}\n")

    # --- Select Target Language ---
    print("STEP 2: Select the TARGET language for translation:")
    for i, config in TRANSLATION_MENU.items():
        print(f" {i}. {config['name']}")
    
    try:
        sel_tgt_idx = int(input("\nEnter number for target language: ").strip())
        if sel_tgt_idx not in TRANSLATION_MENU:
            sel_tgt_idx = 2  # Default to Hindi
    except ValueError:
        sel_tgt_idx = 2
        
    target_config = TRANSLATION_MENU[sel_tgt_idx]
    print(f"✅ Target selected: {target_config['name']}\n")
    
    return src_config, target_config


def main():
    """Main entry point."""
    src_config, target_config = choose_languages()
    pipeline = SpeechTranslatorPipeline(src_config, target_config)
    pipeline.run()


if __name__ == "__main__":
    main()
    
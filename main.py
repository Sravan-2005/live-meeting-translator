# realtime_translator.py

import warnings
# This line will ignore the specific UserWarning coming from the pkg_resources module.
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')

import os, sys, time, json, queue, pathlib
import numpy as np
import sounddevice as sd
import webrtcvad
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from vosk import Model as VoskModel, KaldiRecognizer


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
#                          TRANSLATION COMPONENT
# ██████████████████████████████████████████████████████████████████████████████

class Translator:
    """A simplified translator class to be used within the main pipeline."""
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- TRANSLATION: Using device: {self.device} ---")
        print("--- TRANSLATION: Loading model... (This may take a moment) ---")
        self.translator = pipeline(
            "translation",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        print("--- TRANSLATION: Model loaded successfully! ---\n")

    def translate(self, text, src_lang_code, tgt_lang_code):
        """Performs the translation."""
        if not text or not src_lang_code or not tgt_lang_code:
            return ""
        try:
            result = self.translator(text, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
            return result[0]['translation_text']
        except Exception as e:
            print(f"Translation Error: {e}")
            return "[Translation failed]"

# ██████████████████████████████████████████████████████████████████████████████
#                         MAIN STT & TRANSLATION PIPELINE
# ██████████████████████████████████████████████████████████████████████████████

class SpeechTranslatorPipeline:
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

        # Initialize the translator component
        self.translator = Translator()

    def load_whisper(self):
        model_id = WHISPER_MODELS[self.src_lang_code]
        print(f"--- STT: Loading Whisper model: {model_id} for {self.src_lang_name.upper()} ---")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to("cpu").eval()
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.src_lang_code, task="transcribe"
        )
        return model_id

    def load_vosk(self):
        path = VOSK_MODELS[self.src_lang_code]
        if not os.path.exists(path):
            print(f"❌ Vosk model not found: {path}")
            sys.exit(1)
        print(f"--- STT: Loading Vosk model: {path} ---")
        model = VoskModel(path)
        self.recognizer = KaldiRecognizer(model, SR)
        self.recognizer.SetWords(True)
        self.recognizer.SetPartialWords(True)
        return path

    def whisper_transcribe(self, pcm32):
        feats = self.processor(pcm32, sampling_rate=SR, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = self.model.generate(feats, max_new_tokens=128, do_sample=False)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def process_and_translate(self, text, model_id="vosk"):
        """Handles translation and printing for any transcribed text."""
        if not text:
            return
            
        ts = utc_iso()
        nllb_src_code = NLLB_LANG_MAP.get(self.src_lang_code)
        
        # Translate the text
        translated_text = self.translator.translate(text, nllb_src_code, self.target_lang_code)

        # Print results
        print(f"\n[{ts}]")
        print(f"🎙️ SRC ({self.src_lang_name}): {text}")
        print(f"🌍 TGT ({self.target_lang_name}): {translated_text}")

        # Log to file
        log_data = {
            "timestamp": ts,
            "source_language": self.src_lang_name,
            "target_language": self.target_lang_name,
            "engine": self.engine,
            "model": model_id,
            "transcribed_text": text,
            "translated_text": translated_text,
        }
        with LOG_PATH.open("a", encoding="utf-8") as log_fh:
            log_fh.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            log_fh.flush()


    def run_whisper(self):
        model_id = self.load_whisper()
        vad = webrtcvad.Vad(VAD_MODE)
        q_in = queue.Queue()
        buf = np.empty(0, dtype=np.int16)

        print(f"\n📝 Logging to: {LOG_PATH.absolute()}")
        def audio_cb(indata, frames, t, status):
            if status: print("Audio:", status, file=sys.stderr)
            q_in.put(bytes(indata))
        
        print(f"\n🎤 Listening... ({self.src_lang_name} → {self.target_lang_name}) | Ctrl+C to stop.")
        with sd.RawInputStream(samplerate=SR, blocksize=FRAME_SAMP, channels=CHANNELS, dtype=DTYPE, callback=audio_cb):
            try:
                while True:
                    chunk = q_in.get()
                    if vad.is_speech(chunk, SR):
                        buf = np.concatenate((buf, np.frombuffer(chunk, np.int16)))
                        if buf.size >= WHISPER_BUF_SAMP:
                            pcm32 = buf.astype(np.float32) / 32768.0
                            text = self.whisper_transcribe(pcm32)
                            self.process_and_translate(text, model_id)
                            buf = np.empty(0, dtype=np.int16)
                    elif buf.size >= int(0.8 * SR): # Process remaining buffer after speech stops
                        pcm32 = buf.astype(np.float32) / 32768.0
                        text = self.whisper_transcribe(pcm32)
                        self.process_and_translate(text, model_id)
                        buf = np.empty(0, dtype=np.int16)

            except KeyboardInterrupt:
                print("\n🛑 Pipeline stopped.")

    def run_vosk(self):
        model_id = self.load_vosk()
        q_in = queue.Queue()

        def audio_cb(indata, frames, t, status):
            if status: print("Audio:", status, file=sys.stderr)
            q_in.put(bytes(indata))

        print(f"\n🎤 Listening... ({self.src_lang_name} → {self.target_lang_name}) | Ctrl+C to stop.")
        with sd.RawInputStream(samplerate=SR, blocksize=FRAME_SAMP, channels=CHANNELS, dtype=DTYPE, callback=audio_cb):
            try:
                while True:
                    chunk = q_in.get()
                    if self.recognizer.AcceptWaveform(chunk):
                        res = json.loads(self.recognizer.Result())
                        text = res.get("text", "").strip()
                        self.process_and_translate(text, model_id)
                    else:
                        part = json.loads(self.recognizer.PartialResult())
                        ptxt = part.get("partial", "").strip()
                        if ptxt:
                            print(f"… {ptxt}", end="\r")
            except KeyboardInterrupt:
                print("\n🛑 Pipeline stopped.")

    def run(self):
        if self.engine == "whisper":
            self.run_whisper()
        else:
            self.run_vosk()

# ██████████████████████████████████████████████████████████████████████████████
#                              ENTRY POINT
# ██████████████████████████████████████████████████████████████████████████████

def choose_languages():
    """Interactive CLI to select source and target languages."""
    # --- Select Source Language ---
    print("STEP 1: Select the SOURCE language you will be speaking:")
    for i, config in STT_MENU.items():
        print(f" {i}. {config['name']} (Engine: {config['engine']})")
    
    try:
        sel_src_idx = int(input("Enter number for source language: ").strip())
        if sel_src_idx not in STT_MENU: sel_src_idx = 5 # Default to English
    except ValueError:
        sel_src_idx = 5
    
    src_config = STT_MENU[sel_src_idx]
    print(f"✅ Source selected: {src_config['name']}\n")

    # --- Select Target Language ---
    print("STEP 2: Select the TARGET language for translation:")
    for i, config in TRANSLATION_MENU.items():
        print(f" {i}. {config['name']}")
    
    try:
        sel_tgt_idx = int(input("Enter number for target language: ").strip())
        if sel_tgt_idx not in TRANSLATION_MENU: sel_tgt_idx = 2 # Default to Hindi
    except ValueError:
        sel_tgt_idx = 2
        
    target_config = TRANSLATION_MENU[sel_tgt_idx]
    print(f"✅ Target selected: {target_config['name']}\n")
    
    return src_config, target_config

def main():
    src_config, target_config = choose_languages()
    pipeline = SpeechTranslatorPipeline(src_config, target_config)
    pipeline.run()

if __name__ == "__main__":
    main()
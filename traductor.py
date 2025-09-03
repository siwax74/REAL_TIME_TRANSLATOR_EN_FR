#!/usr/bin/env python3
"""
Traducteur en temps réel ultra-optimisé
Anglais → Français avec reconnaissance vocale
Affichage uniquement du texte traduit en français
"""

import queue
import threading
import time
import json
import sys
import os
from typing import Optional, Dict, Tuple
from collections import deque

import sounddevice as sd
import vosk
import torch
from transformers import pipeline, MarianMTModel, MarianTokenizer
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Audio
    MIC_NAME = "USB Microphone"
    SAMPLE_RATE = 16000
    BLOCK_SIZE = 2048
    CHANNELS = 1
    
    # Modèles
    VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
    TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-fr"
    
    # Performance
    USE_GPU = torch.cuda.is_available()
    MAX_CACHE_SIZE = 2000
    MIN_TRANSLATION_DELAY = 0.1  # pour traduction partielle
    MAX_PARTIAL_LENGTH = 200
    
    # Threading
    MAX_WORKERS = min(4, psutil.cpu_count())
    QUEUE_MAX_SIZE = 50

# =============================================================================
# CACHE INTELLIGENT LRU
# =============================================================================
class TranslationCache:
    def __init__(self, max_size: int = Config.MAX_CACHE_SIZE):
        self.cache: Dict[str, str] = {}
        self.access_order = deque()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, text: str) -> Optional[str]:
        clean_text = text.strip().lower()
        if clean_text in self.cache:
            self.hits += 1
            self.access_order.remove(clean_text)
            self.access_order.append(clean_text)
            return self.cache[clean_text]
        self.misses += 1
        return None
    
    def put(self, text: str, translation: str):
        clean_text = text.strip().lower()
        if len(self.cache) >= self.max_size and clean_text not in self.cache:
            oldest = self.access_order.popleft()
            del self.cache[oldest]
        self.cache[clean_text] = translation
        if clean_text in self.access_order:
            self.access_order.remove(clean_text)
        self.access_order.append(clean_text)
    
    def get_stats(self) -> Tuple[int,int,float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return self.hits, self.misses, hit_rate

# =============================================================================
# AUDIO MANAGER
# =============================================================================
class AudioManager:
    def __init__(self):
        self.device_index = None
        self.sample_rate = Config.SAMPLE_RATE
        self.audio_queue = queue.Queue(maxsize=Config.QUEUE_MAX_SIZE)
    
    def initialize_device(self):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if Config.MIC_NAME.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                self.device_index = i
                self.sample_rate = int(dev['default_samplerate'])
                print(f"🎤 Micro: {dev['name']} ({self.sample_rate}Hz)")
                return
        raise RuntimeError(f"❌ Micro '{Config.MIC_NAME}' introuvable!")
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Audio: {status}")
        try:
            self.audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(bytes(indata))
            except queue.Empty:
                pass

# =============================================================================
# SMART TRANSLATOR
# =============================================================================
class SmartTranslator:
    def __init__(self):
        self.cache = TranslationCache()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialize_model()
        self.last_partial = ""
        self.last_translation = ""
    
    def _initialize_model(self):
        print("🔄 Chargement modèle de traduction...")
        try:
            if Config.USE_GPU:
                print("🚀 Utilisation GPU")
                self.pipeline = pipeline(
                    "translation_en_to_fr",
                    model=Config.TRANSLATION_MODEL,
                    device=0,
                    torch_dtype=torch.float16
                )
            else:
                print("💻 Utilisation CPU")
                self.tokenizer = MarianTokenizer.from_pretrained(Config.TRANSLATION_MODEL)
                self.model = MarianMTModel.from_pretrained(Config.TRANSLATION_MODEL)
        except Exception as e:
            print(f"❌ Erreur modèle: {e}")
            sys.exit(1)
    
    def translate_text(self, text: str) -> str:
        if not text.strip():
            return ""
        cached = self.cache.get(text)
        if cached:
            return cached
        try:
            if self.pipeline:
                result = self.pipeline(text, max_length=512)[0]['translation_text']
            else:
                tokens = self.tokenizer(text, return_tensors="pt",
                                        padding=True, truncation=True, max_length=512)
                translated = self.model.generate(**tokens, max_length=512, num_beams=1)
                result = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            self.cache.put(text, result)
            return result
        except Exception as e:
            return f"[ERREUR: {text}]"
    
    def smart_partial_translate(self, current_text: str) -> Optional[str]:
        if not current_text.strip():
            return None
        # Différentiel
        if self.last_partial and current_text.startswith(self.last_partial):
            new_part = current_text[len(self.last_partial):].strip()
            if new_part:
                new_translation = self.translate_text(new_part)
                full_translation = (self.last_translation + " " + new_translation).strip()
                self.last_partial = current_text
                self.last_translation = full_translation
                return full_translation
        # Traduction complète
        translation = self.translate_text(current_text)
        self.last_partial = current_text
        self.last_translation = translation
        return translation
    
    def reset_partial(self):
        self.last_partial = ""
        self.last_translation = ""

# =============================================================================
# REAL TIME TRANSLATOR
# =============================================================================
class RealTimeTranslator:
    def __init__(self):
        self.audio_manager = AudioManager()
        self.translator = SmartTranslator()
        self.vosk_model = None
        self.recognizer = None
        self.translation_queue = queue.Queue(maxsize=Config.QUEUE_MAX_SIZE)
        self.running = False
        self.threads = []
        self.start_time = time.time()
        self.translations_count = 0
    
    def initialize(self):
        print("🚀 Initialisation...")
        self.audio_manager.initialize_device()
        if not os.path.exists(Config.VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Modèle Vosk introuvable: {Config.VOSK_MODEL_PATH}")
        self.vosk_model = vosk.Model(Config.VOSK_MODEL_PATH)
        self.recognizer = vosk.KaldiRecognizer(self.vosk_model, self.audio_manager.sample_rate)
        print("✅ Initialisation terminée")
    
    def translation_worker(self):
        while self.running:
            try:
                text_en = self.translation_queue.get(timeout=1)
                if text_en == "__STOP__":
                    break
                self._clear_line()
                text_fr = self.translator.translate_text(text_en)
                print(f"{text_fr}\n")
                self.translations_count += 1
            except queue.Empty:
                continue
    
    def _clear_line(self):
        print('\r' + ' ' * 120, end='\r')
    
    def run(self):
        self.running = True
        translation_thread = threading.Thread(target=self.translation_worker, daemon=True)
        translation_thread.start()
        self.threads.append(translation_thread)
        last_time = 0.0
        
        print("🎙️ Parlez en anglais...")
        try:
            with sd.RawInputStream(
                samplerate=self.audio_manager.sample_rate,
                blocksize=Config.BLOCK_SIZE,
                dtype='int16',
                channels=Config.CHANNELS,
                callback=self.audio_manager.audio_callback,
                device=self.audio_manager.device_index
            ):
                while self.running:
                    try:
                        data = self.audio_manager.audio_queue.get(timeout=1)
                        if self.recognizer.AcceptWaveform(data):
                            result = json.loads(self.recognizer.Result())
                            text_en = result.get('text', '').strip()
                            if text_en:
                                self.translation_queue.put(text_en)
                                self.translator.reset_partial()
                                self._clear_line()
                        else:
                            partial_result = json.loads(self.recognizer.PartialResult())
                            partial_text = partial_result.get('partial', '').strip()
                            if partial_text and len(partial_text) <= Config.MAX_PARTIAL_LENGTH:
                                if time.time() - last_time > Config.MIN_TRANSLATION_DELAY:
                                    translation = self.translator.smart_partial_translate(partial_text)
                                    if translation:
                                        print(f"{translation}", end='\r')
                                        last_time = time.time()
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
    
    def shutdown(self):
        print("\n⏹️ Arrêt en cours...")
        self.running = False
        try:
            self.translation_queue.put("__STOP__")
        except queue.Full:
            pass
        for t in self.threads:
            t.join(timeout=2)
        self.print_stats()
        print("✅ Arrêt terminé!")
    
    def print_stats(self):
        uptime = time.time() - self.start_time
        hits, misses, hit_rate = self.translator.cache.get_stats()
        print(f"\n📊 STATISTIQUES")
        print(f"⏱️  Durée: {uptime:.1f}s")
        print(f"🔄 Traductions: {self.translations_count}")
        print(f"💾 Cache: {hit_rate:.1f}% hits ({hits}/{hits + misses})")
        print(f"🎯 GPU: {'Oui' if Config.USE_GPU else 'Non'}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🌟 Traducteur Temps Réel Ultra-Optimisé v4.0")
    print("="*50)
    translator = RealTimeTranslator()
    try:
        translator.initialize()
        translator.run()
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

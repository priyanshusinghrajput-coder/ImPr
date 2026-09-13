"""
inference/tts_engine.py
=========================
Lightweight offline text-to-speech wrapper around pyttsx3, tuned for a
Raspberry Pi routed to a 3.5mm aux output (or USB/HDMI audio - pyttsx3 just
uses whatever ALSA/PulseAudio default sink is configured).

Why a background thread + queue:
pyttsx3's `runAndWait()` blocks until speech finishes. Calling it directly
from the main camera loop would freeze frame capture and landmark
extraction for as long as speech takes, causing dropped/late gesture
predictions. Instead, phrases are pushed onto a queue and spoken by a
dedicated worker thread, so the vision pipeline keeps running smoothly.

Raspberry Pi audio routing reminder:
    sudo raspi-config  ->  System Options -> Audio -> Force 3.5mm ('headphone') jack
or, on newer Pi OS:
    wpctl set-default <sink-id>          # PipeWire
    amixer cset numid=3 1                # force analog/aux output (older ALSA setups)
"""

import queue
import threading

import pyttsx3


class TTSEngine:
    def __init__(self, rate: int = 150, volume: float = 1.0, voice_index: int = 0):
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)      # words per minute; slower is often clearer on tiny speakers
        self._engine.setProperty("volume", volume)

        voices = self._engine.getProperty("voices")
        if voices:
            idx = min(voice_index, len(voices) - 1)
            self._engine.setProperty("voice", voices[idx].id)

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                continue
            self._engine.say(text)
            self._engine.runAndWait()
            self._queue.task_done()

    def speak(self, text: str):
        """Non-blocking: queues text to be spoken by the worker thread."""
        if text and text.strip():
            self._queue.put(text.strip())

    def flush(self):
        """Blocks until all currently queued speech has finished."""
        self._queue.join()

    def shutdown(self):
        self._stop_event.set()
        self._thread.join(timeout=2)


if __name__ == "__main__":
    # Quick standalone smoke test: `python inference/tts_engine.py`
    tts = TTSEngine()
    tts.speak("Dynamic sign language recognition system ready.")
    tts.flush()
    tts.shutdown()

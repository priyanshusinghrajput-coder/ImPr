"""
inference/sentence_builder.py
================================
Turns a raw, noisy stream of per-frame gesture predictions into a clean,
coherent phrase.

A naive approach (speak every predicted class immediately) fails for two
reasons:
  1. Continuous signing produces the *same* predicted class for many
     consecutive frames while the sign is held/repeated -> without
     debouncing you'd get "hello hello hello hello".
  2. Individual frame predictions are noisy - a single low-confidence
     misclassification shouldn't be enough to commit a wrong word.

This module implements a small state machine:
  - a gesture must be predicted for STABLE_FRAMES_REQUIRED consecutive
    frames (above the confidence threshold) before it's "accepted" as a word
  - after acceptance, a cooldown window prevents immediately re-accepting
    the same gesture (so holding a sign doesn't spam it)
  - control tokens ("space", "delete", "done") manipulate the sentence
    buffer directly instead of being appended as words
"""

from __future__ import annotations  # allows `str | None` hints on Python < 3.10 (some Pi OS images ship 3.9)

from collections import deque

from config import STABLE_FRAMES_REQUIRED, COOLDOWN_FRAMES_AFTER_ACCEPT


class SentenceBuilder:
    def __init__(self,
                 stable_frames_required: int = STABLE_FRAMES_REQUIRED,
                 cooldown_frames: int = COOLDOWN_FRAMES_AFTER_ACCEPT):
        self.stable_frames_required = stable_frames_required
        self.cooldown_frames = cooldown_frames

        self._recent_predictions = deque(maxlen=stable_frames_required)
        self._cooldown_remaining = 0
        self._last_accepted_word = None

        self.words = []          # accepted words, in order
        self.finalized = False   # set True when the "done" control token is accepted

    # -- public API ----------------------------------------------------

    def update(self, predicted_label: str, confidence: float, min_confidence: float) -> str | None:
        """
        Feed one new frame-level prediction into the state machine.

        Returns the word that was just accepted this call, or None if
        nothing new was accepted (still stabilizing, in cooldown, or below
        the confidence threshold).
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        if confidence < min_confidence:
            self._recent_predictions.clear()
            return None

        self._recent_predictions.append(predicted_label)

        if len(self._recent_predictions) < self.stable_frames_required:
            return None

        # Only accept if the whole recent window agrees on one label
        if len(set(self._recent_predictions)) != 1:
            return None

        stable_label = self._recent_predictions[0]

        # Avoid immediately re-accepting the same word twice in a row
        if stable_label == self._last_accepted_word:
            return None

        self._accept(stable_label)
        return stable_label

    def get_sentence(self) -> str:
        return " ".join(self.words)

    def reset(self):
        self.words = []
        self.finalized = False
        self._recent_predictions.clear()
        self._cooldown_remaining = 0
        self._last_accepted_word = None

    # -- internals -------------------------------------------------------

    def _accept(self, label: str):
        self._last_accepted_word = label
        self._cooldown_remaining = self.cooldown_frames
        self._recent_predictions.clear()

        if label == "space":
            pass  # words are already space-joined by get_sentence(); nothing to do
        elif label == "delete":
            if self.words:
                self.words.pop()
        elif label == "done":
            self.finalized = True
        else:
            self.words.append(label)

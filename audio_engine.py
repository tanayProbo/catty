"""Procedural sound synthesizer using NumPy and Pygame mixer."""

from __future__ import annotations

from typing import Optional
import numpy as np
import pygame


class AudioEngine:
    """Synthesizes cyberpunk klaxons, alarms, jump/roll FX, and chimes with zero external assets."""

    SAMPLE_RATE = 22050

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.available = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        self._init_mixer()

    def _init_mixer(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=2, buffer=512)
            self.available = True
            self._build_sounds()
        except Exception:
            self.available = False

    def _to_sound(self, mono_wave: np.ndarray, volume: float = 0.35) -> Optional[pygame.mixer.Sound]:
        if not self.available:
            return None
        clipped = np.clip(mono_wave * volume, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        stereo = np.column_stack((pcm, pcm))
        return pygame.sndarray.make_sound(stereo)

    def _envelope(self, n: int, attack: float = 0.03, release: float = 0.15) -> np.ndarray:
        env = np.ones(n, dtype=np.float32)
        att_n = max(1, int(n * attack))
        rel_n = max(1, int(n * release))
        env[:att_n] = np.linspace(0.0, 1.0, att_n, dtype=np.float32)
        env[-rel_n:] = np.linspace(1.0, 0.0, rel_n, dtype=np.float32)
        return env

    def _build_sounds(self) -> None:
        sr = self.SAMPLE_RATE

        # 1. Warning Klaxon (dual-pulse sawtooth/square sweep)
        dur = 0.42
        t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
        freq = 540.0 + 260.0 * np.sin(2.0 * np.pi * 6.0 * t)
        phase = np.cumsum(2.0 * np.pi * freq / sr)
        wave = np.sign(np.sin(phase)) * 0.65 + np.sin(phase * 0.5) * 0.35
        klaxon = wave * self._envelope(len(t), 0.04, 0.2)
        snd = self._to_sound(klaxon, volume=0.28)
        if snd:
            self._sounds["klaxon"] = snd

        # 2. Busted Alarm Siren (urgent descending/ascending security siren)
        dur_a = 0.85
        ta = np.linspace(0, dur_a, int(sr * dur_a), endpoint=False, dtype=np.float32)
        freq_a = 720.0 + 380.0 * np.sign(np.sin(2.0 * np.pi * 5.0 * ta))
        phase_a = np.cumsum(2.0 * np.pi * freq_a / sr)
        wave_a = np.sign(np.sin(phase_a)) * 0.7 + np.sin(phase_a * 2.0) * 0.3
        alarm = wave_a * self._envelope(len(ta), 0.02, 0.12)
        snd = self._to_sound(alarm, volume=0.32)
        if snd:
            self._sounds["alarm"] = snd

        # 3. Freeze Clear Chime (ascending matrix green triad)
        notes = [523.25, 659.25, 783.99]
        parts = []
        for f in notes:
            tn = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False, dtype=np.float32)
            w = np.sin(2.0 * np.pi * f * tn) * 0.7 + np.sin(4.0 * np.pi * f * tn) * 0.3
            parts.append(w * self._envelope(len(tn), 0.05, 0.4))
        snd = self._to_sound(np.concatenate(parts), volume=0.30)
        if snd:
            self._sounds["clear"] = snd

        # 4. Victory Fanfare
        v_notes = [523.25, 659.25, 783.99, 1046.5]
        v_parts = []
        for idx, f in enumerate(v_notes):
            d = 0.24 if idx == len(v_notes) - 1 else 0.11
            tv = np.linspace(0, d, int(sr * d), endpoint=False, dtype=np.float32)
            w = np.sin(2.0 * np.pi * f * tv) * 0.65 + np.sign(np.sin(2.0 * np.pi * f * tv)) * 0.25
            v_parts.append(w * self._envelope(len(tv), 0.05, 0.35))
        snd = self._to_sound(np.concatenate(v_parts), volume=0.32)
        if snd:
            self._sounds["victory"] = snd

        # 5. Subtle Bouncy Paw Step
        ts = np.linspace(0, 0.045, int(sr * 0.045), endpoint=False, dtype=np.float32)
        step_w = np.sin(2.0 * np.pi * 195.0 * ts) * self._envelope(len(ts), 0.08, 0.7)
        snd = self._to_sound(step_w, volume=0.10)
        if snd:
            self._sounds["step"] = snd

        # 6. Cat Jump Spring (ascending frequency sweep)
        tj = np.linspace(0, 0.18, int(sr * 0.18), endpoint=False, dtype=np.float32)
        f_jump = np.linspace(260.0, 680.0, len(tj), dtype=np.float32)
        p_jump = np.cumsum(2.0 * np.pi * f_jump / sr)
        w_jump = (np.sin(p_jump) * 0.75 + np.sign(np.sin(p_jump)) * 0.25) * self._envelope(len(tj), 0.05, 0.35)
        snd = self._to_sound(w_jump, volume=0.25)
        if snd:
            self._sounds["jump"] = snd

        # 7. Cat Crouch-Roll Whoosh (descending synth + filtered noise spin)
        tr = np.linspace(0, 0.22, int(sr * 0.22), endpoint=False, dtype=np.float32)
        f_roll = np.linspace(480.0, 140.0, len(tr), dtype=np.float32)
        p_roll = np.cumsum(2.0 * np.pi * f_roll / sr)
        w_roll = (np.sin(p_roll) * 0.8 + np.sin(p_roll * 2.5) * 0.2) * self._envelope(len(tr), 0.06, 0.4)
        snd = self._to_sound(w_roll, volume=0.26)
        if snd:
            self._sounds["roll"] = snd

        # 8. Obstacle Dodge / Roll-Under Bonus Blip
        td = np.linspace(0, 0.09, int(sr * 0.09), endpoint=False, dtype=np.float32)
        w_dodge = np.sin(2.0 * np.pi * 880.0 * td) * self._envelope(len(td), 0.05, 0.5)
        snd = self._to_sound(w_dodge, volume=0.22)
        if snd:
            self._sounds["dodge"] = snd

    def play(self, name: str) -> None:
        if not self.enabled or not self.available:
            return
        snd = self._sounds.get(name)
        if snd is not None:
            try:
                snd.play()
            except Exception:
                pass

"""Senal de CERRAR / RELAJA para el piloto: la misma para las dos metas salvo la palabra (o el orden de dos tonos).

Por que existe: con el casco real la potencia posterior de "cerrar" salio mas alta que la de "relajar" y la
senal era roja contra azul. Un decoder puede aprender esa pista visual en lugar del ERD motor, asi que las
dos senales tienen el mismo color, tamano, grosor y ancho. Todo lo que se ve o se oye se define en
config.CUE_VISUAL y config.CUE_AUDIO; aqui solo se arma. Sin Qt ni hardware, para probarlo:

  texto(meta, ...)      la palabra que se ve ('CERRAR' / 'RELAJA', o el '+' de la senal solo auditiva)
  estilo(...)           la hoja de estilo del tablero: SIN depender de la meta
  tonos(meta)           los dos tonos [(Hz, ms), ...] de la senal auditiva
  Audio                 los suena en un hilo aparte (no frena el lazo); `reproducir` se puede inyectar
"""
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import wave

import config


def texto(meta, visual=True):
    """Lo que se ve: la palabra de la meta, o un '+' fijo si la senal es solo auditiva."""
    return config.CUE_VISUAL['texto'][1 if meta > 0 else -1] if visual else config.CUE_VISUAL['neutro']


def estilo():
    """Hoja de estilo del tablero para la senal. No recibe la meta: es imposible que dependa de ella."""
    v = config.CUE_VISUAL
    return f"font-size:{v['px']}px;font-weight:bold;padding:4px;color:{v['color']};font-family:{v['familia']};"


def tonos(meta):
    """Los dos tonos de la senal auditiva: [(frecuencia Hz, duracion ms), ...]. Los mismos dos para
    las dos metas; CERRAR sube (grave -> agudo) y RELAJA baja."""
    f1, f2 = config.CUE_AUDIO['frecuencias_hz']
    ms = config.CUE_AUDIO['tono_ms']
    return [(f1, ms), (f2, ms)] if meta > 0 else [(f2, ms), (f1, ms)]


def onda(meta, fs=22050):
    """Los tonos como muestras enteras de 16 bits (con subida y bajada suaves, sin clic)."""
    muestras = []
    amp = config.CUE_AUDIO['volumen'] * 32767
    for hz, ms in tonos(meta):
        n = int(fs * ms / 1000)
        rampa = max(1, int(0.01 * fs))
        for i in range(n):
            ventana = min(1.0, i / rampa, (n - 1 - i) / rampa)
            muestras.append(int(amp * ventana * math.sin(2 * math.pi * hz * i / fs)))
    return muestras


def _reproducir_sistema(meta):
    """Suena la senal en este sistema. Devuelve True si pudo, False si no hay como (sin audio)."""
    if sys.platform == 'win32':
        try:
            import winsound
            for hz, ms in tonos(meta):
                winsound.Beep(int(hz), int(ms))
            return True
        except Exception:
            return False
    reproductor = next((r for r in ('afplay', 'paplay', 'aplay') if shutil.which(r)), None)
    if reproductor is None:
        return False
    fs = 22050
    fd, ruta = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        with wave.open(ruta, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes(b''.join(struct.pack('<h', m) for m in onda(meta, fs)))
        return subprocess.run([reproductor, ruta], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=5).returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(ruta)
        except OSError:
            pass


class Audio:
    """Senal auditiva opcional (--cue-audio). sonar() vuelve de inmediato: el sonido va en un hilo.
    Si el sistema no puede sonar lo dice una vez (`fallo`) y el lazo sigue: la senal visual no se pierde."""

    def __init__(self, activo=False, reproducir=None, avisar=print):
        self.activo, self.reproducir, self.avisar = activo, reproducir or _reproducir_sistema, avisar
        self.fallo, self.sonadas, self._hilo, self._avisado = None, [], None, False

    def sonar(self, meta):
        if not self.activo:
            return
        self.sonadas.append(1 if meta > 0 else -1)
        self._hilo = threading.Thread(target=self._sonar, args=(meta,), daemon=True)
        self._hilo.start()

    def _sonar(self, meta):
        try:
            ok = self.reproducir(meta)
        except Exception as e:                     # un fallo de audio nunca detiene la sesion
            ok, self.fallo = False, f'{type(e).__name__}: {e}'
        if not ok and self.fallo is None:
            self.fallo = 'este sistema no tiene como reproducir el tono'
        if not ok and not self._avisado:           # una sola vez
            self._avisado = True
            self.avisar(f'AVISO: la senal auditiva no sono ({self.fallo}); la senal visual sigue.')

    def esperar(self, segundos=2.0):
        """Para las pruebas: espera a que termine el ultimo tono."""
        if self._hilo is not None:
            self._hilo.join(segundos)

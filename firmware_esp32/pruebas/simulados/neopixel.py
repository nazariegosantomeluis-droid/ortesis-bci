"""Módulo neopixel simulado (solo pruebas)."""
class NeoPixel:
    def __init__(self, pin, n):
        self.n, self.buf = n, [(0, 0, 0)] * n
    def __setitem__(self, i, c):
        self.buf[i] = c
    def __getitem__(self, i):
        return self.buf[i]
    def write(self):
        pass

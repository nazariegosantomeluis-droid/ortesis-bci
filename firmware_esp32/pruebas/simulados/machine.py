"""Módulo machine simulado para correr el firmware en CPython (solo pruebas)."""
class Pin:
    OUT, IN, PULL_UP = 1, 0, 2
    entradas = {}                      # pin → lo que lee un Pin de entrada (1 = suelto, por el pull-up)
    def __init__(self, n, modo=None, pull=None, value=None):
        self.n, self.modo = n, modo
        self.v = 0 if value is None else value
    def value(self, v=None):
        if v is None:
            return Pin.entradas.get(self.n, 1) if self.modo == Pin.IN else self.v
        self.v = v
class PWM:
    registro = {}
    def __init__(self, pin, freq=50):
        self.pin, self.ns = pin, 0
        PWM.registro[pin.n] = self
    def duty_ns(self, ns=None):
        if ns is None:
            return self.ns
        self.ns = ns
    def deinit(self):
        self.apagado = True
class ADC:
    ATTN_0DB, ATTN_11DB = 0, 3
    valores_uv = {}
    def __init__(self, pin):
        self.pin = pin
    def atten(self, a):
        pass
    def read_uv(self):
        return ADC.valores_uv.get(self.pin.n, 0)

"""Módulo network simulado (solo pruebas)."""
AP_IF, STA_IF = 1, 0
class WLAN:
    def __init__(self, i):
        self.i = i
    def active(self, *a):
        return True
    def config(self, **k):
        pass
    def ifconfig(self):
        return ('192.168.4.1', '255.255.255.0', '192.168.4.1', '8.8.8.8')

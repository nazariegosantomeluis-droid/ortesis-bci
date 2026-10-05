"""Mano virtual a pantalla completa: una mano grande que se abre y se cierra con las MISMAS ordenes que la ortesis.

Sirve para dos cosas:
  - calibrar el ErrP cuando la ortesis no esta lista: el piloto mira esta mano en lugar de la ortesis
    (python orquestador.py real --ortesis-sim --mano-virtual);
  - como segunda pantalla en la demo, junto a la ortesis real (el jurado y el piloto ven lo que ordena el sistema).

Uso:
  python mano_virtual.py                       sigue el flujo Estado, a pantalla completa (Esc sale, F alterna)
  python mano_virtual.py --pantalla 1          en la segunda pantalla (0 = la principal)
  python mano_virtual.py --ventana             en una ventana normal
  python mano_virtual.py --demo                se abre y se cierra sola, sin orquestador (para ver como luce)
  python mano_virtual.py --captura mano.png    guarda una imagen de la mano (--cierre 0..1) y sale

De que se alimenta (config.MANO_VIRTUAL): el evento 'mano' de Estado que publica el orquestador con
--mano-virtual (una por cada orden a la ortesis: pasos, centrado, pausa segura, movimientos ajenos...). Sin
el, sigue el angulo de los eventos 'paso' y 'ajeno'. Encima de la mano muestra la misma senal neutra de
CERRAR/RELAJA que el tablero (cue.py): el piloto necesita saber hacia donde DEBIA moverse para notar un error.

Movimiento rapido y claro: la animacion dura lo que la orden (250 ms por paso) y arranca de golpe y frena al
final (ease-out): el inicio, que es lo que ancla la epoca del ErrP, se ve de inmediato. Con el evento 'mano'
la animacion empieza en `inicio`, el mismo instante que el orquestador usa como ancla de la epoca (con la
ortesis simulada, ACK + su latencia mecanica simulada); queda el retraso de leer el flujo y de dibujar el
cuadro (~20 a 40 ms, SIN medir con un fotodiodo) y el de la pantalla.

Este archivo se divide en lo que se puede probar sin pantalla (geometria, animacion, estado, dibujo contra un
"pintor" cualquiera, espejo de la ortesis) y la ventana Qt, que solo traduce el pintor a QPainter.
"""
import argparse
import json
import math
import sys
import threading

import config
import cue

# ---------------------------------------------------------------- geometria (unidades: largo de la palma = 1)
# Vista de perfil de una mano derecha mirada desde el lado del pulgar, apuntando hacia arriba. La palma mira a
# la derecha y los dedos se enroscan hacia ella. x a la derecha, y hacia arriba.
PALMA = [(-0.42, 0.00), (0.34, 0.00), (0.58, 0.20), (0.60, 0.45), (0.50, 0.95), (0.42, 1.10), (-0.42, 1.10)]   # con el bulto del pulgar
MUNECA = [(-0.36, -0.75), (0.36, -0.75), (0.40, 0.02), (-0.40, 0.02)]
PUNO = [(-0.50, -0.75), (0.50, -0.75), (0.50, -0.50), (-0.50, -0.50)]          # la banda de la ortesis
# (x de la base, largos de las 3 falanges, grosor): de atras hacia adelante, el indice queda al frente
DEDOS = [
    (-0.30, (0.34, 0.20, 0.17), 0.15),     # menique
    (-0.10, (0.42, 0.26, 0.20), 0.17),     # anular
    (0.10, (0.46, 0.30, 0.22), 0.18),      # medio
    (0.30, (0.43, 0.26, 0.20), 0.18),      # indice
]
FLEXION_DEDO = (90.0, 100.0, 75.0)         # grados de flexion de cada articulacion con la mano cerrada del todo
RETRASO_DEDO = (0.0, 0.10, 0.22)           # la flexion de cada falange arranca un poco mas tarde (fraccion del recorrido)
# el pulgar abierto sale a 40 grados y al cerrar sube y se tiende sobre los dedos enroscados (flexion negativa = antihoraria)
PULGAR = ((0.40, 0.35), 40.0, (0.40, 0.32, 0.26), 0.21, (10.0, -50.0, -50.0))   # base, direccion, largos, grosor, flexion
CAJA = (-0.95, -0.85, 1.45, 2.20)          # (xmin, ymin, xmax, ymax) fija: la mano no cambia de escala al moverse

COLOR_FONDO = '#10151c'
COLOR_PIEL = ('#e9bf9f', '#e1b393', '#d9a888', '#cf9d7d')     # de adelante hacia atras se oscurece un poco
COLOR_BORDE = '#7a4f38'
COLOR_PALMA = '#d9a888'
COLOR_BANDA = '#3b6ea5'
COLOR_TEXTO = '#e8edf2'
COLOR_AJENO = '#a678e6'


def _suave(p):
    """0..1 -> 0..1: sube de golpe y frena al final (ease-out cuadratico), para que el inicio se vea de inmediato."""
    p = min(1.0, max(0.0, p))
    return 1.0 - (1.0 - p) ** 2


def cadena(base, direccion, largos, flexiones, cierre, retraso=(0.0, 0.0, 0.0)):
    """Puntos de una cadena de falanges: parte de `base` mirando a `direccion` (grados desde +x, 90 = arriba) y cada
    articulacion gira `flexiones[i] * avance` grados hacia la palma (sentido horario). avance = el cierre (0..1)
    corrido por el retraso de esa falange."""
    puntos, (x, y), ang = [tuple(base)], tuple(base), float(direccion)
    for largo, flex, ret in zip(largos, flexiones, retraso):
        avance = min(1.0, max(0.0, (cierre - ret) / (1.0 - ret))) if ret < 1.0 else 0.0
        ang -= flex * avance
        x += largo * math.cos(math.radians(ang))
        y += largo * math.sin(math.radians(ang))
        puntos.append((x, y))
    return puntos


def primitivas(cierre):
    """La mano en un cierre dado (0 abierta .. 1 cerrada) como figuras en orden de dibujo, de atras hacia adelante:
       ('poligono', puntos, relleno, borde, grosor_borde)   y   ('cadena', puntos, grosor, relleno, borde)."""
    c = min(1.0, max(0.0, float(cierre)))
    fig = [('poligono', MUNECA, COLOR_PALMA, COLOR_BORDE, 0.03), ('poligono', PUNO, COLOR_BANDA, COLOR_BORDE, 0.03)]
    for k, (x, largos, grosor) in enumerate(DEDOS):
        pts = cadena((x, PALMA[-1][1]), 90.0, largos, FLEXION_DEDO, c, RETRASO_DEDO)
        fig.append(('cadena', pts, grosor, COLOR_PIEL[len(DEDOS) - 1 - k], COLOR_BORDE))
    fig.append(('poligono', PALMA, COLOR_PALMA, COLOR_BORDE, 0.03))
    base, direccion, largos, grosor, flex = PULGAR
    fig.append(('cadena', cadena(base, direccion, largos, flex, c), grosor, COLOR_PIEL[0], COLOR_BORDE))
    return fig


def punta_de_dedos(cierre):
    """Posicion de la punta de cada dedo (de menique a indice) y del pulgar: para las pruebas y para ver que se cierra."""
    return [primitivas(cierre)[2 + k][1][-1] for k in range(len(DEDOS))] + [primitivas(cierre)[-1][1][-1]]


# ---------------------------------------------------------------- animacion
class Animacion:
    """El cierre (0..1) en el tiempo: cada orden lleva la mano del valor que tenga en ese instante al destino,
    en `dur_s` segundos desde `t0`. Una orden nueva puede llegar con la anterior a medias."""

    def __init__(self, inicial=0.0):
        self.tramos = [(float('-inf'), 0.0, float(inicial), float(inicial))]      # (t0, dur, desde, hasta)

    def valor(self, t):
        t0, dur, desde, hasta = next(tr for tr in reversed(self.tramos) if tr[0] <= t)
        return desde + _suave((t - t0) / dur if dur > 0 else 1.0) * (hasta - desde)

    def ir_a(self, destino, dur_s, t0):
        """Una orden que empieza en t0 (puede ser un poco en el futuro)."""
        t0 = max(t0, self.tramos[-1][0])              # nunca antes del tramo anterior: el tiempo no retrocede
        self.tramos.append((t0, float(dur_s), self.valor(t0), min(1.0, max(0.0, float(destino)))))
        del self.tramos[:-12]

    def destino(self):
        return self.tramos[-1][3]


class EstadoMano:
    """Lo que se ve, a partir de los eventos del flujo Estado. Sin Qt: se prueba con eventos a mano."""

    def __init__(self):
        self.animacion = Animacion(0.0)
        self.exacto = False              # ya llego algun evento 'mano': los 'paso' y 'ajeno' dejan de mover la mano
        self.meta, self.visual, self.ajeno = None, True, False
        self.ordenes = 0

    def _duracion(self, ms):
        m = config.MANO_VIRTUAL
        return min(m['dur_max_ms'], max(m['dur_min_ms'], float(ms))) / 1000.0

    def procesar(self, e, ahora):
        tipo = e.get('tipo')
        m = config.MANO_VIRTUAL
        if tipo == m['evento']:
            self.exacto = True
            t0 = ahora
            inicio = e.get('inicio')
            if inicio is not None and -m['inicio_antes_s'] <= inicio - ahora <= m['inicio_despues_s']:
                t0 = inicio                                   # arranca cuando el orquestador ancla la epoca del ErrP
            self.animacion.ir_a(e['angulo'], self._duracion(e.get('ms', config.DURACION_PASO_MS)), t0)
            self.ordenes += 1
        elif tipo in ('paso', 'ajeno') and not self.exacto and e.get('angulo') is not None:
            self.animacion.ir_a(e['angulo'], self._duracion(config.DURACION_PASO_MS), ahora)
            self.ordenes += 1
        if tipo == 'cue':
            self.meta, self.visual, self.ajeno = e['meta'], e.get('visual', True), False
        elif tipo == 'aviso_ajeno':
            self.ajeno = True
        elif tipo == 'ajeno':
            self.ajeno = False

    def cierre(self, ahora):
        return self.animacion.valor(ahora)

    def rotulo(self):
        """(texto, color) de la franja de arriba: la senal neutra de la meta, o AUTOMATICO en un movimiento ajeno."""
        if self.ajeno:
            return 'AUTOMATICO', COLOR_AJENO
        if self.meta is None:
            return '', COLOR_TEXTO
        return cue.texto(self.meta, self.visual), COLOR_TEXTO         # el mismo color para las dos metas


# ---------------------------------------------------------------- dibujo contra un "pintor"
def dibujar(pintor, ancho, alto, estado, ahora, conectado=True):
    """Dibuja el cuadro en un pintor con: rellenar(color), poligono(puntos, relleno, borde, grosor_px),
    cadena(puntos, grosor_px, relleno, borde), texto(txt, x, y, px, color). Coordenadas en pixeles (y hacia abajo)."""
    xmin, ymin, xmax, ymax = CAJA
    alto_util = alto * 0.76                                  # arriba queda la franja del texto, abajo la barra
    esc = min(ancho / (xmax - xmin), alto_util / (ymax - ymin))
    ox = ancho / 2 - esc * (xmin + xmax) / 2
    oy = alto * 0.13 + alto_util / 2 + esc * (ymin + ymax) / 2
    px = lambda p: (ox + esc * p[0], oy - esc * p[1])
    pintor.rellenar(COLOR_FONDO)
    cierre = estado.cierre(ahora)
    for f in primitivas(cierre):
        if f[0] == 'poligono':
            _, pts, relleno, borde, g = f
            pintor.poligono([px(p) for p in pts], relleno, borde, g * esc)
        else:
            _, pts, g, relleno, borde = f
            pintor.cadena([px(p) for p in pts], g * esc, relleno, borde)
    txt, color = estado.rotulo()
    if not conectado:
        txt, color = 'Esperando al orquestador...', COLOR_TEXTO
    if txt:
        pintor.texto(txt, ancho / 2, alto * 0.07, int(alto * 0.07), color)
    # barra de cierre: 0 % abierta .. 100 % cerrada
    x0, x1, y = ancho * 0.25, ancho * 0.75, alto * 0.965
    pintor.poligono([(x0, y - 4), (x1, y - 4), (x1, y + 4), (x0, y + 4)], '#2a323d', '#2a323d', 1)
    llena = x0 + (x1 - x0) * cierre
    pintor.poligono([(x0, y - 4), (llena, y - 4), (llena, y + 4), (x0, y + 4)], COLOR_BANDA, COLOR_BANDA, 1)
    pintor.texto(f'{cierre * 100:.0f} % cerrada', ancho / 2, alto * 0.935, int(alto * 0.03), COLOR_TEXTO)


class PintorRegistro:
    """Anota lo que se le pide dibujar (pruebas)."""

    def __init__(self):
        self.ordenes = []

    def rellenar(self, color):
        self.ordenes.append(('rellenar', color))

    def poligono(self, puntos, relleno, borde, grosor):
        self.ordenes.append(('poligono', list(puntos), relleno, borde, grosor))

    def cadena(self, puntos, grosor, relleno, borde):
        self.ordenes.append(('cadena', list(puntos), grosor, relleno, borde))

    def texto(self, txt, x, y, px, color):
        self.ordenes.append(('texto', txt, x, y, px, color))


def imagen(estado, ruta, ancho=900, alto=900, ahora=0.0):
    """Guarda el cuadro como imagen con matplotlib (no necesita Qt): para ver como luce y para las capturas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    class PintorMpl:
        def __init__(self, ax):
            self.ax = ax

        def rellenar(self, color):
            self.ax.add_patch(Polygon([(0, 0), (ancho, 0), (ancho, alto), (0, alto)], closed=True, fc=color, ec='none'))

        def poligono(self, puntos, relleno, borde, grosor):
            self.ax.add_patch(Polygon(puntos, closed=True, fc=relleno, ec=borde, lw=max(0.5, grosor * 72 / 100),
                                      joinstyle='round'))

        def cadena(self, puntos, grosor, relleno, borde):
            xs, ys = zip(*puntos)
            pt = lambda px_: px_ * 72 / 100                   # pixeles a puntos con dpi = 100
            self.ax.plot(xs, ys, color=borde, lw=pt(grosor * 1.18), solid_capstyle='round', solid_joinstyle='round')
            self.ax.plot(xs, ys, color=relleno, lw=pt(grosor), solid_capstyle='round', solid_joinstyle='round')

        def texto(self, txt, x, y, px, color):
            self.ax.text(x, y, txt, color=color, ha='center', va='center', fontsize=px * 72 / 100, fontweight='bold',
                         family='monospace')

    fig = plt.figure(figsize=(ancho / 100, alto / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ancho)
    ax.set_ylim(alto, 0)
    ax.axis('off')
    dibujar(PintorMpl(ax), ancho, alto, estado, ahora)
    fig.savefig(ruta, dpi=100)
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------- espejo de la ortesis
def espejar(ortesis, publicar, latencia_s=None):
    """Hace que cada orden `mover` de la ortesis (real o simulada) se publique tambien en Estado como evento
    'mano', con el momento en que debe empezar a verse el movimiento: el ACK mas, si se da, `latencia_s(seq)`
    (la latencia mecanica de la ortesis simulada). Cambia solo el metodo `mover` de ESA instancia: todo lo demas
    (lecturas, ACK, telemetria, isinstance, el caos, la reanudacion) sigue siendo de la ortesis. Un fallo al
    publicar nunca detiene la sesion. Devuelve la misma ortesis."""
    original = ortesis.mover

    def mover(fraccion, dur_ms=config.DURACION_PASO_MS):
        r = original(fraccion, dur_ms)
        try:
            seq, t_ack = r[0], r[1]
            inicio = None if t_ack is None else float(t_ack) + (float(latencia_s(seq)) if latencia_s else 0.0)
            publicar(tipo=config.MANO_VIRTUAL['evento'], angulo=float(fraccion), ms=float(dur_ms), seq=seq,
                     inicio=inicio, ack=t_ack is not None)
        except Exception:
            pass
        return r
    ortesis.mover = mover
    return ortesis


# ---------------------------------------------------------------- ventana Qt
def _crear_ventana(estado, demo=False, ventana=False, pantalla=None):
    """La ventana Qt. Se importa Qt aqui: el resto del modulo (y el orquestador) no lo necesita."""
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    from pylsl import StreamInlet, resolve_byprop

    class PintorQt:
        def __init__(self, p):
            self.p = p

        def rellenar(self, color):
            self.p.fillRect(self.p.viewport(), QtGui.QColor(color))

        def _pluma(self, color, grosor):
            return QtGui.QPen(QtGui.QColor(color), max(1.0, grosor), QtCore.Qt.SolidLine, QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin)

        def poligono(self, puntos, relleno, borde, grosor):
            self.p.setPen(self._pluma(borde, grosor))
            self.p.setBrush(QtGui.QBrush(QtGui.QColor(relleno)))
            self.p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in puntos]))

        def cadena(self, puntos, grosor, relleno, borde):
            ruta = QtGui.QPainterPath()
            ruta.moveTo(*puntos[0])
            for x, y in puntos[1:]:
                ruta.lineTo(x, y)
            self.p.setBrush(QtCore.Qt.NoBrush)
            for color, g in ((borde, grosor * 1.18), (relleno, grosor)):      # contorno y relleno
                self.p.setPen(self._pluma(color, g))
                self.p.drawPath(ruta)

        def texto(self, txt, x, y, px, color):
            fuente = QtGui.QFont('Consolas', 10)
            fuente.setPixelSize(max(8, px))
            fuente.setBold(True)
            self.p.setFont(fuente)
            self.p.setPen(QtGui.QColor(color))
            self.p.drawText(QtCore.QRectF(x - 2000, y - px, 4000, 2 * px), QtCore.Qt.AlignCenter, txt)

    class Ventana(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('ortesis-bci · Mano virtual')
            self.estado, self.demo, self.inlet, self._encontrado, self._buscando = estado, demo, None, None, False
            self.t_demo = 0.0
            self.setStyleSheet(f'background:{COLOR_FONDO};')
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self._cuadro)
            self.timer.start(8)                                 # lee el flujo y dibuja a ~120 Hz: el inicio no se retrasa

        def _conectar(self):
            if self._buscando:
                return
            self._buscando = True

            def buscar():
                try:
                    s = resolve_byprop('name', 'Estado', timeout=3.0)
                    if s:
                        self._encontrado = StreamInlet(s[0], max_buflen=60)
                finally:
                    self._buscando = False
            threading.Thread(target=buscar, daemon=True).start()

        def _leer(self):
            if self.inlet is None:
                if self._encontrado is not None:
                    self.inlet = self._encontrado
                else:
                    self._conectar()
                return
            while True:
                m, _ = self.inlet.pull_sample(timeout=0.0)
                if m is None:
                    return
                try:
                    self.estado.procesar(json.loads(m[0]), _ahora())
                except (ValueError, KeyError, TypeError):
                    pass                                          # un evento raro no tumba la pantalla

        def _guion_demo(self, ahora):
            """--demo: se cierra y se abre sola cada 1.8 s, con la senal de la meta."""
            if ahora - self.t_demo >= 1.8:
                self.t_demo = ahora
                cerrar = self.estado.animacion.destino() < 0.5
                self.estado.procesar({'tipo': 'cue', 'meta': 1 if cerrar else -1}, ahora)
                self.estado.procesar({'tipo': config.MANO_VIRTUAL['evento'], 'angulo': 1.0 if cerrar else 0.0,
                                      'ms': config.DURACION_PASO_MS}, ahora + 0.3)

        def _cuadro(self):
            if self.demo:
                self._guion_demo(_ahora())
            else:
                self._leer()
            self.update()

        def paintEvent(self, _):
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            dibujar(PintorQt(p), self.width(), self.height(), self.estado, _ahora(), conectado=self.demo or self.inlet is not None)
            p.end()

        def keyPressEvent(self, ev):
            if ev.key() == QtCore.Qt.Key_Escape:
                self.close()
            elif ev.key() == QtCore.Qt.Key_F:
                self.showNormal() if self.isFullScreen() else self.showFullScreen()

    v = Ventana()
    pantallas = QtWidgets.QApplication.screens()
    if pantalla is not None and 0 <= pantalla < len(pantallas):
        g = pantallas[pantalla].geometry()
        v.move(g.x(), g.y())
    if ventana:
        v.resize(900, 900)
        v.show()
    else:
        v.showFullScreen()
    return v


def _ahora():
    """La hora en la que se comparan los 'inicio' del orquestador: el reloj de LSL de esta maquina."""
    from pylsl import local_clock
    return local_clock()


def main(argv=None):
    ap = argparse.ArgumentParser(description='Mano virtual a pantalla completa, movida por las mismas ordenes que la ortesis')
    ap.add_argument('--ventana', action='store_true', help='en una ventana normal en lugar de pantalla completa')
    ap.add_argument('--pantalla', type=int, default=None, help='numero de la pantalla donde abrir (0 = la principal)')
    ap.add_argument('--demo', action='store_true', help='se abre y se cierra sola, sin orquestador')
    ap.add_argument('--captura', metavar='RUTA', help='guarda una imagen de la mano y sale (no abre ventana)')
    ap.add_argument('--cierre', type=float, default=0.0, help='con --captura: cierre de la mano, 0 (abierta) a 1 (cerrada)')
    a = ap.parse_args(argv)
    estado = EstadoMano()
    if a.captura:
        estado.animacion = Animacion(a.cierre)
        estado.meta = 1 if a.cierre >= 0.5 else -1
        print('Imagen guardada en', imagen(estado, a.captura))
        return 0
    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication(sys.argv[:1])
    v = _crear_ventana(estado, demo=a.demo, ventana=a.ventana, pantalla=a.pantalla)      # noqa: F841 (la referencia mantiene viva la ventana)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())

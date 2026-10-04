"""Tablero en vivo: escucha el flujo 'Estado' del orquestador y dibuja 5 paneles.

  1. p cruda del decoder y umbral b del agente
  2. cierre de la ortesis contra la meta
  3. P_hat por paso (rojo = el paso fue erroneo)
  4. error movil del agente contra el decoder en la sombra
  5. beta +- 2 desviaciones, confianza viva del detector y cambios detectados

Con --sham (control causal) una linea dice que bloque va, solo con su letra (A o B): cual es el
real y cual el sham se ve al pulsar 'Revelar bloques'. Al terminar compara los dos lado a lado.

En la cabecera, tres semaforos de salud (EEG, ortesis, detector; gris = el detector
aun calienta). En PAUSA_SEGURA el estado se pone en rojo y dice el motivo y, si es
un electrodo, cual falla y por que.

Con --copiloto, una caja de texto para preguntarle al copiloto clinico (copiloto.py) sobre la
sesion mas reciente; responde en otro hilo, sin detener el tablero.

Con --flechas, un sexto panel con la contribucion de cada ErrP al cambio de beta: una flecha por paso,
hacia arriba si empuja a beta hacia CERRAR y hacia abajo si hacia RELAJAR, tan larga como el cambio;
violeta si el detector marco ese paso como ErrP, gris si no. Sale de la diferencia de beta entre pasos
consecutivos del flujo 'Estado' (no cambia el contrato). En el control causal (--sham) queda en blanco
hasta pulsar 'Revelar bloques' (el tamano de las flechas delataria el bloque sham).

Con --narrador, una franja con la ultima frase del narrador (narrador.py, flujo 'Narracion').

Con --estado-sistema, una franja con el estado del sistema (estado_sistema.py): casco, flujos LSL,
procesos, latencia del ACK, laptop, disco, llave de la API y version de git; cada falla con su
solucion. Se revisa en otro hilo cada config.ESTADO_SISTEMA['periodo_s'] segundos.

Con orquestador.py --coinvestigador, al terminar un bloque aparece la propuesta con los botones
Aprobar y Rechazar: la decision se escribe en el registro de propuestas de la sesion.

Uso:  python tablero.py                     (arrancalo antes o despues del orquestador)
      python tablero.py --copiloto          (con la caja de preguntas)
      python tablero.py --flechas           (con el panel de la contribucion de cada ErrP a beta)
      python tablero.py --narrador          (con la franja del narrador; correr tambien narrador.py)
      python tablero.py --estado-sistema    (con la franja del estado del sistema)
      python tablero.py --captura fig.png --segundos 20   (guarda una imagen y sale)
"""
import argparse
import html
import json
import sys
import threading
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pylsl import StreamInlet, resolve_byprop

import config

N = 300          # pasos visibles
COLORES_SALUD = {config.VERDE: '#2ca02c', config.AMARILLO: '#e6b800', config.ROJO: '#d62728',
                 config.CALENTANDO: '#9e9e9e'}
SEMAFOROS = {'eeg': 'EEG', 'ortesis': 'ORTESIS', 'detector': 'DETECTOR', 'piloto': 'PILOTO'}
VENTANA = 20     # para el error movil
COLORES_SISTEMA = {'OK': COLORES_SALUD[config.VERDE], 'AVISO': COLORES_SALUD[config.AMARILLO],
                   'FALLA': COLORES_SALUD[config.ROJO]}

TITULO_FLECHAS = 'contribucion de cada ErrP a beta (flecha arriba = hacia CERRAR, abajo = hacia RELAJAR; violeta = ErrP detectado)'
COLOR_ERRP, COLOR_SIN_ERRP = '#9467bd', '#9e9e9e'    # flechas: el detector marco ErrP / no lo marco


def contribucion_beta(beta_anterior, e, e_anterior=None):
    """Cambio de beta que dejo este paso (+ hacia CERRAR, - hacia RELAJAR); 0 si no hay con que
    compararlo: primer paso, sesion nueva (el numero de paso retrocede) o cambio de bloque del control
    causal (el agente se reinicia y beta vuelve a 0 sin que ningun ErrP lo mueva)."""
    if beta_anterior is None or e_anterior is None:
        return 0.0
    if e['paso'] <= e_anterior['paso'] or e.get('bloque') != e_anterior.get('bloque'):
        return 0.0
    return float(e['beta'] - beta_anterior)


class Tablero(QtWidgets.QWidget):
    def __init__(self, copiloto=False, narrador=False, flechas=False, estado_sistema=False):
        super().__init__()
        self.setWindowTitle('ortesis-bci · Tablero')
        self.resize(1200, 900)
        pg.setConfigOptions(antialias=True, background='w', foreground='k')
        lay = QtWidgets.QVBoxLayout(self)

        cab = QtWidgets.QHBoxLayout()
        self.lbl_estado = QtWidgets.QLabel('Esperando al orquestador...')
        self.lbl_cue = QtWidgets.QLabel('')
        self.lbl_cp = QtWidgets.QLabel('')
        for w, tam in ((self.lbl_estado, 16), (self.lbl_cue, 28), (self.lbl_cp, 12)):
            w.setStyleSheet(f'font-size:{tam}px; font-weight:bold; padding:4px;')
        cab.addWidget(self.lbl_estado, 3)
        cab.addWidget(self.lbl_cue, 2)
        self.semaforos = {}
        for sub, texto in SEMAFOROS.items():
            self.semaforos[sub] = QtWidgets.QLabel(texto)
            self.semaforos[sub].setAlignment(QtCore.Qt.AlignCenter)
            cab.addWidget(self.semaforos[sub], 1)
        self._semaforos({'eeg': config.VERDE, 'ortesis': config.VERDE, 'detector': config.CALENTANDO,
                         'piloto': config.CALENTANDO}, {})
        lay.addLayout(cab)
        # narrador para el jurado (--narrador): la ultima frase del flujo Narracion
        self.narracion, self._narr_encontrado, self._narr_buscando = None, None, False
        self.lbl_narrador = QtWidgets.QLabel('')
        self.lbl_narrador.setWordWrap(True)
        self.lbl_narrador.setStyleSheet('font-size:20px; font-weight:bold; color:#1a3c6e; background:#eef3fb; '
                                        'border-radius:8px; padding:8px 12px;')
        self.lbl_narrador.setVisible(narrador)
        self.con_narrador = narrador
        lay.addWidget(self.lbl_narrador)
        # estado del sistema (--estado-sistema): una ficha por revision y, debajo, cada falla con su solucion
        self._sistema, self.latencias = None, deque(maxlen=config.CP1_MOVIMIENTOS)
        self.lbl_sistema = QtWidgets.QLabel('<b>SISTEMA</b> &nbsp; revisando...')
        self.lbl_sistema.setWordWrap(True)
        self.lbl_sistema.setStyleSheet('font-size:11px; background:#f4f4f4; border-radius:6px; padding:4px 8px;')
        self.lbl_sistema.setVisible(estado_sistema)
        lay.addWidget(self.lbl_sistema)
        if estado_sistema:
            threading.Thread(target=self._vigilar_sistema, daemon=True).start()
        lay.addWidget(self.lbl_cp)
        # Tarea 2: una linea con el IIC (exploratorio) en lugar de un panel: con ~12 movimientos
        # ajenos por sesion su curva seria casi toda ruido; el numero con su intervalo basta
        self.lbl_iic = QtWidgets.QLabel('')
        self.lbl_iic.setStyleSheet('font-size:12px; color:#555; padding:2px 4px;')
        lay.addWidget(self.lbl_iic)
        # control causal (--sham): ciego hasta que el operador pide ver cual bloque es cual
        fila = QtWidgets.QHBoxLayout()
        self.lbl_sham = QtWidgets.QLabel('')
        self.lbl_sham.setStyleSheet('font-size:13px; font-weight:bold; padding:2px 4px;')
        self.btn_sham = QtWidgets.QPushButton('Revelar bloques')
        self.btn_sham.setCheckable(True)
        self.btn_sham.setVisible(False)
        self.btn_sham.toggled.connect(lambda _: self._sham())
        fila.addWidget(self.lbl_sham, 1)
        fila.addWidget(self.btn_sham)
        lay.addLayout(fila)
        self.sham_bloque, self.sham_comparacion = None, None
        # co-investigador: la propuesta entre bloques, con Aprobar y Rechazar (nada cambia sin una persona)
        fila = QtWidgets.QHBoxLayout()
        self.lbl_propuesta = QtWidgets.QLabel('')
        self.lbl_propuesta.setWordWrap(True)
        self.lbl_propuesta.setStyleSheet('font-size:13px; padding:2px 4px;')
        self.btn_aprobar, self.btn_rechazar = QtWidgets.QPushButton('Aprobar'), QtWidgets.QPushButton('Rechazar')
        fila.addWidget(self.lbl_propuesta, 1)
        for b, aprobar in ((self.btn_aprobar, True), (self.btn_rechazar, False)):
            b.setVisible(False)
            b.clicked.connect(lambda _, aprobar=aprobar: self._decidir(aprobar))
            fila.addWidget(b)
        lay.addLayout(fila)
        self.propuesta = None
        # copiloto clinico (--copiloto): una pregunta sobre la sesion mas reciente, respondida en otro hilo
        self._respuesta, self._pensando = None, False
        if copiloto:
            self.caja = QtWidgets.QLineEdit()
            self.caja.setPlaceholderText('Pregunta al copiloto sobre la sesion y pulsa Enter')
            self.caja.returnPressed.connect(self._preguntar)
            self.lbl_copiloto = QtWidgets.QLabel('')
            self.lbl_copiloto.setWordWrap(True)
            self.lbl_copiloto.setStyleSheet('font-size:12px; padding:2px 4px;')
            lay.addWidget(self.caja)
            lay.addWidget(self.lbl_copiloto)

        self.g = pg.GraphicsLayoutWidget()
        lay.addWidget(self.g)
        titulos = ['p cruda y umbral b', 'cierre vs meta', 'P_hat por paso',
                   f'error movil ({VENTANA} pasos)', 'beta, confianza del detector y cambios']
        self.con_flechas = flechas
        if flechas:
            titulos.append(TITULO_FLECHAS)
        self.p = []
        for i, t in enumerate(titulos):
            pl = self.g.addPlot(row=i, col=0, title=t)
            pl.showGrid(x=True, y=True, alpha=0.3)
            pl.addLegend(offset=(5, 5))
            if i:
                pl.setXLink(self.p[0])
            self.p.append(pl)
        pen = lambda c, w=2, s=QtCore.Qt.SolidLine: pg.mkPen(c, width=w, style=s)
        self.c_p = self.p[0].plot(pen=pen('#1f77b4'), name='p cruda')
        self.c_b = self.p[0].plot(pen=pen('#d62728', 2, QtCore.Qt.DashLine), name='umbral b')
        self.p[0].setYRange(0, 1)
        self.c_ang = self.p[1].plot(pen=pen('#2ca02c'), name='cierre')
        self.c_meta = self.p[1].plot(pen=pen('k', 1, QtCore.Qt.DashLine), name='meta')
        self.p[1].setYRange(-0.05, 1.05)
        self.c_ph = self.p[2].plot(pen=None, symbol='o', symbolSize=6, name='P_hat')
        self.p[2].setYRange(0, 1)
        self.c_ea = self.p[3].plot(pen=pen('#ff7f0e'), name='agente')
        self.c_es = self.p[3].plot(pen=pen('#7f7f7f'), name='sombra (sin aprender)')
        self.p[3].setYRange(0, 0.7)
        self.c_beta = self.p[4].plot(pen=pen('#1f77b4'), name='beta')
        self.c_sup = self.p[4].plot(pen=pen('#1f77b4', 1))
        self.c_inf = self.p[4].plot(pen=pen('#1f77b4', 1))
        self.p[4].addItem(pg.FillBetweenItem(self.c_sup, self.c_inf, brush=(31, 119, 180, 50)))
        self.c_ev = self.p[4].plot(pen=pen('#9467bd'), name='Youden vivo del detector')
        self.c_fi = self.p[4].plot(pen=pen('#8c564b', 1), name='fiabilidad (0 = congelado)')

        self.d = {k: deque(maxlen=N) for k in
                  ('paso', 'p', 'b', 'ang', 'meta', 'ph', 'err', 'es', 'beta', 'sd', 'ev', 'fi', 'db', 'det')}
        self._anterior = None                                    # el paso anterior, para la contribucion a beta
        if flechas:
            pl = self.p[5]
            pl.setLabel('left', 'cambio de beta')
            pl.getAxis('left').enableAutoSIPrefix(False)
            pl.setMinimumHeight(200)
            pl.addLine(y=0, pen=pg.mkPen('k', width=1))
            self.c_tallos = pg.PlotCurveItem(connect='pairs')
            self.c_puntas = pg.ScatterPlotItem(pen=None, size=11)
            pl.addItem(self.c_tallos)
            pl.addItem(self.c_puntas)
            self.lbl_flecha = QtWidgets.QLabel('')
            self.lbl_flecha.setStyleSheet('font-size:14px; font-weight:bold; padding:2px 4px;')
            lay.insertWidget(lay.indexOf(self.g), self.lbl_flecha)
        self.inlet, self._encontrado, self._buscando = None, None, False
        self._conectar()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._actualizar)
        self.timer.start(150)

    def _conectar(self):
        """Busca el flujo 'Estado' en un hilo aparte: en Windows, con varias tarjetas
        de red, el descubrimiento de LSL puede tardar mas de lo que la ventana aguanta
        congelada, asi que la ventana nunca se bloquea esperando."""
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

    def _vigilar_sistema(self):
        """En otro hilo: no toca la ventana, deja la revision para _actualizar(). Nunca tumba el tablero."""
        import estado_sistema as es
        monitor = es.Monitor(self.latencias)
        while True:
            try:
                self._sistema = monitor.ciclo()
            except Exception as e:
                self._sistema = [es.res('sistema', es.AVISO, f'la revision fallo: {type(e).__name__}: {e}',
                                        'corre python estado_sistema.py en otra terminal')]
            time.sleep(config.ESTADO_SISTEMA['periodo_s'])

    def _franja_sistema(self, resultados):
        c = COLORES_SISTEMA
        fichas = ' &nbsp; '.join(f'<span style="color:{c[r["estado"]]}">&#9679;</span> {r["clave"]}' for r in resultados)
        malas = ''.join(f'<br><b style="color:{c[r["estado"]]}">{r["estado"]}</b> {r["clave"]}: {html.escape(r["texto"])}'
                        f' &rarr; <i>{html.escape(r["solucion"])}</i>'
                        for r in sorted(resultados, key=lambda r: r['estado'] != 'FALLA') if r['estado'] != 'OK')
        self.lbl_sistema.setText('<b>SISTEMA</b> &nbsp; ' + fichas + malas)
        self.lbl_sistema.setToolTip('\n'.join(f"{r['estado']:6s} {r['clave']}: {r['texto']}" for r in resultados))

    def _decidir(self, aprobar):
        """El operador decide: queda escrito en el registro de propuestas, que el orquestador esta leyendo."""
        import copiloto
        e = self.propuesta
        copiloto.decidir(config.RESULTADOS / e['csv'], e['id'], aprobar, por='operador (tablero)')
        for b in (self.btn_aprobar, self.btn_rechazar):
            b.setVisible(False)
        self.lbl_propuesta.setText(self.lbl_propuesta.text() + '  ->  ' + ('aprobada' if aprobar else 'rechazada') + '...')

    def _preguntar(self):
        q = self.caja.text().strip()
        if q and not self._pensando:
            self._pensando = True
            self.lbl_copiloto.setText('El copiloto esta consultando la sesion...')
            threading.Thread(target=self._responder, args=(q,), daemon=True).start()

    def _responder(self, q):
        """En otro hilo: no toca la ventana, deja la respuesta para _mostrar_respuesta()."""
        try:
            import copiloto
            import ia
            ruta = max(config.RESULTADOS.glob('sesion_*.csv'), key=lambda p: p.stat().st_mtime)
            txt, origen, _ = copiloto.responder(copiloto.Sesion(ruta), q, ia.cliente())
            self._respuesta = f'[{ruta.name} · {origen}] {txt}'
        except Exception as e:                   # sin sesiones, CSV a medias...: nunca tumba el tablero
            self._respuesta = f'El copiloto no pudo responder: {e}'

    def _mostrar_respuesta(self):
        if self._respuesta is not None:
            self.lbl_copiloto.setText(self._respuesta)
            self._respuesta, self._pensando = None, False

    def _narrar(self):
        """Lee el flujo Narracion (lo busca en otro hilo, como _conectar) y muestra la ultima frase."""
        if self.narracion is None:
            if self._narr_encontrado is not None:
                self.narracion = self._narr_encontrado
            elif not self._narr_buscando:
                self._narr_buscando = True

                def buscar():
                    try:
                        s = resolve_byprop('name', 'Narracion', timeout=3.0)
                        if s:
                            self._narr_encontrado = StreamInlet(s[0], max_buflen=60)
                    finally:
                        self._narr_buscando = False
                threading.Thread(target=buscar, daemon=True).start()
            return
        while True:
            m, _ = self.narracion.pull_sample(timeout=0.0)
            if m is None:
                break
            self._frase(json.loads(m[0]))

    def _frase(self, d):
        self.lbl_narrador.setText(d['texto'])

    def _actualizar(self):
        self._mostrar_respuesta()
        if self._sistema is not None:
            self._franja_sistema(self._sistema)
            self._sistema = None
        if self.con_narrador:
            self._narrar()
        if self.inlet is None:
            if self._encontrado is not None:
                self.inlet = self._encontrado
                self.lbl_estado.setText('Conectado al orquestador')
            else:
                self._conectar()
            return
        nuevos = False
        while True:
            m, _ = self.inlet.pull_sample(timeout=0.0)
            if m is None:
                break
            self._procesar(json.loads(m[0]))
            nuevos = True
        if nuevos:
            self._dibujar()

    def _semaforos(self, colores, detalle):
        for sub, lbl in self.semaforos.items():
            color = COLORES_SALUD.get(colores.get(sub), COLORES_SALUD[config.CALENTANDO])
            lbl.setStyleSheet(f'font-size:13px; font-weight:bold; color:white; background:{color}; '
                              f'border-radius:8px; padding:4px 10px;')
            lbl.setToolTip(detalle.get(sub, ''))

    def _procesar(self, e):
        tipo = e.get('tipo')
        if tipo == 'salud':
            self._semaforos(e['colores'], e['detalle'])
            if e['estado'] == 'PAUSA_SEGURA':
                motivo = e.get('motivo') or ''
                detalle = e['detalle'].get('ortesis' if motivo == 'ortesis' else 'eeg', '')
                self.lbl_estado.setText(' · '.join(x for x in ('PAUSA SEGURA', motivo, detalle) if x))
                self.lbl_estado.setStyleSheet(f'font-size:16px;font-weight:bold;padding:4px;'
                                              f'color:{COLORES_SALUD[config.ROJO]};')
        elif tipo == 'cue':
            self._cue(e['meta'])
        elif tipo == 'aviso_ajeno':                              # Tarea 2: la ortesis se movera sola
            self.lbl_cue.setText('AUTOMATICO')
            self.lbl_cue.setStyleSheet('font-size:28px;font-weight:bold;padding:4px;color:#7f3fbf;')
        elif tipo in ('ajeno', 'iic'):
            if tipo == 'ajeno':
                self._cue(e['meta'])                             # vuelve la meta del ensayo
            self._iic(e['iic'])
        elif tipo == 'propuesta':
            self.propuesta = e
            p = e['propuesta']
            self.lbl_propuesta.setText(
                f"Co-investigador ({'Claude' if e['origen'] == 'api' else 'reglas'}) tras {e['bloque']}: "
                f"{p['accion'].replace('_', ' ').upper()}" + (f" {p['parametro']} = {p['valor']}" if p['valor'] is not None else '')
                + f". {p['justificacion']}" + ('' if e['valida'] else ' [INVALIDA: no se puede aprobar]'))
            for b in (self.btn_aprobar, self.btn_rechazar):
                b.setVisible(bool(e['valida'] and e['espera_s'] > 0))
        elif tipo == 'decision':
            if self.propuesta and self.propuesta['id'] == e['id']:
                self.lbl_propuesta.setText(self.lbl_propuesta.text().split('  ->  ')[0]
                                           + f"  ->  {e['decision'].upper()}: {e['efecto']}")
                for b in (self.btn_aprobar, self.btn_rechazar):
                    b.setVisible(False)
        elif tipo == 'bloque_sham':
            self.sham_bloque, self.sham_comparacion = e, None
            self._sham()
        elif tipo == 'sham':
            self.sham_comparacion = e
            self._sham()
        elif tipo == 'checkpoint':
            color = '#2ca02c' if e['ok'] else '#d62728'
            self.lbl_cp.setText(e['texto'])
            self.lbl_cp.setStyleSheet(f'font-size:12px;font-weight:bold;color:{color};')
        elif tipo == 'paso':
            d = self.d
            if 'salud' in e:                                     # las sesiones viejas no lo traen
                self._semaforos(e['salud'], {})
            if d['paso'] and e['paso'] < d['paso'][-1]:          # nueva sesion
                for q in d.values():
                    q.clear()
                self._anterior = None
            db = contribucion_beta(None if self._anterior is None else self._anterior['beta'], e, self._anterior)
            d['db'].append(db); d['det'].append(e.get('detectado'))
            self._anterior = e
            d['paso'].append(e['paso']); d['p'].append(e['p_crudo']); d['b'].append(e['b'])
            d['ang'].append(e['angulo']); d['meta'].append(1.0 if e['meta'] > 0 else 0.0)
            d['ph'].append(np.nan if e['P_hat'] is None else e['P_hat'])
            d['err'].append(e['error']); d['es'].append(e['error_sombra'])
            d['beta'].append(e['beta']); d['sd'].append(e['sd_beta']); d['ev'].append(e['youden'])
            d['fi'].append(e['fiabilidad'])
            if e.get('cambio'):
                color = '#9467bd' if e['cambio'] == 'sesgo' else '#8c564b'
                self.p[4].addItem(pg.InfiniteLine(e['paso'], angle=90, pen=pg.mkPen(color, width=1)))
            if not e.get('perturbado'):                          # cada bloque del control causal trae la suya
                self._pert_marcada = False
            if e.get('perturbado') and not getattr(self, '_pert_marcada', False):
                for pl in self.p:
                    pl.addItem(pg.InfiniteLine(e['paso'], angle=90,
                                               pen=pg.mkPen('r', width=1, style=QtCore.Qt.DotLine)))
                self._pert_marcada = True
            self.latencias.append(float('nan') if e['latencia_ms'] is None else e['latencia_ms'])
            if 'iic' in e:                                       # las sesiones viejas no lo traen
                self._iic(e['iic'])
            congel = ' · APRENDIZAJE CONGELADO' if e['congelado'] else ''
            ack = 'sin ACK' if e['latencia_ms'] is None else f"ACK {e['latencia_ms']:.1f} ms"
            self.lbl_estado.setText(f"{e['estado']} · paso {e['paso']} · beta {e['beta']:+.2f} · "
                                    f"{ack}{congel}")
            self.lbl_estado.setStyleSheet('font-size:16px;font-weight:bold;padding:4px;' +
                                          ('color:#d62728;' if e['congelado'] else ''))

    def _sham(self):
        """Control causal: durante el ciego solo la letra del bloque; con 'Revelar bloques', cual es cual."""
        self.btn_sham.setVisible(True)
        ver = self.btn_sham.isChecked()
        c = self.sham_comparacion
        if c is None:
            b = self.sham_bloque
            self.lbl_sham.setText(f"Control causal · bloque {b['letra']}" + (f" = {b['nombre'].upper()}" if ver else '')
                                  + f" · {b['pasos']} pasos")
            return
        partes = []
        for letra, nombre in zip('AB', c['orden']):
            r = c.get(nombre)
            if r:
                rec = f"se recupero en {r['pasos']} pasos ({r['seg']:.0f} s)" if r['pasos'] else 'NO se recupero'
                partes.append(f"Bloque {letra}" + (f" = {nombre.upper()}" if ver else '')
                              + f": error {r['agente']:.2f} [{r['ic_agente'][0]:.2f}, {r['ic_agente'][1]:.2f}], {rec}")
        if ver and c.get('ic_dif'):
            partes.append(f"sham - real {c['dif']:+.2f} [{c['ic_dif'][0]:+.2f}, {c['ic_dif'][1]:+.2f}]")
        self.lbl_sham.setText('Control causal · ' + '   |   '.join(partes))

    def _cue(self, meta):
        cerrar = meta > 0
        self.lbl_cue.setText('CERRAR' if cerrar else 'RELAJA')
        self.lbl_cue.setStyleSheet(f'font-size:28px;font-weight:bold;padding:4px;'
                                   f'color:{"#d62728" if cerrar else "#1f77b4"};')

    def _iic(self, r):
        if not r or r.get('iic') is None:
            n = (r or {}).get('n_ajenos', 0)
            self.lbl_iic.setText(f'IIC (exploratorio): sin estimar todavia ({n} movimientos ajenos)')
            return
        ic = ' [{:+.2f}, {:+.2f}]'.format(*r['ic']) if r.get('ic') else ''
        self.lbl_iic.setText(f"IIC (exploratorio, atenuacion de la N1 en PO7/Oz/PO8): {r['iic']:+.2f}{ic}"
                             f" · {r['n_ajenos']} movimientos ajenos, {r['n_propios']} propios correctos")

    def _dibujar(self):
        d = {k: np.array(v, dtype=float) for k, v in self.d.items()}
        x = d['paso']
        if len(x):
            # modo "seguir": el eje x siempre muestra los ultimos pasos, aunque alguien
            # haya hecho zoom o scroll con el mouse (eso apaga el auto-rango de pyqtgraph)
            self.p[0].setXRange(max(x[0], x[-1] - N), x[-1] + 1, padding=0.02)
            self.p[4].enableAutoRange(axis='y')
        self.c_p.setData(x, d['p'])
        self.c_b.setData(x, d['b'])
        self.c_ang.setData(x, d['ang'])
        self.c_meta.setData(x, d['meta'], stepMode=None)
        colores = [pg.mkBrush('#d62728' if v else '#2ca02c') for v in d['err']]
        ok = np.isfinite(d['ph'])
        self.c_ph.setData(x[ok], d['ph'][ok], symbolBrush=[c for c, o in zip(colores, ok) if o])
        if len(x) >= 2:
            k = np.ones(min(VENTANA, len(x))) / min(VENTANA, len(x))
            ea = np.convolve(d['err'], k, mode='valid')
            es = np.convolve(d['es'], k, mode='valid')
            xx = x[len(x) - len(ea):]
            self.c_ea.setData(xx, ea)
            self.c_es.setData(xx, es)
        self.c_beta.setData(x, d['beta'])
        self.c_sup.setData(x, d['beta'] + 2 * d['sd'])
        self.c_inf.setData(x, d['beta'] - 2 * d['sd'])
        self.c_ev.setData(x, d['ev'])
        self.c_fi.setData(x, d['fi'])
        if self.con_flechas:
            self._dibujar_flechas(x, d)

    def _flechas_visibles(self):
        """En el control causal ciego, el tamano de las flechas delataria el bloque sham."""
        return self.sham_bloque is None or self.btn_sham.isChecked()

    def _dibujar_flechas(self, x, d):
        pl = self.p[5]
        if not self._flechas_visibles():
            self.c_tallos.setData([], [])
            self.c_puntas.setData([], [])
            pl.setTitle(TITULO_FLECHAS + ' (oculto: control causal ciego)')
            self.lbl_flecha.setText('')
            return
        pl.setTitle(TITULO_FLECHAS)
        db = d['db']
        hay = np.abs(db) > 1e-9
        xs, ys = x[hay], db[hay]
        tallo_x, tallo_y = np.repeat(xs, 2), np.column_stack([np.zeros(len(ys)), ys]).ravel()
        self.c_tallos.setData(tallo_x, tallo_y, pen=pg.mkPen('#555555', width=2))
        det = np.array([bool(v) for v in d['det']], dtype=bool)[hay] if len(d['det']) else np.zeros(0, bool)
        self.c_puntas.setData(
            xs, ys, symbol=['t1' if v > 0 else 't' for v in ys],
            brush=[pg.mkBrush(COLOR_ERRP if v else COLOR_SIN_ERRP) for v in det])
        tope = max(0.2, 1.2 * float(np.abs(db).max())) if len(db) else 0.2
        pl.setYRange(-tope, tope, padding=0)
        if len(x) and hay[-1] and self._anterior is not None:
            v = float(db[-1])
            origen = 'el ErrP detectado' if self.d['det'][-1] else 'la ausencia de ErrP'
            self.lbl_flecha.setText(f"Paso {int(x[-1])}: {origen} mueve beta {v:+.2f} hacia {'CERRAR' if v > 0 else 'RELAJAR'}")
        elif len(x):
            self.lbl_flecha.setText(f'Paso {int(x[-1])}: este paso no cambio beta (sin epoca util o aprendizaje congelado)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captura', help='guarda una imagen del tablero y sale')
    ap.add_argument('--segundos', type=float, default=15)
    ap.add_argument('--copiloto', action='store_true', help='caja de preguntas al copiloto clinico')
    ap.add_argument('--flechas', action='store_true', help='panel con la contribucion de cada ErrP al cambio de beta')
    ap.add_argument('--narrador', action='store_true', help='franja con las frases de narrador.py')
    ap.add_argument('--estado-sistema', dest='estado_sistema', action='store_true',
                    help='franja con el estado del sistema (estado_sistema.py), cada falla con su solucion')
    a = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    t = Tablero(copiloto=a.copiloto, narrador=a.narrador, flechas=a.flechas, estado_sistema=a.estado_sistema)
    t.show()
    if a.captura:
        def salir():
            t.grab().save(a.captura)
            app.quit()
        QtCore.QTimer.singleShot(int(a.segundos * 1000), salir)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

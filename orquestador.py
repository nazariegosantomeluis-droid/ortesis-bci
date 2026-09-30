"""Orquestador del lazo: maquina de estados + bucle + registro.
Hoy corre con el piloto simulado. En el hackathon se cambia el backend
por el real (EEG + ortesis + detector de B2) sin tocar el resto.

Uso:  python orquestador.py --ciclo 0          (rapido, para probar)
      python orquestador.py                    (a ritmo real, 2.1 s por paso)
      python orquestador.py --falla_detector
"""
import argparse
import csv
import time
from datetime import datetime
import numpy as np
from pylsl import StreamOutlet, local_clock
import config
from agente_errp import AgenteErrP, ConfigAgente, MonitorDetector
from simulador_lazo import PilotoSimulado, ControladorEstatico, calibrar


# ======================= Maquina de estados =======================
class MaquinaEstados:
    def __init__(self, publicar):
        self.estado = config.ESTADOS[0]
        self.publicar = publicar
        self.publicar(config.m_bloque(self.estado))

    def ir_a(self, nuevo):
        if nuevo == self.estado:
            return
        if nuevo not in config.TRANSICIONES[self.estado]:
            raise ValueError(f'Transicion no permitida: {self.estado} -> {nuevo}')
        print(f'  [estado] {self.estado} -> {nuevo}')
        self.estado = nuevo
        self.publicar(config.m_bloque(nuevo))


# ======================= Backends =======================
class BackendSimulado:
    """Piloto sintetico. Misma interfaz que tendra el backend real."""

    def __init__(self, sens, espec, semilla=0, falla=None):
        mk = lambda r: PilotoSimulado(sens=sens, espec=espec,
                                      semilla_sujeto=semilla, semilla_ruido=r)
        self.w0, self.c0 = calibrar(mk(99))        # simula CAL_MI de B1
        self.piloto = mk(semilla + 1)
        self.falla, self.t, self.seq = falla, 0, 0

    def revisar_impedancias(self):
        return True

    def leer_rasgos(self, meta):
        if self.falla:
            self.piloto.detector_degradado(self.falla[0] <= self.t < self.falla[1])
        self.t += 1
        return self.piloto.rasgos(meta)

    def enviar_paso(self, angulo, dur):
        self.seq += 1
        return self.seq                              # ACK inmediato

    def tomar_errp(self, seq, erroneo):
        return self.piloto.errp(erroneo)             # (p_errp, artefacto)

    def perturbar(self):
        self.piloto.perturbar(self.w0, config.PERTURBACION_LOGITS)

    def cerrar(self):
        pass


class BackendReal:
    """Se llena en el hackathon. Cada metodo dice quien lo provee."""

    def __init__(self):
        # B1: el decoder publica p(t) en el flujo 'Intencion'. El agente
        # trabaja sobre su logit, asi que w0 = [1], c0 = 0.
        self.w0, self.c0 = np.array([1.0]), 0.0
        raise NotImplementedError('BackendReal: se integra en el hackathon')

    def revisar_impedancias(self):   # B1
        ...
    def leer_rasgos(self, meta):     # B1: logit de la media de p(t) en VENTANA_MI
        ...
    def enviar_paso(self, angulo, dur):  # P2: "M,<seq>,<ang>,<dur>" y esperar ACK
        ...
    def tomar_errp(self, seq, erroneo):  # B2: epoca alrededor de paso_ack:<seq>
        ...
    def perturbar(self):             # sumar PERTURBACION_LOGITS al logit de B1
        ...
    def cerrar(self):
        ...


# ======================= Orquestador =======================
class Orquestador:
    def __init__(self, backend, sens, espec, eta, ciclo):
        self.b, self.ciclo = backend, ciclo
        self.out_marc = StreamOutlet(config.crear_info('Marcadores'))
        self.out_paso = StreamOutlet(config.crear_info('Paso'))
        self.fsm = MaquinaEstados(self.marcador)
        self.sens, self.espec = sens, espec
        self.cfg = ConfigAgente(eta_beta=eta, sens=sens, espec=espec)
        self.angulo, self.filas = 45.0, []
        config.RESULTADOS.mkdir(exist_ok=True)
        nombre = datetime.now().strftime('sesion_%Y%m%d_%H%M%S.csv')
        self.ruta_csv = config.RESULTADOS / nombre
        self.f_csv = open(self.ruta_csv, 'w', newline='')
        self.csv = csv.DictWriter(self.f_csv, fieldnames=config.COLUMNAS_CSV)
        self.csv.writeheader()

    def marcador(self, texto):
        self.out_marc.push_sample([texto], local_clock())

    # ---------- fases ----------
    def calibrar(self):
        if not self.b.revisar_impedancias():
            raise RuntimeError('Impedancias altas: no avanzar (checkpoint 1)')
        self.fsm.ir_a('CAL_MI')
        w0, c0 = self.b.w0, self.b.c0
        self.agente = AgenteErrP(w0, c0, self.cfg)
        self.sombra = ControladorEstatico(w0, c0)
        self.monitor = MonitorDetector(sensibilidad=self.sens,
                                       especificidad=self.espec)
        self.fsm.ir_a('CAL_ERRP')
        self.fsm.ir_a('LAZO_ESTATICO')

    def go_checkpoint3(self):
        ba = (self.sens + self.espec) / 2
        ok = ba >= config.BA_MIN and self.espec >= config.ESPEC_MIN
        print(f'  [CP3] BA del ErrP = {ba:.2f}, especificidad = {self.espec:.2f}'
              f' -> {"GO" if ok else "NO GO: plan B (sesion grabada)"}')
        return ok

    # ---------- un paso del lazo ----------
    def paso(self, meta, aprender):
        t0 = time.perf_counter()
        phi = self.b.leer_rasgos(meta)
        dec = self.agente.decidir(phi)
        sombra = self.sombra.decidir(phi)
        self.out_paso.push_sample([dec.p_prima, dec.direccion, dec.dtheta])

        self.angulo = float(np.clip(self.angulo + dec.direccion * dec.dtheta, 0, 90))
        seq = self.b.enviar_paso(self.angulo, dur=0.5)
        self.marcador(config.m_paso_ack(seq))

        erroneo = dec.direccion != meta
        p_errp, art = self.b.tomar_errp(seq, erroneo)
        fiab = self.monitor(erroneo, p_errp > 0.5, not art)
        info = self.agente.actualizar(p_errp, art, fiab if aprender else 0.0)

        # congelar / reanudar segun el CUSUM
        if aprender or self.fsm.estado == 'APRENDIZAJE_CONGELADO':
            if self.monitor.congelado and self.fsm.estado == 'LAZO_ADAPTATIVO':
                self.fsm.ir_a('APRENDIZAJE_CONGELADO')
            elif not self.monitor.congelado and self.fsm.estado == 'APRENDIZAJE_CONGELADO':
                self.fsm.ir_a('LAZO_ADAPTATIVO')

        fila = {
            't_iso': datetime.now().isoformat(timespec='milliseconds'),
            'estado': self.fsm.estado, 'meta': meta, 'angulo': round(self.angulo, 2),
            'p_prima': round(dec.p_prima, 4), 'direccion': dec.direccion,
            'dtheta': round(dec.dtheta, 3),
            'P_hat': '' if np.isnan(info['P_hat']) else round(info['P_hat'], 4),
            'artefacto': int(art), 'fiabilidad': round(fiab, 3),
            'beta': round(info['beta'], 4), 'error_verdadero': int(erroneo),
            'error_sombra': int(sombra.direccion != meta),
        }
        self.csv.writerow(fila)
        self.f_csv.flush()                   # si algo truena, el CSV ya esta
        self.filas.append(fila)

        espera = self.ciclo - (time.perf_counter() - t0)
        if espera > 0:
            time.sleep(espera)

    def bloque(self, n_pasos, aprender, pasos_ensayo=5, perturbar_en=None, rng=None):
        orden = []
        for t in range(n_pasos):
            if t % pasos_ensayo == 0:
                if not orden:
                    orden = list(rng.permutation([1, -1]))
                meta = int(orden.pop())
                self.marcador(config.CUE_CERRAR if meta > 0 else config.CUE_RELAJA)
            if perturbar_en is not None and t == perturbar_en:
                estado_previo = self.fsm.estado
                self.fsm.ir_a('PERTURBACION')
                self.marcador(config.PERTURBACION_ON)
                self.b.perturbar()
                self.fsm.ir_a('LAZO_ADAPTATIVO' if estado_previo == 'LAZO_ADAPTATIVO'
                              else 'APRENDIZAJE_CONGELADO')
            self.paso(meta, aprender)

    def resumen(self):
        e = np.array([f['error_verdadero'] for f in self.filas])
        s = np.array([f['error_sombra'] for f in self.filas])
        est = np.array([f['estado'] for f in self.filas])
        print('\n=== EVALUACION ===')
        for nombre in ['LAZO_ESTATICO', 'LAZO_ADAPTATIVO', 'APRENDIZAJE_CONGELADO']:
            m = est == nombre
            if m.any():
                print(f'  {nombre:22s} pasos={m.sum():4d}  error agente={e[m].mean():.2f}'
                      f'  error sombra={s[m].mean():.2f}')
        print(f'  beta final = {self.filas[-1]["beta"]}')
        print(f'  CSV: {self.ruta_csv}')

    def cerrar(self):
        self.f_csv.close()
        self.b.cerrar()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=['sim', 'real'], default='sim')
    ap.add_argument('--sens', type=float, default=config.SENS)
    ap.add_argument('--espec', type=float, default=config.ESPEC)
    ap.add_argument('--eta', type=float, default=config.ETA_BETA)
    ap.add_argument('--ciclo', type=float, default=config.CICLO_S,
                    help='segundos por paso (0 = lo mas rapido posible)')
    ap.add_argument('--pasos_estatico', type=int, default=60)
    ap.add_argument('--pasos_adaptativo', type=int, default=300)
    ap.add_argument('--semilla', type=int, default=0)
    ap.add_argument('--falla_detector', action='store_true')
    a = ap.parse_args()

    n_est = a.pasos_estatico
    falla = (n_est + 200, n_est + 240) if a.falla_detector else None
    backend = (BackendSimulado(a.sens, a.espec, a.semilla, falla)
               if a.backend == 'sim' else BackendReal())
    orq = Orquestador(backend, a.sens, a.espec, a.eta, a.ciclo)
    rng = np.random.default_rng(a.semilla + 100)
    try:
        orq.calibrar()
        print('Bloque LAZO_ESTATICO...')
        orq.bloque(n_est, aprender=False, rng=rng)
        if orq.go_checkpoint3():
            orq.fsm.ir_a('LAZO_ADAPTATIVO')
            print('Bloque LAZO_ADAPTATIVO (perturbacion al primer tercio)...')
            orq.bloque(a.pasos_adaptativo, aprender=True,
                       perturbar_en=a.pasos_adaptativo // 3, rng=rng)
    except KeyboardInterrupt:
        print('\nInterrumpido por el usuario.')
    finally:
        if orq.fsm.estado != 'EVALUACION' and 'EVALUACION' in \
                config.TRANSICIONES[orq.fsm.estado]:
            orq.fsm.ir_a('EVALUACION')
        if orq.filas:
            orq.resumen()
        orq.cerrar()


if __name__ == '__main__':
    main()
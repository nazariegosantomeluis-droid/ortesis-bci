# Firmware de la Órtesis Adaptrode (ESP32)

Esta carpeta tiene:

- **`main.py`** (versión 1.3) va en la ESP32 DevKit V1. Es **un solo programa con todo**: servos, nervio de luz, vibradores, paro, corriente, sensor de fuerza, Wi-Fi, cable USB y calibración.
- **`probar_esp32.py`** va en la laptop. Mueve y calibra la órtesis sin el casco, por Wi-Fi o por cable USB.
- **`ortesis_udp.py`** y **`ortesis_usb.py`** son clientes de referencia (`OrtesisUDP` y `OrtesisUSB`, con su propia interfaz: `mover(cierre=...)` devuelve un dict): latido, ACK, telemetría y paro. **El orquestador no los usa:** usa `hardware.OrtesisUDP` (`--ortesis-udp`) y `hardware.OrtesisUSB` (`--ortesis-usb COMx`), que hablan el mismo JSON con la interfaz del orquestador.
- **`pruebas/`** corre `main.py` en la laptop con el hardware simulado: `python pruebas/prueba_firmware.py` (Linux o Mac).
- **`CAMBIOS_v1.3.diff`** muestra qué cambió desde la 1.2, para revisarlo antes de aprobarlo.

## 1. Cargar `main.py` en la ESP32 (con Thonny)

1. Conecta la DevKit a la laptop con un cable USB **de datos**. Si la laptop no la reconoce, instala el driver del chip que está junto al conector: **CP210x** o **CH340**.
2. En Thonny: *Herramientas → Opciones → Intérprete*. Elige **MicroPython (ESP32)** y el puerto de la placa.
3. La primera vez, en esa misma ventana entra a *Instalar o actualizar MicroPython* y escoge la variante **ESP32 / WROOM**. Si se queda en «Connecting…», mantén apretado el botón **BOOT** de la placa.
4. Abre `main.py`, elige *Guardar como → Dispositivo MicroPython* y guárdalo con el nombre `main.py`.
5. Presiona el botón **EN** (reset). En la consola debe salir: `Adaptrode 1.3 · DEVKIT_V1 · red Adaptrode · IP 192.168.4.1 · canal 6 · USB sí`, y la tira debe prender en **ámbar**.

Si usas la ESP32-C3 SuperMini, cambia la línea `PLACA = 'DEVKIT_V1'` por `PLACA = 'C3_SUPERMINI'`.

**Con VS Code en lugar de Thonny (extensión MicroPico):**

1. Instala la extensión **MicroPico** y abre esta carpeta (`firmware_esp32`) en VS Code.
2. Paleta de comandos (Ctrl+Shift+P) → *MicroPico: Configure project*. MicroPico se conecta solo al puerto de la placa; si no, *MicroPico: Connect* y elige el COM.
3. Clic derecho sobre `main.py` → *Upload current file to Pico* (el nombre dice «Pico», pero sirve con la ESP32 que ya tiene MicroPython). Se guarda como `main.py` en la placa.
4. *MicroPico: Reset → Soft reset* (o el botón **EN**) y en la terminal de MicroPico debe salir el mismo saludo `Adaptrode 1.3 ...`.
5. **Antes de usar la órtesis desde la laptop, desconecta MicroPico** (*MicroPico: Disconnect*, o cierra VS Code): mientras está conectado se queda con el puerto. MicroPython hay que instalarlo antes en la placa (paso 3 de Thonny, o con `esptool`).

### Si la laptop no ve la placa

- **El cable.** Muchos cables USB solo cargan: no tienen las líneas de datos. Si la placa prende pero no aparece ningún puerto nuevo, prueba con otro cable (uno que sepas que pasa datos, por ejemplo el de un teléfono con el que copias archivos).
- **El driver.** Mira el chip junto al conector USB de la placa: **CP2102** (o CP210x) o **CH340**. En Windows instala su driver (Silicon Labs «CP210x VCP» o WCH «CH341SER») y vuelve a conectar. En el Administrador de dispositivos, *Puertos (COM y LPT)*, debe aparecer «Silicon Labs CP210x ... (COMx)» o «USB-SERIAL CH340 (COMx)»; ese número es el que va en `--usb COMx` y `--ortesis-usb COMx`. `python probar_esp32.py --puertos` lista lo mismo.
- **Un solo programa a la vez en el puerto.** Si el puerto existe pero sale «Acceso denegado», «PermissionError» o «could not open port», otro programa lo tiene abierto: Thonny, el monitor serie de Arduino, las extensiones de VS Code **MicroPico** o **Pymakr**, otro `probar_esp32.py`, `verificar_ortesis.py` u `orquestador.py` en otra terminal, o cualquier script de Python que haya quedado abierto. Ciérralo y vuelve a intentar.
- **Sigue sin aparecer:** prueba otro puerto USB de la laptop (sin concentrador) y, si nada funciona, mantén apretado **BOOT** mientras conectas el cable.

**Si la tira no prende:** la flechita impresa en la tira debe apuntar alejándose del cable de datos (el DIN entra por el primer LED), el GND de la tira debe estar unido al GND de la ESP32 y la tira necesita su +5V.

## 2. Prueba rápida sin laptop (botón BOOT)

Con todo encendido, deja apretado el botón **BOOT** de la placa 1 segundo: la barra de luz sube y baja, zumban los vibradores y los dos servos van al 50 % y regresan. Dura 6 segundos. Si llega una orden de la laptop, la prueba se cancela. El paro de emergencia la detiene como siempre.

## 3. Probar desde la laptop

**Por cable USB** (recomendado, menos latencia; necesita `pip install pyserial`):

1. Cierra Thonny (solo un programa puede usar el puerto a la vez).
2. `python probar_esp32.py --puertos` muestra los puertos; la ESP32 suele salir como «CP210x» o «CH340».
3. `python probar_esp32.py --usb COM5` (en Mac o Linux, algo como `/dev/ttyUSB0`).

**Por Wi-Fi:** conecta la laptop a la red **Adaptrode** (clave `adaptrode2026`) y corre `python probar_esp32.py`.

| Orden | Qué hace |
| --- | --- |
| `c 0.5` | Cierra los dedos a la mitad (0 = abierta, 1 = cerrada) |
| `p 0.3` | Mueve el pulgar al 30 % |
| `a` / `x` | Abre todo / cierra todo |
| `v` / `e` | Vibra 200 ms / destello rojo |
| `t` | Muestra la telemetría (posición, corriente, paro, ángulos) |
| `lat` | Latencia: tiempo de ida y vuelta de los ACK |
| `ciclo` | Abre y cierra 3 veces, despacio |
| `cal dedos 20 140` | Cambia los ángulos abierta y cerrada y **los guarda en la ESP32** |
| `q` | Sale; la órtesis se abre sola en medio segundo |

## 4. Corriente con el cable USB conectado

Cuando la laptop está conectada por USB, **quiten el cable de VIN** de la ESP32: la placa se alimenta del USB y el cubo de 5 V sigue alimentando servos, tira y vibradores. El GND de la ESP32 se queda unido al GND del circuito. Así no hay dos fuentes peleando dentro de la placa. Durante la sesión, la laptop va con batería.

## 5. Calibrar

Con la órtesis puesta y la mano relajada, prueba `x` (cerrar) y `a` (abrir):

- si cierra de más, baja el segundo número: `cal dedos 20 130`;
- si no cierra lo suficiente, súbelo de 10 en 10;
- si no termina de abrir, baja el primero.

Haz lo mismo con `cal pulgar …`. Los ángulos quedan en `calibracion.json` dentro de la ESP32 y se cargan solos al encender. Para volver a los de fábrica, borra ese archivo desde Thonny.

## 6. Pines (DevKit V1)

| Qué | Pin |
| --- | --- |
| Servo de los dedos (señal naranja) | GPIO18 |
| Servo del pulgar (señal naranja) | GPIO19 |
| Tira LED (DIN, con resistencia de 330 Ω) | GPIO23 |
| Vibradores (IN) | GPIO25 y GPIO26 |
| Lectura del paro (divisor 10 k / 10 k) | GPIO32 |
| Corriente de los servos (resistencia de 0.1 Ω) | GPIO35 |
| Sensor de fuerza (opcional, entre 3V3 y GPIO34, con 10 kΩ a GND) | GPIO34 |
| Botón de prueba (el BOOT de la placa) | GPIO0 |
| Alimentación | VIN a +5V y GND a GND |

**Sensor de fuerza FSR402 (circular).** Un pin del FSR a 3V3, el otro a GPIO34 y de ahí 10 kΩ a GND. La telemetría manda `fuerza` (V/3.3) y `fuerza.py` la convierte a resistencia y newtons con la curva de la hoja de datos (`config.FSR_*`). `python firmware_esp32/probar_esp32.py` la imprime junto al resto de la telemetría y `python fuerza.py --puerto 192.168.4.1` la imprime sola, en vivo. Es **orientativa** (hasta ~±30 % por pieza): para newtons de verdad, calibrar con pesas (`fuerza.calibrar`). No probado con el sensor real.

En la placa los pines pueden decir D18, IO18 o G18: son lo mismo.

## 7. Qué significa la luz

- **Ámbar:** encendida, esperando órdenes.
- **Barra azul a verde agua:** recibe órdenes; sube cuando el decodificador quiere cerrar.
- **Destello rojo:** la laptop detectó un potencial de error (ErrP).
- **Rojo fijo:** paro de emergencia oprimido.

## 8. Protecciones

- **Sin órdenes por medio segundo:** abre la mano despacio.
- **Corriente arriba de 1.8 A por más de 0.3 s:** retrocede un poco y limita el cierre 2 s. Si la fuente se apaga o la ESP32 se reinicia al cerrar, bajen `i_bloqueo_ma` a 1500.
- **Con el paro oprimido:** los servos se quedan sin corriente y las metas vuelven a «abierta», así que al soltarlo la mano no salta a cerrarse. La ESP32 sigue prendida para avisar.
- **Si el programa se detiene** (un error o Stop en Thonny): apaga la señal de los servos y los vibradores.

## 9. Protocolo, ACK y latido (para el orquestador)

Las órdenes son el mismo JSON por los dos caminos: **una línea** por el cable USB (115200 baudios) o **un paquete UDP** al puerto 8888 por Wi-Fi. La ESP32 contesta por el mismo camino.

- **Orden:** `{"seq": 12, "cierre": 0.35, "pulgar": 0.2, "p": 0.71}`. `cierre` y `pulgar` van de 0 (abierta) a 1 (cerrada); `p` sube la barra del nervio de luz.
- **Destello de ErrP:** agrega `"errp": 1`. **Vibración:** `"vib": 120` (ms).
- **Calibrar:** `{"cmd": "calibrar", "servo": "dedos", "abierta": 20, "cerrada": 150}`.
- **ACK inmediato** por cada `seq` nuevo: `{"ack": 12, "t_ms": 532118}`, con la hora de la ESP32 al aplicar la orden.
- **Telemetría** cada 100 ms por Wi-Fi y cada 200 ms por USB: `cierre`, `pulgar`, `i_ma`, `fuerza`, `paro`, `bloqueo`, `vigilancia`, `msg`, `t_ms`. Por USB, `ang` y `ver` van una vez por segundo.
- **Latido obligatorio:** si la ESP32 pasa 0.5 s sin recibir nada, abre la mano. El cliente debe reenviar el estado cada 100 ms.

`OrtesisUDP` y `OrtesisUSB` ya hacen todo eso:

```python
from ortesis_usb import OrtesisUSB             # o: from ortesis_udp import OrtesisUDP; o = OrtesisUDP('192.168.4.1')
o = OrtesisUSB('COM5')                         # arranca el latido de 100 ms
o.on_paro(lambda oprimido: print('PAUSA_SEGURA' if oprimido else 'paro suelto'))
e = o.mover(cierre=0.6, p=0.72)                # e['t_aplicado']: hora del movimiento en time.monotonic()
print(e['metodo'], e['rtt_ms'])                # 'ack' o 'respaldo' (sin ACK: hora de envío + latencia medida)
o.set_p(0.4)                                   # nueva probabilidad del decoder, viaja en el siguiente latido
o.errp()                                       # destello rojo cuando el detector marca un ErrP
tel = o.telemetria()                           # incluye 't_local' (hora de la muestra en el reloj de la laptop)
print(o.latencia())                            # RTT mínimo, mediana y p95 de los últimos ~30 s
o.cerrar()
```

Cada latido trae su ACK; con esas muestras el cliente calcula el desfase entre el reloj de la ESP32 y el de la laptop y sigue su deriva.

**Con el orquestador por cable:** `python orquestador.py real --ortesis-usb COM5` (en lugar de `--puerto COM5`, que es el protocolo viejo de texto `M`/`A`/`T` y con este firmware da NO GO en el CP1 por ACK perdidos). Verificación previa: `python verificar_ortesis.py --ortesis-usb COM5`. Detalles en el README, sección *Protocolo del ESP32*.

## 10. Cómo bajar la latencia

1. **Usen el cable USB.** Es lo más parejo: unos 10 a 20 ms de ida y vuelta (la mayor parte es el tiempo de mandar los caracteres a 115200 baudios) y casi sin variación. El Wi-Fi puede ser más rápido en promedio, pero tiene picos de decenas de milisegundos. Además, con el cable la laptop conserva su Wi-Fi normal con internet, así que la IA no cae a plantillas.
2. **Si van por Wi-Fi:** la laptop a 1 o 2 m de la órtesis, sin nada más conectado a la red «Adaptrode», y el plan de energía de la laptop en «máximo rendimiento» (el ahorro de energía del Wi-Fi agrega retrasos). Si hay muchas redes alrededor, cambien `'canal'` a 1 u 11.
3. **Bluetooth no conviene:** en el ESP32, MicroPython solo maneja Bluetooth de baja energía, que tiene más retraso y más variación que el Wi-Fi o el USB, sobre todo en Windows.
4. Midan con la orden `lat`. Lo que importa para alinear el ErrP es que el retraso sea parejo; el ACK con la hora de la ESP32 corrige el resto.

## 11. Cambios en la 1.3

- Órdenes también por **cable USB** (líneas JSON), al mismo tiempo que por Wi-Fi; cada camino contesta por sí mismo.
- Ciclo a **100 Hz** (antes 50 Hz): una orden espera 10 ms como máximo antes de aplicarse.
- **Canal de Wi-Fi** configurable (`'canal'`).
- **Prueba rápida sin laptop** con el botón BOOT.
- Mismos pines, mismas protecciones, mismo formato de órdenes y el mismo ACK de la 1.2.

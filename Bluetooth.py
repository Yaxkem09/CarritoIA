import time

import serial

from Config import (
    BAUDRATE,
    COMANDO_DETENER,
    PUERTO_BLUETOOTH,
    TIMEOUT_BLUETOOTH,
)


class BluetoothCarrito:
    """Maneja la comunicacion Bluetooth entre Python y Arduino."""

    def __init__(
        self,
        puerto=None,
        baudrate=None,
        timeout=None,
        intervalo_minimo_envio=0.25,
    ):
        self.puerto = puerto if puerto is not None else PUERTO_BLUETOOTH
        self.baudrate = baudrate if baudrate is not None else BAUDRATE
        self.timeout = timeout if timeout is not None else TIMEOUT_BLUETOOTH
        self.conexion = None

        # Tiempo minimo entre envios al Arduino.
        # 0.25 segundos = maximo 4 comandos por segundo.
        self.intervalo_minimo_envio = intervalo_minimo_envio

        # Guarda informacion del ultimo comando enviado.
        self.ultimo_comando_enviado = None
        self.tiempo_ultimo_envio = 0.0

        # Comandos permitidos para navegacion base.
        self.comandos_validos = {
            "I",  # avanzar
            "J",  # curva suave izquierda
            "L",  # curva suave derecha
            "U",  # giro fuerte izquierda / giro 90 izquierda
            "O",  # giro fuerte derecha / giro 90 derecha
            "K",  # detener
        }

    def conectar(self):
        """Abre la conexion serial con el modulo Bluetooth."""
        try:
            self.conexion = serial.Serial(
                port=self.puerto,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )

            # Espera a que la conexion serial quede estable.
            time.sleep(2)

            print(f"Bluetooth conectado en {self.puerto} a {self.baudrate} baudios.")
            return True

        except serial.SerialException as error:
            self.conexion = None
            print(f"Error al conectar Bluetooth en {self.puerto}: {error}")
            return False

    def enviar(self, comando, forzar=False):
        """
        Envia un comando al Arduino.

        Por defecto, evita mandar demasiados comandos seguidos.
        Solo envia si:
        - el comando cambio,
        - ya paso el intervalo minimo de envio,
        - o se usa forzar=True.

        Esto evita saturar al Arduino cuando el modelo predice
        muchas veces seguidas la misma accion.
        """
        if not self.esta_conectado():
            print("No hay conexion Bluetooth activa. No se envio el comando.")
            return False

        comando_normalizado = self._normalizar_comando(comando)

        if comando_normalizado is None:
            print("Comando invalido o vacio. No se envio nada.")
            return False

        if not self._debe_enviar(comando_normalizado, forzar):
            return False

        texto = comando_normalizado + "\n"

        try:
            self.conexion.write(texto.encode("utf-8"))
            self.conexion.flush()

            self.ultimo_comando_enviado = comando_normalizado
            self.tiempo_ultimo_envio = time.monotonic()

            print(f"Comando enviado: {comando_normalizado}")
            return True

        except serial.SerialException as error:
            print(f"Error al enviar comando por Bluetooth: {error}")
            return False

    def detener(self):
        """Detiene el carrito inmediatamente."""
        return self.enviar(COMANDO_DETENER, forzar=True)

    def cerrar(self):
        """Detiene el carrito y cierra la conexion serial."""
        if self.esta_conectado():
            try:
                self.detener()
                time.sleep(0.1)
            finally:
                self.conexion.close()
                print("Conexion Bluetooth cerrada.")

        self.conexion = None

    def esta_conectado(self):
        """Devuelve True si el puerto serial esta abierto."""
        return self.conexion is not None and self.conexion.is_open

    def _normalizar_comando(self, comando):
        """
        Convierte el comando recibido a una letra valida.

        Acepta comandos como:
        - "i"
        - "I"
        - "I\n"
        - "I "
        """
        if comando is None:
            return None

        texto = str(comando).strip().upper()

        if not texto:
            return None

        # Solo tomamos el primer caracter.
        # Esto evita mandar cadenas largas por accidente.
        comando_limpio = texto[0]

        if comando_limpio not in self.comandos_validos:
            return None

        return comando_limpio

    def _debe_enviar(self, comando, forzar=False):
        """
        Decide si se debe mandar el comando al Arduino.

        Esto reduce comandos repetidos excesivos.
        """
        if forzar:
            return True

        ahora = time.monotonic()
        tiempo_desde_ultimo_envio = ahora - self.tiempo_ultimo_envio

        # Si el comando cambio, se envia inmediatamente.
        if comando != self.ultimo_comando_enviado:
            return True

        # Si es el mismo comando, solo se vuelve a enviar
        # cuando ya paso el intervalo minimo.
        if tiempo_desde_ultimo_envio >= self.intervalo_minimo_envio:
            return True

        return False


def probar_bluetooth():
    """Permite probar manualmente la comunicacion Bluetooth desde consola."""
    bluetooth = BluetoothCarrito()

    if not bluetooth.conectar():
        return

    print("\nComandos disponibles:")
    print("I = avanzar")
    print("J = curva suave izquierda")
    print("L = curva suave derecha")
    print("U = giro fuerte izquierda / giro 90 izquierda")
    print("O = giro fuerte derecha / giro 90 derecha")
    print("K = detener")
    print("q = salir")
    print("\nNota:")
    print("En modo final, la laptop debe mandar comandos cortos continuamente.")
    print("El Arduino se detendra solo si deja de recibir comandos recientes.")

    try:
        while True:
            comando = input("\nEscriba comando: ").strip()

            if comando.lower() == "q":
                bluetooth.detener()
                break

            enviado = bluetooth.enviar(comando, forzar=True)

            if not enviado:
                print("No se pudo enviar el comando.")

    except KeyboardInterrupt:
        print("\nPrueba interrumpida por el usuario.")

    finally:
        bluetooth.cerrar()


if __name__ == "__main__":
    probar_bluetooth()
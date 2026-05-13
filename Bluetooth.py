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

    def __init__(self, puerto=None, baudrate=None, timeout=None):
        self.puerto = puerto if puerto is not None else PUERTO_BLUETOOTH
        self.baudrate = baudrate if baudrate is not None else BAUDRATE
        self.timeout = timeout if timeout is not None else TIMEOUT_BLUETOOTH
        self.conexion = None

    def conectar(self):
        """Abre la conexion serial con el modulo Bluetooth."""
        try:
            self.conexion = serial.Serial(
                port=self.puerto,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            time.sleep(2)
            print(f"Bluetooth conectado en {self.puerto} a {self.baudrate} baudios.")
            return True
        except serial.SerialException as error:
            self.conexion = None
            print(f"Error al conectar Bluetooth en {self.puerto}: {error}")
            return False

    def enviar(self, comando):
        """Envia un comando de texto al Arduino."""
        if not self.esta_conectado():
            print("No hay conexion Bluetooth activa. No se envio el comando.")
            return False

        if comando is None:
            print("Comando vacio. No se envio nada.")
            return False

        texto = str(comando).strip()
        if not texto:
            print("Comando vacio. No se envio nada.")
            return False

        if not texto.endswith("\n"):
            texto += "\n"

        try:
            self.conexion.write(texto.encode("utf-8"))
            self.conexion.flush()
            print(f"Comando enviado: {texto.strip()}")
            return True
        except serial.SerialException as error:
            print(f"Error al enviar comando por Bluetooth: {error}")
            return False

    def cerrar(self):
        """Detiene el carrito y cierra la conexion serial."""
        if self.esta_conectado():
            try:
                self.enviar(COMANDO_DETENER)
            finally:
                self.conexion.close()
                print("Conexion Bluetooth cerrada.")
        self.conexion = None

    def esta_conectado(self):
        """Devuelve True si el puerto serial esta abierto."""
        return self.conexion is not None and self.conexion.is_open


def probar_bluetooth():
    """Permite probar manualmente la comunicacion Bluetooth desde consola."""
    bluetooth = BluetoothCarrito()

    if not bluetooth.conectar():
        return

    print("\nComandos disponibles:")
    print("I  = avanzar")
    print("J  = izquierda suave")
    print("L  = derecha suave")
    print("U  = giro fuerte izquierda")
    print("O  = giro fuerte derecha")
    print("K  = detener")
    print("1 = velocidad baja")
    print("2 = velocidad normal")
    print("q  = salir")

    try:
        while True:
            comando = input("\nEscriba comando: ").strip()

            if comando.lower() == "q":
                bluetooth.enviar(COMANDO_DETENER)
                break

            bluetooth.enviar(comando)
    except KeyboardInterrupt:
        print("\nPrueba interrumpida por el usuario.")
    finally:
        bluetooth.cerrar()


if __name__ == "__main__":
    probar_bluetooth()

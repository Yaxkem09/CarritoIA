#include <SoftwareSerial.h>

// Bluetooth HC-05
// D2 = RX Arduino, conectado al TXD del Bluetooth
// D3 = TX Arduino, conectado al RXD del Bluetooth
SoftwareSerial bluetooth(2, 3);

// Pines L298N
const int ENA = 5;   // PWM motor izquierdo
const int ENB = 6;   // PWM motor derecho

const int IN1 = 7;
const int IN2 = 8;
const int IN3 = 9;
const int IN4 = 10;

// ----------------------------------------------------
// CONFIGURACIÓN PARA NAVEGACIÓN BASE
// ----------------------------------------------------

// Si el Arduino no recibe ningún comando durante este tiempo,
// detiene el carrito por seguridad.
// Recomendado probar entre 400 y 700 ms.
const unsigned long TIEMPO_SIN_COMANDO = 500;

// Velocidades generales
const int VELOCIDAD_BAJA = 130;
const int VELOCIDAD_NORMAL = 210;

// Velocidad del motor lento en curvas suaves.
// Si la curva gira muy poco, baja este valor.
// Si la curva gira demasiado fuerte, sube este valor.
const int VELOCIDAD_CURVA_LENTA = 60;

// Velocidad actual
int velocidadActual = VELOCIDAD_NORMAL;

// Último momento en que llegó un comando válido
unsigned long ultimoComando = 0;

// Último comando de movimiento recibido
char comandoActual = 'K';

// Control para no repetir mensajes de timeout muchas veces
bool detenidoPorTimeout = true;

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  detener();

  Serial.begin(9600);
  bluetooth.begin(9600);

  Serial.println("Carrito listo por Bluetooth.");
  Serial.println("Modo: Navegacion Base sin Bonus.");
  Serial.println("Comandos disponibles:");
  Serial.println("I = avanzar");
  Serial.println("J = curva suave izquierda");
  Serial.println("L = curva suave derecha");
  Serial.println("U = giro fuerte izquierda / giro 90 izquierda");
  Serial.println("O = giro fuerte derecha / giro 90 derecha");
  Serial.println("K = detener");
  Serial.println("1 = velocidad baja");
  Serial.println("2 = velocidad normal");
}

void loop() {
  // Leer todos los comandos disponibles por Bluetooth
  while (bluetooth.available()) {
    char comando = bluetooth.read();

    // Ignorar saltos de línea o ENTER
    if (comando == '\n' || comando == '\r') {
      continue;
    }

    Serial.print("Comando recibido: ");
    Serial.println(comando);

    procesarComando(comando);
  }

  // Seguridad: si la laptop deja de mandar comandos, detener el carrito
  if (millis() - ultimoComando > TIEMPO_SIN_COMANDO) {
    if (!detenidoPorTimeout) {
      detener();
      comandoActual = 'K';
      detenidoPorTimeout = true;
      Serial.println("Sin comandos recientes. Carrito detenido por seguridad.");
    }
  }
}

void procesarComando(char c) {
  // Convertir minúsculas a mayúsculas
  if (c >= 'a' && c <= 'z') {
    c = c - 32;
  }

  switch (c) {
    case 'I':
      comandoActual = 'I';
      ultimoComando = millis();
      detenidoPorTimeout = false;
      avanzar();
      break;

    case 'J':
      comandoActual = 'J';
      ultimoComando = millis();
      detenidoPorTimeout = false;
      izquierdaSuave();
      break;

    case 'L':
      comandoActual = 'L';
      ultimoComando = millis();
      detenidoPorTimeout = false;
      derechaSuave();
      break;

    case 'U':
      comandoActual = 'U';
      ultimoComando = millis();
      detenidoPorTimeout = false;
      giroFuerteIzquierda();
      break;

    case 'O':
      comandoActual = 'O';
      ultimoComando = millis();
      detenidoPorTimeout = false;
      giroFuerteDerecha();
      break;

    case 'K':
      comandoActual = 'K';
      ultimoComando = millis();
      detenidoPorTimeout = true;
      detener();
      break;

    case '1':
      velocidadActual = VELOCIDAD_BAJA;
      ultimoComando = millis();
      Serial.println("Velocidad baja activada.");
      ejecutarUltimoMovimiento();
      break;

    case '2':
      velocidadActual = VELOCIDAD_NORMAL;
      ultimoComando = millis();
      Serial.println("Velocidad normal activada.");
      ejecutarUltimoMovimiento();
      break;

    default:
      Serial.println("Comando no reconocido. Se detiene por seguridad.");
      comandoActual = 'K';
      ultimoComando = millis();
      detenidoPorTimeout = true;
      detener();
      break;
  }
}

void ejecutarUltimoMovimiento() {
  switch (comandoActual) {
    case 'I':
      detenidoPorTimeout = false;
      avanzar();
      break;

    case 'J':
      detenidoPorTimeout = false;
      izquierdaSuave();
      break;

    case 'L':
      detenidoPorTimeout = false;
      derechaSuave();
      break;

    case 'U':
      detenidoPorTimeout = false;
      giroFuerteIzquierda();
      break;

    case 'O':
      detenidoPorTimeout = false;
      giroFuerteDerecha();
      break;

    case 'K':
    default:
      detenidoPorTimeout = true;
      detener();
      break;
  }
}

void aplicarVelocidad(int velocidadIzquierda, int velocidadDerecha) {
  analogWrite(ENA, velocidadIzquierda);
  analogWrite(ENB, velocidadDerecha);
}

void avanzar() {
  aplicarVelocidad(velocidadActual, velocidadActual);

  // Motor izquierdo hacia adelante
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Motor derecho hacia adelante
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void izquierdaSuave() {
  // Curva suave izquierda:
  // motor izquierdo más lento, motor derecho normal
  aplicarVelocidad(VELOCIDAD_CURVA_LENTA, velocidadActual);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void derechaSuave() {
  // Curva suave derecha:
  // motor izquierdo normal, motor derecho más lento
  aplicarVelocidad(velocidadActual, VELOCIDAD_CURVA_LENTA);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void giroFuerteIzquierda() {
  // Giro fuerte / giro de 90 grados hacia la izquierda
  aplicarVelocidad(velocidadActual, velocidadActual);

  // Motor izquierdo hacia atrás
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);

  // Motor derecho hacia adelante
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void giroFuerteDerecha() {
  // Giro fuerte / giro de 90 grados hacia la derecha
  aplicarVelocidad(velocidadActual, velocidadActual);

  // Motor izquierdo hacia adelante
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Motor derecho hacia atrás
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void detener() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
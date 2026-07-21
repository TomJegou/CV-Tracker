/*
 * Leonardo — réception de vecteurs de mouvement via USB Serial.
 *
 * Protocole : "<X,Y>\n"  (ex. "<12,-34>\n")
 * Baud rate : 115200
 *
 * Les deltas parsés sont stockés dans moveX / moveY, avec le flag movePending
 * prêt pour Mouse.move() une fois le USB Host Shield branché.
 */

#define SERIAL_BAUD 115200

// Deltas prêts à consommer (ex. Mouse.move(moveX, moveY, 0))
volatile int16_t moveX = 0;
volatile int16_t moveY = 0;
volatile bool movePending = false;

// Debug : renvoyer sur Serial ce qui a été parsé (désactiver en prod)
#define DEBUG_ECHO true

enum ParseState : uint8_t {
  WAIT_START,
  READ_X,
  READ_Y,
};

ParseState state = WAIT_START;

char xBuf[8];
char yBuf[8];
uint8_t xIdx = 0;
uint8_t yIdx = 0;

void resetParser() {
  state = WAIT_START;
  xIdx = 0;
  yIdx = 0;
  xBuf[0] = '\0';
  yBuf[0] = '\0';
}

bool commitMove() {
  long parsedX = atol(xBuf);
  long parsedY = atol(yBuf);

  if (parsedX < -32768 || parsedX > 32767 || parsedY < -32768 || parsedY > 32767) {
    return false;
  }

  moveX = (int16_t)parsedX;
  moveY = (int16_t)parsedY;
  movePending = true;
  return true;
}

void processChar(char c) {
  switch (state) {
    case WAIT_START:
      if (c == '<') {
        xIdx = 0;
        yIdx = 0;
        xBuf[0] = '\0';
        yBuf[0] = '\0';
        state = READ_X;
      }
      break;

    case READ_X:
      if (c == ',') {
        xBuf[xIdx] = '\0';
        state = READ_Y;
      } else if (c == '>' || c == '\n' || c == '\r') {
        resetParser();
      } else if (xIdx < sizeof(xBuf) - 1) {
        xBuf[xIdx++] = c;
      } else {
        resetParser();
      }
      break;

    case READ_Y:
      if (c == '>') {
        yBuf[yIdx] = '\0';
        if (commitMove()) {
#if DEBUG_ECHO
          Serial.print(F("OK <"));
          Serial.print(moveX);
          Serial.print(F(","));
          Serial.print(moveY);
          Serial.println(F(">"));
#endif
        }
        resetParser();
      } else if (c == '\n' || c == '\r') {
        resetParser();
      } else if (yIdx < sizeof(yBuf) - 1) {
        yBuf[yIdx++] = c;
      } else {
        resetParser();
      }
      break;
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) {
    ;  // Attendre l'ouverture du port USB CDC (Leonardo)
  }
  resetParser();

  Serial.println(F("serial_mouse ready — format <X,Y> @ 115200"));
}

void loop() {
  while (Serial.available() > 0) {
    processChar((char)Serial.read());
  }

  if (movePending) {
    movePending = false;

    // TODO (USB Host Shield) : injecter le mouvement sur le bus HID
    // Mouse.move(moveX, moveY, 0);
  }
}

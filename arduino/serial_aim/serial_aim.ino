/*
 * Leonardo — aim AI only (2e souris HID)
 *
 * Setup "deux souris" :
 *   - Souris gamer → port USB du PC (contrôle + G HUB)
 *   - Leonardo USB → PC (HID Mouse + Serial CDC) — deltas aim uniquement
 *   - Host Shield NON requis
 *
 * Protocole (identique à mouse_fusion / core/mouse.py) :
 *   <dx,dy>\n   @ 115200   ex. <12,-34>\n
 *
 * Flash : carte "Arduino Leonardo", ce sketch.
 */

#include <Mouse.h>

#define SERIAL_BAUD 115200
#define DEBUG_ECHO false

#define HID_MOVE_CHUNK 127
#define ACC_LIMIT 4096
#define SERIAL_READ_BUDGET 64
#define FLUSH_STEPS 8

static int16_t accX = 0;
static int16_t accY = 0;

static int8_t clampHidStep(int16_t value) {
  if (value > HID_MOVE_CHUNK) {
    return HID_MOVE_CHUNK;
  }
  if (value < -HID_MOVE_CHUNK) {
    return (int8_t)(-HID_MOVE_CHUNK);
  }
  return (int8_t)value;
}

static void accumulateMove(int16_t dx, int16_t dy) {
  int32_t nx = (int32_t)accX + dx;
  int32_t ny = (int32_t)accY + dy;
  if (nx > ACC_LIMIT) {
    nx = ACC_LIMIT;
  } else if (nx < -ACC_LIMIT) {
    nx = -ACC_LIMIT;
  }
  if (ny > ACC_LIMIT) {
    ny = ACC_LIMIT;
  } else if (ny < -ACC_LIMIT) {
    ny = -ACC_LIMIT;
  }
  accX = (int16_t)nx;
  accY = (int16_t)ny;
}

static void flushMoves(uint8_t maxSteps) {
  while (maxSteps-- > 0 && (accX != 0 || accY != 0)) {
    int8_t stepX = clampHidStep(accX);
    int8_t stepY = clampHidStep(accY);
    Mouse.move(stepX, stepY, 0);
    accX -= stepX;
    accY -= stepY;
  }
}

// ---------------------------------------------------------------------------
// Parser Serial — format <X,Y>
// ---------------------------------------------------------------------------
enum ParseState : uint8_t {
  WAIT_START,
  READ_X,
  READ_Y,
};

static ParseState parseState = WAIT_START;
static char xBuf[8];
static char yBuf[8];
static uint8_t xIdx = 0;
static uint8_t yIdx = 0;

static void resetParser() {
  parseState = WAIT_START;
  xIdx = 0;
  yIdx = 0;
  xBuf[0] = '\0';
  yBuf[0] = '\0';
}

static bool isSignedIntToken(const char *buf) {
  if (buf[0] == '\0') {
    return false;
  }
  uint8_t i = 0;
  if (buf[0] == '-' || buf[0] == '+') {
    i = 1;
    if (buf[i] == '\0') {
      return false;
    }
  }
  for (; buf[i] != '\0'; i++) {
    if (buf[i] < '0' || buf[i] > '9') {
      return false;
    }
  }
  return true;
}

static bool commitAiMove() {
  if (!isSignedIntToken(xBuf) || !isSignedIntToken(yBuf)) {
    return false;
  }

  long parsedX = atol(xBuf);
  long parsedY = atol(yBuf);

  if (parsedX < -32768L || parsedX > 32767L ||
      parsedY < -32768L || parsedY > 32767L) {
    return false;
  }

  accumulateMove((int16_t)parsedX, (int16_t)parsedY);

#if DEBUG_ECHO
  Serial.print(F("AI <"));
  Serial.print((int16_t)parsedX);
  Serial.print(F(","));
  Serial.print((int16_t)parsedY);
  Serial.println(F(">"));
#endif
  return true;
}

static void processSerialChar(char c) {
  switch (parseState) {
    case WAIT_START:
      if (c == '<') {
        xIdx = 0;
        yIdx = 0;
        xBuf[0] = '\0';
        yBuf[0] = '\0';
        parseState = READ_X;
      }
      break;

    case READ_X:
      if (c == ',') {
        xBuf[xIdx] = '\0';
        parseState = READ_Y;
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
        commitAiMove();
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

static void pollSerial() {
  uint8_t budget = SERIAL_READ_BUDGET;
  while (budget-- > 0 && Serial.available() > 0) {
    processSerialChar((char)Serial.read());
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 2000) {
    ;
  }

  Mouse.begin();
  resetParser();
}

void loop() {
  pollSerial();
  flushMoves(FLUSH_STEPS);
}

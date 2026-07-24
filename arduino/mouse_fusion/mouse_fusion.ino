/*
 * Leonardo + USB Host Shield 2.0
 * Fusion souris physique (HID Host) + vecteurs IA (Serial <X,Y> @ 115200).
 *
 * Branchement :
 *   - Leonardo USB native → PC (HID Mouse + Serial CDC)
 *   - Souris / dongle → port USB du Host Shield
 *
 * Logitech G Pro X Superlight 2 (LIGHTSPEED) :
 *   Rapport 13 octets : [buttons16][dx16][dy16][wheel8][pan8][vendor×5]
 *   Préférer le filaire pour l'aim assist (lag LIGHTSPEED via Host Shield).
 */

#include <hidboot.h>
#include <usbhub.h>
#include <SPI.h>
#include <Mouse.h>

#define SERIAL_BAUD 115200
#define DEBUG_ECHO false
#define DEBUG_MOUSE_REPORT false

/*
 * Layout HID physique :
 *   0 = AUTO (len>=13 → Superlight LS, len>=5 → XY16, sinon Boot8)
 *   1 = BOOT8
 *   2 = XY16
 *   3 = SKIP_ID_BOOT8
 *   4 = SKIP_ID_XY16
 *   5 = SUPERLIGHT_LS
 */
#define PHYSICAL_REPORT_LAYOUT 0

#define HID_MOVE_CHUNK 127
#define ACC_LIMIT 4096  // évite overflow / téléport si Usb.Task prend du retard

// Accumulateurs unifiés (physique + IA) — vidés uniquement dans loop()
static int16_t accX = 0;
static int16_t accY = 0;
static int8_t accWheel = 0;

static int8_t clampHidStep(int16_t value) {
  if (value > HID_MOVE_CHUNK) {
    return HID_MOVE_CHUNK;
  }
  if (value < -HID_MOVE_CHUNK) {
    return -HID_MOVE_CHUNK;
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

static void accumulateWheel(int8_t wheel) {
  int16_t nw = (int16_t)accWheel + wheel;
  if (nw > HID_MOVE_CHUNK) {
    nw = HID_MOVE_CHUNK;
  } else if (nw < -HID_MOVE_CHUNK) {
    nw = -HID_MOVE_CHUNK;
  }
  accWheel = (int8_t)nw;
}

// Envoie au plus maxSteps paquets HID puis revient (Usb.Task reste réactif)
static void flushMoves(uint8_t maxSteps) {
  while (maxSteps-- > 0 && (accX != 0 || accY != 0 || accWheel != 0)) {
    int8_t stepX = clampHidStep(accX);
    int8_t stepY = clampHidStep(accY);
    int8_t stepW = accWheel;
    Mouse.move(stepX, stepY, stepW);
    accX -= stepX;
    accY -= stepY;
    accWheel -= stepW;
  }
}

// ---------------------------------------------------------------------------
// Parser souris physique
// ---------------------------------------------------------------------------
class MouseRptParser : public MouseReportParser {
  uint8_t prevButtons;

public:
  MouseRptParser() : prevButtons(0) {}
  void Parse(USBHID *hid, bool is_rpt_id, uint8_t len, uint8_t *buf);

private:
  void dispatchButtons(uint8_t buttons);
  bool extractReport(
    uint8_t len,
    uint8_t *data,
    int16_t *dx,
    int16_t *dy,
    int8_t *wheel,
    uint8_t *buttons
  );
};

void MouseRptParser::dispatchButtons(uint8_t buttons) {
  uint8_t changed = buttons ^ prevButtons;
  if (changed & 0x01) {
    if (buttons & 0x01) {
      Mouse.press(MOUSE_LEFT);
    } else {
      Mouse.release(MOUSE_LEFT);
    }
  }
  if (changed & 0x02) {
    if (buttons & 0x02) {
      Mouse.press(MOUSE_RIGHT);
    } else {
      Mouse.release(MOUSE_RIGHT);
    }
  }
  if (changed & 0x04) {
    if (buttons & 0x04) {
      Mouse.press(MOUSE_MIDDLE);
    } else {
      Mouse.release(MOUSE_MIDDLE);
    }
  }
#ifdef MOUSE_BACK
  if (changed & 0x08) {
    if (buttons & 0x08) {
      Mouse.press(MOUSE_BACK);
    } else {
      Mouse.release(MOUSE_BACK);
    }
  }
#endif
#ifdef MOUSE_FORWARD
  if (changed & 0x10) {
    if (buttons & 0x10) {
      Mouse.press(MOUSE_FORWARD);
    } else {
      Mouse.release(MOUSE_FORWARD);
    }
  }
#endif
  prevButtons = buttons;
}

bool MouseRptParser::extractReport(
  uint8_t len,
  uint8_t *data,
  int16_t *dx,
  int16_t *dy,
  int8_t *wheel,
  uint8_t *buttons
) {
  if (data == nullptr || len < 3) {
    return false;
  }

  *wheel = 0;
  uint8_t layout = PHYSICAL_REPORT_LAYOUT;
  if (layout == 0) {
    if (len >= 13) {
      layout = 5;
    } else if (len >= 5) {
      layout = 2;
    } else {
      layout = 1;
    }
  }

  switch (layout) {
    case 5:  // Superlight LIGHTSPEED 13 bytes
      if (len < 7) {
        return false;
      }
      *buttons = data[0];
      *dx = (int16_t)(data[2] | ((uint16_t)data[3] << 8));
      *dy = (int16_t)(data[4] | ((uint16_t)data[5] << 8));
      *wheel = (int8_t)data[6];
      return true;

    case 2:
    case 4:
      if (len < 5) {
        return false;
      }
      *buttons = data[0];
      *dx = (int16_t)(data[1] | ((uint16_t)data[2] << 8));
      *dy = (int16_t)(data[3] | ((uint16_t)data[4] << 8));
      if (len >= 6) {
        *wheel = (int8_t)data[5];
      }
      return true;

    case 1:
    case 3:
    default:
      *buttons = data[0];
      *dx = (int8_t)data[1];
      *dy = (int8_t)data[2];
      if (len >= 4) {
        *wheel = (int8_t)data[3];
      }
      return true;
  }
}

void MouseRptParser::Parse(USBHID *hid, bool is_rpt_id, uint8_t len, uint8_t *buf) {
  (void)hid;
  if (buf == nullptr || len < 3) {
    return;
  }

#if DEBUG_MOUSE_REPORT
  {
    static uint32_t lastDump = 0;
    if (millis() - lastDump >= 80) {
      lastDump = millis();
      Serial.print(F("HID len="));
      Serial.print(len);
      Serial.print(F(" ["));
      for (uint8_t i = 0; i < len && i < 13; i++) {
        if (i) {
          Serial.print(' ');
        }
        if (buf[i] < 16) {
          Serial.print('0');
        }
        Serial.print(buf[i], HEX);
      }
      Serial.println(F("]"));
    }
  }
#endif

  uint8_t *data = buf;
  uint8_t dataLen = len;

  bool skipId = is_rpt_id ||
                (PHYSICAL_REPORT_LAYOUT == 3) ||
                (PHYSICAL_REPORT_LAYOUT == 4);
  if (skipId && dataLen > 3) {
    data++;
    dataLen--;
  }

  int16_t dx = 0;
  int16_t dy = 0;
  int8_t wheel = 0;
  uint8_t buttons = 0;
  if (!extractReport(dataLen, data, &dx, &dy, &wheel, &buttons)) {
    return;
  }

  dispatchButtons(buttons);

  if (dx != 0 || dy != 0) {
    accumulateMove(dx, dy);
  }
  if (wheel != 0) {
    accumulateWheel(wheel);
  }
}

// ---------------------------------------------------------------------------
// USB Host
// ---------------------------------------------------------------------------
USB Usb;
USBHub Hub(&Usb);
HIDBoot<USB_HID_PROTOCOL_MOUSE> HidMouse(&Usb, true);
MouseRptParser mouseParser;

// ---------------------------------------------------------------------------
// Machine à états Serial — format <X,Y>
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

  // Même chemin que le physique → pas de burst Mouse.move dans loop
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
  uint8_t budget = 64;
  while (budget-- > 0 && Serial.available() > 0) {
    processSerialChar((char)Serial.read());
  }
}

// ---------------------------------------------------------------------------
// Setup / Loop
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(SERIAL_BAUD);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 2000) {
    ;
  }

  Mouse.begin();

  if (Usb.Init() == -1) {
    Serial.println(F("ERROR: USB Host Shield init failed — halt"));
    for (;;) {
      ;
    }
  }

  delay(200);
  HidMouse.SetReportParser(0, &mouseParser);
  resetParser();
}

void loop() {
  for (uint8_t i = 0; i < 8; i++) {
    Usb.Task();
    flushMoves(2);
  }
  pollSerial();
  flushMoves(4);  // dégager d'éventuels vecteurs IA reçus ce tour
}

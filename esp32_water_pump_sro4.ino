#include "BluetoothSerial.h"

#define RELAY_PIN 12

BluetoothSerial SerialBT;
bool pumpState = false;

void btCallback(esp_spp_cb_event_t event, esp_spp_cb_param_t *param) {
    if (event == ESP_SPP_SRV_OPEN_EVT) {
        Serial.println("[BT] PC connected");
    } else if (event == ESP_SPP_CLOSE_EVT) {
        Serial.println("[BT] PC disconnected");
        // safety: turn pump off if connection drops
        digitalWrite(RELAY_PIN, LOW);
        pumpState = false;
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);   // NC relay: LOW = pump OFF at startup
    SerialBT.register_callback(btCallback);
    SerialBT.begin("FAW-Drone");
    Serial.println("[BT] Started. Device name: FAW-Drone");
    Serial.println("[BT] Waiting for PC to connect...");
}

void loop() {
    if (SerialBT.available()) {
        char cmd = (char)SerialBT.read();
        if (cmd == '1' && !pumpState) {
            digitalWrite(RELAY_PIN, HIGH);  // NC relay: HIGH = pump ON
            pumpState = true;
            Serial.println("[PUMP] ON  - armyworm detected");
            SerialBT.println("PUMP_ON");
        } else if (cmd == '0' && pumpState) {
            digitalWrite(RELAY_PIN, LOW);   // NC relay: LOW = pump OFF
            pumpState = false;
            Serial.println("[PUMP] OFF - no detection");
            SerialBT.println("PUMP_OFF");
        }
    }
    delay(10);
}
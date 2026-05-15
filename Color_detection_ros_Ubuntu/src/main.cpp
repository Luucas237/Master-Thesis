#include <Arduino.h>
#include <SPI.h>
#include <HighPowerStepperDriver.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

WiFiMulti wifiMulti;


const float KROKI_NA_OBROT = 200.0; 
const float MIKROKROK = 8.0;        // 1/8 kroku
const float PRZELOZENIE = 1.0;      
const int LIMIT_PRADU_MA = 1000;

const float PRZELOZENIE_PAN = 120.0 / 30.0;
const float PRZELOZENIE_TILT = 2.0;

long obliczKrokiPan(float stopnie) {
  return (stopnie / 360.0) * KROKI_NA_OBROT * MIKROKROK * PRZELOZENIE_PAN;
}

long obliczKrokiTilt(float stopnie) {
  return (stopnie / 360.0) * KROKI_NA_OBROT * MIKROKROK * PRZELOZENIE_TILT;
}

HighPowerStepperDriver sd1;
HighPowerStepperDriver sd2;


const uint8_t SCS1_PIN = 5;
const uint8_t STEP1_PIN = 16;
const uint8_t DIR1_PIN = 4;


const uint8_t SCS2_PIN = 17;
const uint8_t STEP2_PIN = 12;
const uint8_t DIR2_PIN = 14;


void wykonajRuch(long krokiPan, long krokiTilt) {

  digitalWrite(DIR1_PIN, (krokiPan > 0) ? HIGH : LOW);
  digitalWrite(DIR2_PIN, (krokiTilt > 0) ? HIGH : LOW);

  long kroki1_abs = abs(krokiPan);
  long kroki2_abs = abs(krokiTilt);
  long maxKroki = max(kroki1_abs, kroki2_abs);


  for (long i = 0; i < maxKroki; i++) {
    if (i < kroki1_abs) digitalWrite(STEP1_PIN, HIGH);
    if (i < kroki2_abs) digitalWrite(STEP2_PIN, HIGH);
    
    delayMicroseconds(5);
    
    digitalWrite(STEP1_PIN, LOW);
    digitalWrite(STEP2_PIN, LOW);
    
    delayMicroseconds(1000);
  }
}

long aktualnaPozycjaPan = 0;
long aktualnaPozycjaTilt = 0;

void setup() {
  Serial.begin(115200);
  SPI.begin(); 
  
  pinMode(STEP1_PIN, OUTPUT);
  pinMode(DIR1_PIN, OUTPUT);
  pinMode(STEP2_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);

  Serial.println("\nKonfiguracja Wi-Fi...");
  
  wifiMulti.addAP("2.4-Vectra-WiFi-B98217", "sf4it0mykh52pagp");
  // wifiMulti.addAP("iPhone", "misiek007");

  Serial.println("Lacze z najlepsza dostepna siecia...");

  int attempts = 0;
  while (wifiMulti.run() != WL_CONNECTED && attempts < 10) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }

  if(WiFi.status() == WL_CONNECTED) {
    Serial.println("\n--- POLACZONO Z WI-FI! ---");
    Serial.print("Siec: "); Serial.println(WiFi.SSID());
    Serial.print("Adres IP urządzenia: "); Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[OSTRZEZENIE] Nie udalo sie polaczyc z Wi-Fi. OTA nie zadziala!");
  }

  if(WiFi.status() == WL_CONNECTED) {
    ArduinoOTA.setHostname("Wiezyczka-Laserowa");
    ArduinoOTA.onStart([]() { Serial.println("\n[OTA] Rozpoczynam wgrywanie..."); });
    ArduinoOTA.onEnd([]() { Serial.println("\n[OTA] Wgrywanie zakonczone!"); });
    ArduinoOTA.onError([](ota_error_t error) { Serial.printf("[OTA] Blad [%u]\n", error); });
    ArduinoOTA.begin();
  }

  sd1.setChipSelectPin(SCS1_PIN);
  sd1.resetSettings();
  sd1.clearStatus();
  sd1.setCurrentMilliamps36v4(LIMIT_PRADU_MA); 
  sd1.setStepMode(HPSDStepMode::MicroStep8);
  sd1.enableDriver(); 

  sd2.setChipSelectPin(SCS2_PIN);
  sd2.resetSettings();
  sd2.clearStatus();
  sd2.setCurrentMilliamps36v4(LIMIT_PRADU_MA); 
  sd2.setStepMode(HPSDStepMode::MicroStep8); 
  sd2.enableDriver(); 

  Serial.println("==========================================");
  Serial.println("SYSTEM GOTOWY NA KOMENDY Z RASPBERRY PI!");
  Serial.println("Oczekuje na format: 'KAT_PAN KAT_TILT' (np: 90 -45)");
  Serial.println("==========================================");
}

void loop() {
  ArduinoOTA.handle();

  if (Serial.available() > 0) {
    String ostatniaKomenda = "";

    while (Serial.available() > 0) {
      String linia = Serial.readStringUntil('\n');
      linia.trim();
      if (linia.length() > 2) {
        ostatniaKomenda = linia;
      }
    }

    if (ostatniaKomenda.length() > 0) {
      int spaceIndex = ostatniaKomenda.indexOf(' ');
      
      if (spaceIndex > 0) {

        float docelowyKatPan = ostatniaKomenda.substring(0, spaceIndex).toFloat();
        float docelowyKatTilt = ostatniaKomenda.substring(spaceIndex + 1).toFloat();

        long doceloweKrokiPan = obliczKrokiPan(docelowyKatPan);
        long doceloweKrokiTilt = obliczKrokiTilt(docelowyKatTilt);

        long ruchPan = doceloweKrokiPan - aktualnaPozycjaPan;
        long ruchTilt = doceloweKrokiTilt - aktualnaPozycjaTilt;

        if (ruchPan != 0 || ruchTilt != 0) {
          wykonajRuch(ruchPan, ruchTilt);

          aktualnaPozycjaPan = doceloweKrokiPan;
          aktualnaPozycjaTilt = doceloweKrokiTilt;
        }
      }
    }
  }
}

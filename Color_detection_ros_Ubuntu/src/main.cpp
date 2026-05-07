#include <Arduino.h>
#include <SPI.h>
#include <HighPowerStepperDriver.h>
#include <WiFi.h>
#include <WiFiMulti.h> // Biblioteka do obsługi wielu sieci Wi-Fi
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

// ==========================================
// 1. KONFIGURACJA SIECI WI-FI (Multi-WiFi)
// ==========================================
WiFiMulti wifiMulti;

// ==========================================
// 2. KONFIGURACJA SILNIKÓW (Mechanika)
// ==========================================
const float KROKI_NA_OBROT = 200.0; 
const float MIKROKROK = 8.0;        // 1/8 kroku dla plynnej i silnej pracy
const float PRZELOZENIE = 1.0;      
const int LIMIT_PRADU_MA = 1000;

// --- NOWE PARAMETRY MECHANICZNE ---
const float PRZELOZENIE_PAN = 120.0 / 30.0; // = 4.0 (Silnik robi 4 obroty na 1 obrót wieży)
const float PRZELOZENIE_TILT = 2.0;         // = 2.0 (Silnik robi 2 obroty na 1 obrót wieży)

long obliczKrokiPan(float stopnie) {
  return (stopnie / 360.0) * KROKI_NA_OBROT * MIKROKROK * PRZELOZENIE_PAN;
}

long obliczKrokiTilt(float stopnie) {
  return (stopnie / 360.0) * KROKI_NA_OBROT * MIKROKROK * PRZELOZENIE_TILT;
}

HighPowerStepperDriver sd1; // PAN (Poziom)
HighPowerStepperDriver sd2; // TILT (Pion)

// Piny PAN
const uint8_t SCS1_PIN = 5;
const uint8_t STEP1_PIN = 16;
const uint8_t DIR1_PIN = 4;

// Piny TILT
const uint8_t SCS2_PIN = 17;
const uint8_t STEP2_PIN = 12;
const uint8_t DIR2_PIN = 14;

// ==========================================
// 3. FUNKCJE POMOCNICZE
// ==========================================
void wykonajRuch(long krokiPan, long krokiTilt) {
  // Ustawienie kierunków na podstawie znaków
  digitalWrite(DIR1_PIN, (krokiPan > 0) ? HIGH : LOW);
  digitalWrite(DIR2_PIN, (krokiTilt > 0) ? HIGH : LOW);

  long kroki1_abs = abs(krokiPan);
  long kroki2_abs = abs(krokiTilt);
  long maxKroki = max(kroki1_abs, kroki2_abs);

  // Zoptymalizowana pętla ruchu dla obu silników (bez delay, tylko delayMicroseconds)
  for (long i = 0; i < maxKroki; i++) {
    if (i < kroki1_abs) digitalWrite(STEP1_PIN, HIGH);
    if (i < kroki2_abs) digitalWrite(STEP2_PIN, HIGH);
    
    delayMicroseconds(5); // Krótki impuls
    
    digitalWrite(STEP1_PIN, LOW);
    digitalWrite(STEP2_PIN, LOW);
    
    delayMicroseconds(1000); // Prędkość ruchu (zmieniaj by przyspieszyć/zwolnić)
  }
}

long aktualnaPozycjaPan = 0;
long aktualnaPozycjaTilt = 0;
// ==========================================
// 4. GŁÓWNY SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  SPI.begin(); 
  
  pinMode(STEP1_PIN, OUTPUT);
  pinMode(DIR1_PIN, OUTPUT);
  pinMode(STEP2_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);

  // --- PODŁĄCZENIE DO WI-FI (WIELE SIECI) ---
  Serial.println("\nKonfiguracja Wi-Fi...");
  
  // DODAJ SWOJE SIECI TUTAJ (Możesz dodać więcej niż dwie)
  wifiMulti.addAP("2.4-Vectra-WiFi-B98217", "sf4it0mykh52pagp");
  // wifiMulti.addAP("iPhone", "misiek007");

  Serial.println("Lacze z najlepsza dostepna siecia...");
  
  // Czekaj na połączenie (sprawdza dostępne sieci i łączy z najsilniejszą)
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
    // Nie robimy restartu, bo ESP32 musi nadal odbierac komendy po kablu!
  }

  // --- KONFIGURACJA OTA ---
  if(WiFi.status() == WL_CONNECTED) {
    ArduinoOTA.setHostname("Wiezyczka-Laserowa");
    ArduinoOTA.onStart([]() { Serial.println("\n[OTA] Rozpoczynam wgrywanie..."); });
    ArduinoOTA.onEnd([]() { Serial.println("\n[OTA] Wgrywanie zakonczone!"); });
    ArduinoOTA.onError([](ota_error_t error) { Serial.printf("[OTA] Blad [%u]\n", error); });
    ArduinoOTA.begin();
  }

  // --- INICJALIZACJA SILNIKÓW (Nasze sprawdzone ustawienia) ---
  sd1.setChipSelectPin(SCS1_PIN);
  sd1.resetSettings();
  sd1.clearStatus();
  sd1.setCurrentMilliamps36v4(LIMIT_PRADU_MA); 
  sd1.setStepMode(HPSDStepMode::MicroStep8); // Wracamy do 1/8 dla płynności
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

// ==========================================
// 5. GŁÓWNA PĘTLA
// ==========================================
void loop() {
  ArduinoOTA.handle();

  if (Serial.available() > 0) {
    String ostatniaKomenda = "";
    
    // 1. Zgarniamy wszystkie komendy z bufora, zachowujemy tylko OSTATNIĄ
    while (Serial.available() > 0) {
      String linia = Serial.readStringUntil('\n');
      linia.trim(); // Usuwa białe znaki
      if (linia.length() > 2) {
        ostatniaKomenda = linia; // Nadpisujemy, dopóki bufor nie będzie pusty
      }
    }

    // 2. Parsujemy i wykonujemy tylko tę najświeższą komendę
    if (ostatniaKomenda.length() > 0) {
      int spaceIndex = ostatniaKomenda.indexOf(' ');
      
      if (spaceIndex > 0) {
        // Wyciągamy liczby z tekstu
        float docelowyKatPan = ostatniaKomenda.substring(0, spaceIndex).toFloat();
        float docelowyKatTilt = ostatniaKomenda.substring(spaceIndex + 1).toFloat();

        // Używamy nowych funkcji z przekładniami!
        long doceloweKrokiPan = obliczKrokiPan(docelowyKatPan);
        long doceloweKrokiTilt = obliczKrokiTilt(docelowyKatTilt);

        long ruchPan = doceloweKrokiPan - aktualnaPozycjaPan;
        long ruchTilt = doceloweKrokiTilt - aktualnaPozycjaTilt;

        // Jeśli jest jakikolwiek ruch do wykonania, kręcimy
        if (ruchPan != 0 || ruchTilt != 0) {
          wykonajRuch(ruchPan, ruchTilt);
          
          // Zapisujemy pozycję po wykonaniu ruchu
          aktualnaPozycjaPan = doceloweKrokiPan;
          aktualnaPozycjaTilt = doceloweKrokiTilt;
        }
      }
    }
  }
}
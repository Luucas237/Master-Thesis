#include <Arduino.h>
#include <SPI.h>
#include <HighPowerStepperDriver.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>
#include <AccelStepper.h>

WiFiMulti wifiMulti;


const float KROKI_NA_OBROT = 200.0; 
const float MIKROKROK = 8.0;        
const int LIMIT_PRADU_MA = 1000;


const float PRZELOZENIE_PAN = 105.0 / 20.0;
const float PRZELOZENIE_TILT = 38.0 / 24.0;

long obliczKrokiPan(float stopnie) {
  return (stopnie / 360.0) * KROKI_NA_OBROT * MIKROKROK * PRZELOZENIE_PAN;
}

long obliczKrokiTilt(float stopnie) {
  return (stopnie / 360.0) * KROKI_NA_OBROT * MIKROKROK * PRZELOZENIE_TILT;
}

HighPowerStepperDriver sd1; // PAN 
HighPowerStepperDriver sd2; // TILT 

// Piny PAN
const uint8_t SCS1_PIN = 5;
const uint8_t STEP1_PIN = 16;
const uint8_t DIR1_PIN = 4;

// Piny TILT
const uint8_t SCS2_PIN = 17;
const uint8_t STEP2_PIN = 12;
const uint8_t DIR2_PIN = 14;

// Laser
const uint8_t LASER_PIN = 25;


AccelStepper stepperPan(AccelStepper::DRIVER, STEP1_PIN, DIR1_PIN);
AccelStepper stepperTilt(AccelStepper::DRIVER, STEP2_PIN, DIR2_PIN);

String commandBuffer = "";

void setup() {
  Serial.begin(115200);
  SPI.begin(); 
  
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);

  Serial.println("\nKonfiguracja Wi-Fi...");
  wifiMulti.addAP("xxx", "yyy");


  int attempts = 0;
  while (wifiMulti.run() != WL_CONNECTED && attempts < 10) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }

  if(WiFi.status() == WL_CONNECTED) {
    Serial.println("\n--- POLACZONO Z WI-FI! ---");
    Serial.print("Adres IP urzadzenia: "); Serial.println(WiFi.localIP());
  }

  if(WiFi.status() == WL_CONNECTED) {
    ArduinoOTA.setHostname("Wiezyczka-Laserowa");
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

  stepperPan.setMaxSpeed(8000.0);     
  stepperPan.setAcceleration(6000.0); 
  
  stepperTilt.setMaxSpeed(4000.0);    
  stepperTilt.setAcceleration(3000.0);

  Serial.println("==========================================");
  Serial.println("Ready");
  Serial.println("==========================================");
}


void loop() {
  ArduinoOTA.handle();

  stepperPan.run();
  stepperTilt.run();

  while (Serial.available() > 0) {
    char c = Serial.read();
    
    if (c == '\n') {
      if (commandBuffer.length() > 2) {
        int pierwszaSpacja = commandBuffer.indexOf(' ');
        int drugaSpacja = commandBuffer.indexOf(' ', pierwszaSpacja + 1);
        
        if (pierwszaSpacja > 0 && drugaSpacja > 0) {
          float docelowyKatPan = commandBuffer.substring(0, pierwszaSpacja).toFloat();
          float docelowyKatTilt = commandBuffer.substring(pierwszaSpacja + 1, drugaSpacja).toFloat();
          int stanLasera = commandBuffer.substring(drugaSpacja + 1).toInt();

          docelowyKatTilt = constrain(docelowyKatTilt, -35.0, 35.0);

          digitalWrite(LASER_PIN, (stanLasera == 1) ? HIGH : LOW);

          long celPan = obliczKrokiPan(docelowyKatPan);
          long celTilt = obliczKrokiTilt(docelowyKatTilt);

          stepperPan.moveTo(celPan);
          stepperTilt.moveTo(celTilt);
        }
      }
      commandBuffer = "";
    } else {
      commandBuffer += c;
    }
    
    stepperPan.run();
    stepperTilt.run();
  }
}
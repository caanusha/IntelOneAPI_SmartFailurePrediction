/**
 * Developed by Anusha and Vishvesh
**/
#include "dht.h"

dht DHT;

#define DHT11_PIN 7
int soundSensorPin=A0;
int vibrationSensorPin=A1;

void setup()
{
  pinMode(soundSensorPin, INPUT);
  pinMode(vibrationSensorPin, INPUT);
  Serial.begin (9600);
}
  
void loop ()
{
  Serial.print(analogRead(soundSensorPin));
  Serial.print(",");
  Serial.print(analogRead(vibrationSensorPin));
  int chk = DHT.read11(DHT11_PIN);
  Serial.print(",");
  Serial.print(DHT.temperature);
  Serial.print(",");
  Serial.print(DHT.humidity);
  Serial.println();
  delay(20);
}

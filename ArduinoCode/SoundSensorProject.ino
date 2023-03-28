/**
 * Developed by Anusha and Vishvesh
**/
#include "dht.h"
#include "SoftwareSerial.h"
SoftwareSerial bluetooth(2, 3);
dht DHT;

#define DHT11_PIN 7
int soundPin=A0;
int vibrationPin=A1;

void setup()
{
  pinMode(soundPin, INPUT);
  pinMode(vibrationPin, INPUT);
  Serial.begin (9600);
  bluetooth.begin(9600);
}
 
void loop ()
{
  bluetooth.print(analogRead(soundPin));
  bluetooth.print(",");
  bluetooth.print(analogRead(vibrationPin));
  int chk = DHT.read11(DHT11_PIN);
  bluetooth.print(",");
  bluetooth.print(DHT.temperature);
  bluetooth.print(",");
  bluetooth.print(DHT.humidity);
  bluetooth.println();
  delay(1000);
 
}

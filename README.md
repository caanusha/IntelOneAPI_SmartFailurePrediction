The system  designed to detect and report on different  factors that can impact the engine including temperature, humidity, noise levels, and vibrations.

Here is a step-by-step guide on how to build this project:

Materials:

Arduino board (e.g., Arduino Uno)
Sound sensor module
Vibration sensor module
DHT11 humidity and temperature sensor module
Breadboard
Jumper wires
USB cable
Computer with Arduino IDE installed
Step 1: Connect the sound sensor module
Connect the sound sensor module to the Arduino board using jumper wires. The sound sensor has three pins: VCC, GND, and AOUT. Connect the VCC pin to the 5V pin on the Arduino board, the GND pin to the GND pin on the board, and the AOUT pin to an analog input pin on the board (e.g., A0).

Step 2: Connect the vibration sensor module
Connect the vibration sensor module to the Arduino board using jumper wires. The vibration sensor has three pins: VCC, GND, and SIG. Connect the VCC pin to the 5V pin on the Arduino board, the GND pin to the GND pin on the board, and the SIG pin to an analog input pin on the board (e.g., A1).

Step 3: Connect the DHT11 humidity and temperature sensor module
Connect the DHT11 humidity and temperature sensor module to the Arduino board using jumper wires. The sensor module has three pins: VCC, GND, and DATA. Connect the VCC pin to the 5V pin on the Arduino board, the GND pin to the GND pin on the board, and the DATA pin to a digital input/output pin on the board (e.g., pin 2).

Step 4: Write the code
Open the Arduino IDE and write the code to read data from the sensors and send it to the serial monitor "COM3"

Step 5: Write the Python code to train the Sensor data using Intel OneApi
Write the Python code to read sensor data from the serial port "COM3" and use that data to train using Intel OneAPI DNN machine learning model to train the sensor data. Need to train using both Good and Bad Engine data to accurately predict the Engine state

Step 6: Write the Python code to predict the Engine State
Write the Python code to read sensor data from the serial port "COM3" and use that data to predict the Engine State using trained Intel OneAPI DNN machine learning model

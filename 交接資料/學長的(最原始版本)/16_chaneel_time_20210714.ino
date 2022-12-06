unsigned long start;
unsigned long sec;
unsigned long secc;

int first = 0;

int s0 = 8;
int s1 = 9;
int s2 = 10;
int s3 = 11;
int SIG_pin = 0;
void setup() {
  pinMode(s0, OUTPUT);
  pinMode(s1, OUTPUT);
  pinMode(s2, OUTPUT);
  pinMode(s3, OUTPUT);
  digitalWrite(s0, LOW);
  digitalWrite(s1, LOW);
  digitalWrite(s2, LOW);
  digitalWrite(s3, LOW);
  pinMode(13, OUTPUT); //輸出
  digitalWrite(13, HIGH);
  Serial.begin(9600);
  //Serial.println("Voltage0,Voltage1,Voltage2,Voltage3,Voltage4,Voltage5,Voltage6,Voltage7,Voltage8,Voltage9,Voltage10,Voltage11,Voltage12,Voltage13,Voltage14,Voltage15,Dir");

  for (int i = 0; i < 16; i++) {
    int v = readMux(i);
    first += v;
  }

  Serial.println("請揮");
}

char var;
int temp = 0;
boolean flag = false;
int _now = 0;
int j = 0;

void loop() {
  int scanloop = 0;
  String Value;
  int sed = 0;
  for (int i = 0; i < 16; i++) {
          int v = readMux(i);
          sed += v;
  }
  
  if(flag == false && first - sed > 500){
    flag = true;
    _now = millis(); // 0 ~ 2000
    
  }
  int clocking = millis();
  
  
  if (flag && j < 20  ){
      j ++;
      
      int vols[15];
      for (int i = 0; i < 16; i++) {
        int v = readMux(i);
        vols[i] = v;

        Serial.print(String(v) + ',');
        
      }
      

      Serial.println("");
      temp = clocking;
  }else{
    if(flag == true){
      clocking = millis();
      //Serial.println( String(clocking - _now) + "ms" );
      Serial.println("finished");
    }
    
    
    flag = false;
    j = 0;

   

    delay(1000);
    Serial.println("請揮");
  }



  {
    while (Serial.available() > 0)
    {
      var = Serial.read();
      if (var == '0') {
        digitalWrite(13, LOW);
      }
      if (var == '1') {
        digitalWrite(13, HIGH);
      }
      if (var == '2') {
       digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
      }
      if (var == '3') {
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
      }
      if (var == '4') {
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
        delay(500);
        digitalWrite(13, HIGH);
        delay(500);
        digitalWrite(13, LOW);
      }
    }
  
}

//delay(1000);
}



void readSensor_leftToright(float startTime)
{
  int v;
  for (int i = 0; i < 16; i++) {
    v = readMux(i) * 3;
    sec = millis() - startTime;
    //Serial.print("{\"address\":" + String(i) + ",");
    Serial.print(String(v));
    //Serial.print("\"second\":" + String(sec) + "}");
    delay(150);
    if (i < 16)
      Serial.print(",");
    delay(7);
  }

}

void readSensor_rightToleft()
{
  int v;
  start = millis();
  for (int i = 0; i < 16; i++) {
    v = readMux(i) * 3;
    sec = millis() - start;
    //Serial.print("{\"address\":" + String(i) + ",");
    Serial.print(String(v));
    //Serial.print("\"second\":" + String(sec) + "}");
    delay(150);
    if (i < 16)
      Serial.print(",");
    delay(7);
  }

}

void readSensor_leftTorightone(float startTime)
{
  int v;

  for (int i = 0; i < 16; i++) {
    v = readMux(i) * 5;
    sec = millis() - startTime;
    Serial.print("{\"address\":" + String(i) + ",");
    Serial.print("\"voltage\":" + String(v) + ",");
    Serial.print("\"second\":" + String(sec) + "}");
    if (i <= 16)
      Serial.print(",");
    delay(1);
  }

}


void readSensor_rightToleftone()
{
  int v;
  start = millis();
  //Serial.print("[");
  for (int i = 0; i < 16; i++) {
    v = readMux(i) * 5;
    sec = millis() - start;
    Serial.print("{\"address\":" + String(i) + ",");
    Serial.print("\"voltage\":" + String(v) + ",");
    Serial.print("\"second\":" + String(sec) + "}");

    if (i <= 14)
      Serial.print(",");
    delay(1);
  }

}


int readMux(int channel) {
  int controlPin[] = {s0, s1, s2, s3};
  int muxChannel[16][4] = {
    {0, 0, 0, 0}, //channel 0
    {1, 0, 0, 0}, //channel 1
    {0, 1, 0, 0}, //channel 2
    {1, 1, 0, 0}, //channel 3
    {0, 0, 1, 0}, //channel 4
    {1, 0, 1, 0}, //channel 5
    {0, 1, 1, 0}, //channel 6
    {1, 1, 1, 0}, //channel 7
    {0, 0, 0, 1}, //channel 8
    {1, 0, 0, 1}, //channel 9
    {0, 1, 0, 1}, //channel 10
    {1, 1, 0, 1}, //channel 11
    {0, 0, 1, 1}, //channel 12
    {1, 0, 1, 1}, //channel 13
    {0, 1, 1, 1}, //channel 14
    {1, 1, 1, 1} //channel 15
  };

  for (int i = 0; i < 4; i ++) {
    digitalWrite(controlPin[i], muxChannel[channel][i]);
  }
  int val = analogRead(SIG_pin);
  //return the value
  return val;
}

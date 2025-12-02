import paho.mqtt.client as mqtt
import json
import random
import time
from datetime import datetime

# إعدادات MQTT Broker (محلي مثلاً)
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
TOPIC_BASE = "city"

# عدد المناطق وعدد البيوت في كل منطقة
REGION= "hybrid"
HOUSES_PER_REGION = 100

# إنشاء عميل MQTT
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

def generate_smart_meter_data(house_id):
    """توليد بيانات عشوائية لعداد ذكي"""
    voltage = round(random.uniform(210, 240), 2)
    current = round(random.uniform(0.5, 10.0), 2)
    power = round(voltage * current, 2)
    energy_consumed = round(random.uniform(0.0, 50.0), 2)

    return {
        "house_id": house_id,
        "device_type": "smart_meter",
        "data": {
            "voltage": voltage,
            "current": current,
            "power": power,
            "energy_consumed": energy_consumed
        },
        "timestamp": datetime.utcnow().isoformat()
    }

while True:
   
    for house_id in range(1, HOUSES_PER_REGION + 1):
        payload = generate_smart_meter_data(house_id)
        topic = f"{TOPIC_BASE}/{REGION}/smartmeter/house{house_id}"
            
        client.publish(topic, json.dumps(payload))
        print(f"Sent data -- {topic}: {payload}")

    time.sleep(5)  # إرسال كل 5 ثوانٍ

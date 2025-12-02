import paho.mqtt.client as mqtt
import json
import random
import time
from datetime import datetime

# إعدادات MQTT Broker (محلي)
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
TOPIC_BASE = "city"

# عدد المناطق وعدد feeders في كل منطقة (عادة واحد لكل منطقة)
REGION = "downtown"

# إنشاء عميل MQTT
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

def generate_feeder_data(feeder_id):
    """توليد بيانات عشوائية للـ Feeder"""
    frequency = round(random.uniform(49.5, 50.5), 2)
    
    return {
        "feeder_id": feeder_id,
        "device_type": "feeder",
        "data": {
            "frequency": frequency
        },
        "timestamp": datetime.utcnow().isoformat()
    }


while True:
    feeder_id = f"feeder_{REGION}" 
    payload = generate_feeder_data(feeder_id)
    topic = f"{TOPIC_BASE}/region{REGION}/feeder/{feeder_id}"
    client.publish(topic, json.dumps(payload))
    print(f"Sent data -- {topic}: {payload}")
        
    time.sleep(5) 

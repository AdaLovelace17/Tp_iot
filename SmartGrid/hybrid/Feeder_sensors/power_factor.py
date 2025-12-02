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
REGION = "hybrid"

# إنشاء عميل MQTT
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

def generate_feeder_data(feeder_id):
    """توليد بيانات عشوائية للـ Feeder"""
    power_factor = round(random.uniform(0.8, 1.0), 2)

    
    return {
        "feeder_id": feeder_id,
        "device_type": "feeder",
        "data": {
            "power_factor": power_factor
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

import paho.mqtt.client as mqtt
import json
import random
import time
from datetime import datetime

BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
TOPIC_BASE = "city"
REGION = "hybrid"

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

def generate_temperature_data():
    return {
        "device_type": "env_sensor",
        "sensor_type": "temperature",
        "data": round(random.uniform(-5, 40), 2),
        "timestamp": datetime.utcnow().isoformat()
    }

while True:
    payload = generate_temperature_data()
    topic = f"{TOPIC_BASE}/{REGION}/env/temperature"
    client.publish(topic, json.dumps(payload))
    print(f"Sent data -- {topic}: {payload}")
    time.sleep(5)
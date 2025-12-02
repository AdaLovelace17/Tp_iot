import paho.mqtt.client as mqtt
import json
import random
import time
from datetime import datetime

BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
TOPIC_BASE = "city"
REGION = "port"

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER_ADDRESS, BROKER_PORT, 60)

def generate_humidity_data():
    return {
        "device_type": "env_sensor",
        "sensor_type": "humidity",
        "data": round(random.uniform(20, 90), 2),
        "timestamp": datetime.utcnow().isoformat()
    }

while True:
    payload = generate_humidity_data()
    topic = f"{TOPIC_BASE}/{REGION}/env/humidity"
    client.publish(topic, json.dumps(payload))
    print(f"Sent data -- {topic}: {payload}")
    time.sleep(5)
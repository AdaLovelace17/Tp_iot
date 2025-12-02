import paho.mqtt.client as mqtt
import json
import random
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
TOPIC_BASE = "city"
REGION = "commercial"

# MQTT Client Setup
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        logger.info(f"Temperature sensor connected to broker")
    else:
        logger.error(f"Connection failed with code: {reason_code}")

def on_disconnect(client, userdata, reason_code, properties=None):
    logger.warning(f"Disconnected from broker. Code: {reason_code}")
    if reason_code != 0:
        logger.info("Attempting to reconnect...")

client.on_connect = on_connect
client.on_disconnect = on_disconnect

def generate_temperature_data():
    return {
        "device_type": "env_sensor",
        "sensor_type": "temperature",
        "data": round(random.uniform(-5, 40), 2),
        "unit": "celsius",
        "timestamp": datetime.utcnow().isoformat()
    }

try:
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_start()
    
    while True:
        try:
            payload = generate_temperature_data()
            # ✅ FIXED: Correct topic for temperature
            topic = f"{TOPIC_BASE}/{REGION}/env/temperature"
            result = client.publish(topic, json.dumps(payload))
            
            if result.rc == 0:
                logger.info(f"Published to {topic}: {payload['data']}°C")
            else:
                logger.error(f"Failed to publish. Return code: {result.rc}")
                
        except Exception as e:
            logger.error(f"Error generating/publishing data: {e}")
        
        time.sleep(5)
        
except KeyboardInterrupt:
    logger.info("Shutting down temperature sensor...")
except Exception as e:
    logger.error(f"Fatal error: {e}")
finally:
    client.loop_stop()
    client.disconnect()
    logger.info("Temperature sensor stopped")
import paho.mqtt.client as mqtt
import json
import logging
from pymongo import MongoClient
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "smart_grid"
REGION = "commercial"  # ✅ Change this for each gateway instance
# REGION options: commercial, downtown, hybrid, port, residential

# Topic subscription pattern
TOPIC = f"city/{REGION}/#"

# MongoDB Connection
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[DB_NAME]
    
    # Collections
    smartmeter_collection = db["smartmeters"]
    feeder_collection = db["feeders"]
    env_collection = db["env_sensors"]
    
    logger.info(f"✅ Connected to MongoDB: {DB_NAME}")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    exit(1)

def on_connect(client, userdata, flags, reason_code, properties=None):
    logger.info(f"✅ Gateway [{REGION}] connected to MQTT Broker")
    client.subscribe(TOPIC)
    logger.info(f"📡 Subscribed to: {TOPIC}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        
        # ✅ IMPORTANT: Ensure region_name is set correctly
        payload["region_name"] = REGION  # Force the correct region
        payload["received_at"] = datetime.utcnow().isoformat()
        payload["topic"] = msg.topic
        
        device_type = payload.get("device_type", "")
        
        # Route to appropriate collection
        if device_type == "smart_meter":
            house_id = payload.get("house_id", "unknown")
            result = smartmeter_collection.insert_one(payload)
            logger.info(f"✅ Stored smart_meter: {house_id} | Region: {REGION}")
            
        elif device_type == "feeder":
            feeder_id = payload.get("feeder_id", "unknown")
            
            # Optional: Aggregate feeder data if needed
            existing = feeder_collection.find_one(
                {"feeder_id": feeder_id, "region_name": REGION},
                sort=[("timestamp", -1)]
            )
            
            # If recent data exists, merge it
            if existing and "data" in existing:
                payload["data"] = {**existing["data"], **payload.get("data", {})}
            
            result = feeder_collection.insert_one(payload)
            logger.info(f"✅ Stored feeder: {feeder_id} | Region: {REGION}")
            
        elif device_type == "env_sensor":
            sensor_type = payload.get("sensor_type", "unknown")
            result = env_collection.insert_one(payload)
            logger.info(f"✅ Stored env_sensor: {sensor_type} | Region: {REGION}")
            
        else:
            logger.warning(f"⚠️ Unknown device type: {device_type}")
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON received: {e}")
    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")

def on_disconnect(client, userdata, reason_code, properties=None):
    logger.warning(f"⚠️ Gateway disconnected. Code: {reason_code}")
    if reason_code != 0:
        logger.info("🔄 Attempting to reconnect...")

# MQTT Client Setup
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

# Start Gateway
try:
    logger.info(f" Starting Gateway for region: {REGION}")
    logger.info(f" Make sure sensors publish to: city/{REGION}/...")
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    logger.info("\n Shutting down gateway...")
except Exception as e:
    logger.error(f" Fatal error: {e}")
finally:
    client.disconnect()
    mongo_client.close()
    logger.info(" Gateway stopped")
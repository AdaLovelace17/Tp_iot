import paho.mqtt.client as mqtt
import json
import logging
from pymongo import MongoClient
from datetime import datetime

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
BROKER_ADDRESS = "localhost"
BROKER_PORT = 1883
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "smart_grid"

# ✅ IMPORTANT: Change this for each gateway instance
REGION = "downtown"  
# Options: commercial, downtown, hybrid, port, residential

# ✅ CORRECT TOPIC FORMAT
TOPIC = f"city/{REGION}/#"  # ← Fixed!

# ============================================
# MONGODB CONNECTION
# ============================================
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

# ============================================
# MQTT CALLBACKS
# ============================================
def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        logger.info(f"✅ Gateway [{REGION}] connected to MQTT Broker")
        client.subscribe(TOPIC)
        logger.info(f"📡 Subscribed to: {TOPIC}")
    else:
        logger.error(f"❌ Connection failed with code: {reason_code}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        
        # ✅ Add metadata
        payload["region_name"] = REGION  # Force correct region
        payload["received_at"] = datetime.utcnow().isoformat()
        payload["topic"] = msg.topic
        
        device_type = payload.get("device_type", "")
        
        # ============================================
        # ROUTE TO APPROPRIATE COLLECTION
        # ============================================
        if device_type == "smart_meter":
            house_id = payload.get("house_id", "unknown")
            result = smartmeter_collection.insert_one(payload)
            logger.info(f"✅ Stored smart_meter: {house_id} | Region: {REGION}")
            
        elif device_type == "feeder":
            feeder_id = payload.get("feeder_id", "unknown")
            
            # ✅ OPTIONAL: Aggregate feeder data
            existing = feeder_collection.find_one(
                {"feeder_id": feeder_id, "region_name": REGION},
                sort=[("timestamp", -1)]
            )
            
            # Merge with existing data if available
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
        
        # ✅ Display received data (optional, for debugging)
        logger.info(f"📦 Data from {msg.topic}:")
        logger.info(json.dumps(payload, indent=2, default=str))
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON received: {e}")
    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")

def on_disconnect(client, userdata, reason_code, properties=None):
    logger.warning(f"⚠️ Gateway disconnected. Code: {reason_code}")
    if reason_code != 0:
        logger.info("🔄 Attempting to reconnect...")

# ============================================
# MQTT CLIENT SETUP
# ============================================
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

# ============================================
# START GATEWAY
# ============================================
try:
    logger.info(f"🚀 Starting Gateway for region: {REGION}")
    logger.info(f"📡 Make sure sensors publish to: city/{REGION}/...")
    
    client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
    client.loop_forever()
    
except KeyboardInterrupt:
    logger.info("\n🛑 Shutting down gateway...")
    
except Exception as e:
    logger.error(f"❌ Fatal error: {e}")
    
finally:
    client.disconnect()
    mongo_client.close()
    logger.info("✅ Gateway stopped")
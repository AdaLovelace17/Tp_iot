import paho.mqtt.client as mqtt
from pymongo import MongoClient
import json
import certifi
from datetime import datetime

# ---- MongoDB Cloud Connection ----
mongo_client = MongoClient(
    "mongodb+srv://AdaLovelace:AdaLovelace1817@cluster0.jfdolkd.mongodb.net/?retryWrites=true&w=majority",
    tls=True,
    tlsCAFile=certifi.where()
)
db = mongo_client["SmartGrid"]

# Collections منفصلة لكل نوع حساس
smartmeter_collection = db["smartmeters"]
feeder_collection = db["feeders"]
env_collection = db["env_sensors"]

print("Connected to MongoDB Atlas")

# ---- MQTT Setup ----
BROKER = "localhost"
PORT = 1883
TOPIC = "city/#"

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with code: {rc}")
    client.subscribe(TOPIC)
    print(f" Subscribed to topic: {TOPIC}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)

        # إضافة topic للتمييز
        data["topic"] = msg.topic
        
        # استخراج region_name من topic بشكل صحيح
        # Topic format: city/commercial/smart_meter/house_1
        if "region_name" not in data or data["region_name"] is None:
            try:
                topic_parts = msg.topic.split("/")
                if len(topic_parts) >= 2:
                    region_name = topic_parts[1]  # commercial, downtown, etc.
                    data["region_name"] = region_name
                    print(f"Extracted region: {region_name}")
                else:
                    data["region_name"] = "unknown"
                    print("Could not extract region from topic")
            except Exception as e:
                data["region_name"] = "unknown"
                print(f"Error extracting region: {e}")

        # حفظ البيانات حسب نوع الجهاز
        device_type = data.get("device_type", "")
        
        if device_type == "smart_meter":
            smartmeter_collection.insert_one(data)
            house_id = data.get("house_id", "unknown")
            print(f"Saved smart_meter: {house_id} | Region: {data['region_name']}")
            
        elif device_type == "feeder":
            feeder_collection.insert_one(data)
            feeder_id = data.get("feeder_id", "unknown")
            print(f" Saved feeder: {feeder_id} | Region: {data['region_name']}")
            
        elif device_type == "env_sensor":
            env_collection.insert_one(data)
            sensor_type = data.get("sensor_type", "unknown")
            print(f"Saved env_sensor: {sensor_type} | Region: {data['region_name']}")
            
        else:
            print(f" Unknown device type: {device_type} from topic {msg.topic}")

    except json.JSONDecodeError as e:
        print(f" Invalid JSON: {e}")
    except Exception as e:
        print(f" Error processing message: {e}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"Unexpected disconnection. Code: {rc}")

# ---- تشغيل MQTT Client ----
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

try:
    print(" Starting Cloud MQTT Client...")
    client.connect(BROKER, PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\n Shutting down cloud client...")
    client.disconnect()
    mongo_client.close()
except Exception as e:
    print(f" Fatal error: {e}")
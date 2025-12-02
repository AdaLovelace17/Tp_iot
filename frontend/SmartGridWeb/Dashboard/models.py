# models.py
# We're using MongoDB, so these are reference models (not actual Django ORM models)
# They help document the data structure

"""
Smart Meter Document Structure:
{
    "_id": ObjectId,
    "house_id": "commercial_house_1",
    "device_type": "smart_meter",
    "region_name": "commercial",
    "data": {
        "voltage": 220.5,
        "current": 5.2,
        "power": 1.15,
        "energy_consumed": 25.3,
        "power_factor": 0.92
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "received_at": "2025-01-15T10:30:01Z"
}

Feeder Document Structure:
{
    "_id": ObjectId,
    "feeder_id": "feeder_commercial",
    "device_type": "feeder",
    "region_name": "commercial",
    "data": {
        "load_current": 125.5,
        "load_voltage": 225.0,
        "frequency": 50.1,
        "power_factor": 0.95
    },
    "timestamp": "2025-01-15T10:30:00Z"
}

Environmental Sensor Document Structure:
{
    "_id": ObjectId,
    "device_type": "env_sensor",
    "sensor_type": "temperature",  # or "humidity", "wind_speed"
    "region_name": "commercial",
    "data": 24.5,
    "unit": "celsius",  # or "%", "m/s"
    "timestamp": "2025-01-15T10:30:00Z"
}
"""

# Helper functions for data validation
def validate_smart_meter(data):
    required = ["house_id", "device_type", "data", "timestamp"]
    return all(field in data for field in required)

def validate_feeder(data):
    required = ["feeder_id", "device_type", "data", "timestamp"]
    return all(field in data for field in required)

def validate_env_sensor(data):
    required = ["device_type", "sensor_type", "data", "timestamp"]
    return all(field in data for field in required)
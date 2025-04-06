from flask import Flask, jsonify, request
from pymongo import MongoClient
from datetime import datetime
from jsonschema import validate, ValidationError

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['points']  # Replace with your database name
addresses = db['points_collection']

# Insert a sample coordinate (for testing/demo)
addresses.insert_one({
    'type': 'Point',
    'coordinates': [41.39826, 17.13559],
    'time': datetime.now()
})

# Global variable to store the latest OBD data
latest_obd_data = None

# JSON schema for a point
point_schema = { 
    "type": "object",
    "properties": {
        "coordinates": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
        },
        "car_number": {"type": "string"},
        "route_id": {"type": "integer"}
    },
    "required": ["coordinates", "car_number", "route_id"]
}

# Endpoint to get all points (GeoJSON format)
@app.route('/points', methods=['GET'])
def get_all_points():
    points = []
    for address in addresses.find({'type': 'Point'}):
        points.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": address['coordinates']
            },
            "properties": {
                "time": address.get('time'),
                "car_number": address.get('car_number'),
                "route_id": address.get('route_id')
            }
        })
    return jsonify(points)

# Endpoint to get the most recent coordinate
@app.route('/last_point', methods=['GET'])
def get_last_point():
    last_address = addresses.find_one({'type': 'Point'}, sort=[('time', -1)])
    if last_address:
        result = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": last_address['coordinates']
            },
            "properties": {
                "time": last_address.get('time'),
                "car_number": last_address.get('car_number'),
                "route_id": last_address.get('route_id')
            }
        }
        return jsonify(result)
    return jsonify({"error": "No points found"}), 404

# Endpoint to add a new point (from Raspberry Pi via GSM)
@app.route('/add_point', methods=['POST'])
def add_point():
    data = request.get_json()
    try:
        validate(instance=data, schema=point_schema)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    
    addresses.insert_one({
        'type': 'Point',
        'coordinates': data['coordinates'],
        'car_number': data['car_number'],
        'route_id': data['route_id'],
        'time': datetime.now()  # Store the current time
    })
    return jsonify({'status': 'success'}), 201

# Endpoint to send a kill switch command
@app.route('/kill_switch', methods=['POST'])
def kill_switch():
    # Simulate sending a kill switch signal to the Raspberry Pi.
    command = {"command": "kill_switch", "signal": 1, "timestamp": datetime.now()}
    print("Kill switch activated:", command)
    response = {
        "status": "Kill switch activated",
        "notification": "Kill switch is on, command sent to Raspberry Pi."
    }
    return jsonify(response), 200

# Endpoint to receive OBD data from Raspberry Pi
@app.route('/obd', methods=['POST'])
def receive_obd():
    global latest_obd_data
    obd_data = request.get_json()
    # You can add validation for OBD data as needed.
    print("Received OBD data:", obd_data)
    # Store the latest OBD data
    latest_obd_data = {
        "data": obd_data,
        "timestamp": datetime.now()
    }
    response = {
        "status": "OBD data received",
        "notification": "New OBD data has been reported."
    }
    return jsonify(response), 200

# Endpoint for the mobile app to get the latest OBD notification
@app.route('/latest_obd', methods=['GET'])
def get_latest_obd():
    if latest_obd_data:
        return jsonify(latest_obd_data)
    return jsonify({"status": "No new OBD data"}), 200

if __name__ == "__main__":
    app.run(debug=True)

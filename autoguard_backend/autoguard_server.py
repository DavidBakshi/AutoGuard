from datetime import datetime
from flask import Flask, jsonify, request


app = Flask(__name__)

# Existing “start” location state
location = {
    "latitude": 31.77759352590111,
    "longitude": 35.197599143135946,
    "latitudeDelta": 0.05,
    "longitudeDelta": 0.05
}

# Store last OBD detection
obd_status = {
    "detected": False,
    "time": ""
}


def get_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" ")[1]


@app.route('/current_location', methods=['GET'])
def get_current_location():
    return jsonify(location)


@app.route('/set_location', methods=['POST'])
def update_location():
    data = request.json
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    location.update({
        "latitude": data['latitude'],
        "longitude": data['longitude'],
        "latitudeDelta": data.get('latitudeDelta', location['latitudeDelta']),
        "longitudeDelta": data.get('longitudeDelta', location['longitudeDelta']),
    })
    return jsonify({"message": "Location updated successfully", "new_location": location}), 200


# New POST route: record that OBD was detected
@app.route('/obd_detected', methods=['POST'])
def record_obd_detected():
    obd_status['detected'] = True
    obd_status['time'] = get_time()
    return jsonify({
        "message": "OBD detection recorded",
        "obd_status": obd_status
    }), 200


# New GET route: fetch the last OBD status
@app.route('/obd_detected', methods=['GET'])
def get_obd_status():
    return jsonify(obd_status), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

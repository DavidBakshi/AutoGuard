# GSM GPS Client

This folder contains Python scripts to interact with a SIM7600 (or similar) GSM+GPS module connected to a Raspberry Pi. It allows sending HTTP requests and fetching GPS coordinates over serial using AT commands.

## Requirements
- Raspberry Pi
- SIM7600X

## Files

### `send_gps_post.py`
- Sends JSON payload (latitude, longitude) using an HTTP POST.
- Used to report OBD access events or current location.
- Communicates with remote server via GSM network.

### `send_get.py`
- Sends an HTTP GET request.
- Useful for notifying server events like system boot or requesting configuration.

### `gps.py`
- Powers on the SIM7600 module using GPIO.
- Starts GPS session, queries for location using `AT+CGPSINFO`.
- Powers off the module after reading location.
- Good for testing GPS readiness and signal.

## Use Case
These scripts are designed for a car security system (e.g., Auto-Guard) where:
- GPS data is fetched and sent on OBD tampering detection.
- System status is queried or reported remotely.

## Running the scripts
Make sure `/dev/ttyS0` is connected and you have permission:
```bash
sudo usermod -a -G dialout $USER
sudo apt install python3-serial

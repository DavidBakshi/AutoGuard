import serial
import time

# Initialize the serial connection (adjust the port and baud rate as needed)
ser = serial.Serial('/dev/ttyS0', 115200, timeout=1)  # Adjust the port if needed
ser.flushInput()


def send_at(command, back, timeout=1):
    """ Send AT command and check the response """
    ser.write((command + '\r\n').encode())
    time.sleep(timeout)
    if ser.inWaiting():
        time.sleep(0.01)
        rec_buff = ser.read(ser.inWaiting()).decode().strip()
        print(f"Sent: {command}")  # Print the sent command
        print(f"Response: {rec_buff}")  # Print the response
        if back in rec_buff:
            return True
        else:
            print(f"ERROR: {rec_buff}")
            return False
    return False


def http_get_request(url):
    """ Perform an HTTP GET request using AT commands """
    
    # Try to initialize HTTP service
    if not send_at('AT+HTTPINIT', 'OK', 2):
        print("HTTP INIT failed. Attempting to terminate any previous session...")
        
        # If initialization fails, try terminating any previous HTTP session
        if send_at('AT+HTTPTERM', 'OK', 2):
            print("Previous HTTP session terminated successfully.")
            # Retry initializing the HTTP service
            if not send_at('AT+HTTPINIT', 'OK', 2):
                print("HTTP INIT still failed after terminating previous session.")
                return False
        else:
            print("Failed to terminate previous HTTP session.")
            return False

    # Set the URL for the GET request
    if not send_at(f'AT+HTTPPARA="URL","{url}"', 'OK', 2):
        return False

    # Send HTTP GET request
    if not send_at('AT+HTTPACTION=0', '+HTTPACTION:'):
        return False

    # Check the HTTP response header
    if not send_at('AT+HTTPHEAD', 'OK', 2):
        return False

    # Terminate the HTTP session
    send_at('AT+HTTPTERM', 'OK', 2)
    return True

def main():
    # Example URL for testing

    url = "http://example.com"
    print("Starting HTTP GET request...")
    
    if http_get_request(url):
        print("HTTP GET request completed successfully.")
    else:
        print("HTTP GET request failed.")


if __name__ == "__main__":
    main()

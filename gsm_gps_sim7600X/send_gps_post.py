import serial, time

ser = serial.Serial('/dev/ttyS0', 115200, timeout=1)
ser.flushInput()

def at(cmd, timeout=0.5):
    ser.write((cmd+'\r\n').encode())
    time.sleep(timeout)
    resp = ''
    while ser.inWaiting():
        resp += ser.read(ser.inWaiting()).decode()
        time.sleep(0.05)
    return resp.strip()

def http_post(url, body):
    # Init
    print(at('AT+HTTPTERM'))
    print(at('AT+HTTPINIT'))
    print(at(f'AT+HTTPPARA="URL","{url}"'))
    print(at('AT+HTTPPARA="CONTENT","application/json"'))
    # Send body
    print(at(f'AT+HTTPDATA={len(body)},10000', 0.1))
    ser.write(body.encode())
    time.sleep(0.5)
    # Fire POST
    print(at('AT+HTTPACTION=1', 0.1))   # prints OK
    # Now wait for +HTTPACTION URC
    deadline = time.time() + 15
    urc = None
    while time.time() < deadline:
        line = ser.readline().decode().strip()
        if not line: 
            continue
        print('URC:', line)
        if line.startswith('+HTTPACTION:'):
            urc = line
            break
    if not urc:
        print("❌ no +HTTPACTION URC—timeout")
        return False
    # Proceed
    print(at('AT+HTTPHEAD'))
    print(at('AT+HTTPREAD'))
    print(at('AT+HTTPTERM'))
    return True

if __name__=='__main__':
    url = "http://10.0.0.5:5000/location"
    body = '{"latitude":32.309333,"longitude":34.855499}'
    if http_post(url, body):
        print("✅ POST succeeded")
    else:
        print("❌ POST failed")

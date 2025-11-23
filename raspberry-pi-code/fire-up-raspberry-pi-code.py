import time, threading
from datetime import datetime
from flask import Flask, Response, jsonify
import cv2
from picamera2 import Picamera2
from DFRobot_DHT20 import DFRobot_DHT20

# camera (mjpeg + fps) 
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration({"format": "BGR888"}))

picam2.start()
time.sleep(0.5)

latest_jpeg = None
cam_fps = 0.0

def camera_loop():
    global latest_jpeg, cam_fps
    frames = 0
    tick = time.time()
    while True:
        frame = picam2.capture_array()  # BGR
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            latest_jpeg = buf.tobytes()
        frames += 1
        if frames % 10 == 0:
            now = time.time()
            cam_fps = 10.0 / (now - tick)
            tick = now

threading.Thread(target=camera_loop, daemon=True).start()

# dht20 temp & humidity sensor initialization i2c-1 at 0x38 
dht20 = DFRobot_DHT20(0x01, 0x38)  # bus=1, address=0x38

# light sensor bh1750 at 0x5C 
import smbus
bus = smbus.SMBus(1)

class LightSensor():
    def __init__(self):
        # Define some constants from the datasheet
        self.DEVICE = 0x5c  # Default device I2C address

        self.POWER_DOWN = 0x00  # No active state
        self.POWER_ON   = 0x01  # Power on
        self.RESET      = 0x07  # Reset data register value

        # Start measurement at 4lx resolution. Time typically 16ms.
        self.CONTINUOUS_LOW_RES_MODE   = 0x13
        # Start measurement at 1lx resolution. Time typically 120ms
        self.CONTINUOUS_HIGH_RES_MODE_1 = 0x10
        # Start measurement at 0.5lx resolution. Time typically 120ms
        self.CONTINUOUS_HIGH_RES_MODE_2 = 0x11
        # Start measurement at 1lx resolution. Time typically 120ms
        # Device is automatically set to power down after measurement.
        self.ONE_TIME_HIGH_RES_MODE_1 = 0x20
        # Start measurement at 0.5lx resolution. Time typically 120ms
        # Device is automatically set to power down after measurement.
        self.ONE_TIME_HIGH_RES_MODE_2 = 0x21
        # Start measurement at 1lx resolution. Time typically 120ms
        # Device is automatically set to power down after measurement.
        self.ONE_TIME_LOW_RES_MODE = 0x23

    def convertToNumber(self, data):
        # convert 2 bytes of data into a decimal number
        return ((data[1] + (256 * data[0])) / 1.2)

    def readLight(self):
        data = bus.read_i2c_block_data(self.DEVICE, self.ONE_TIME_HIGH_RES_MODE_1)
        return self.convertToNumber(data)

light = LightSensor()

# Shared state (10s cadence)
sensors_state = {
    "time": None,
    "temp_c": None,
    "humidity_pct": None,
    "brightness_lux": None,
}

def sensors_loop():
    global sensors_state
    while True:
        # DHT20 (with −10°C offset because of device temperature, adjust accordingly based on your raspberry pi's performance)
        T_c, humidity, crc_error = dht20.get_temperature_and_humidity()
        T_adj = (T_c - 10) if T_c is not None else None

        # BH1750 
        lux = light.readLight()

        sensors_state = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "temp_c": T_adj,
            "humidity_pct": humidity,
            "brightness_lux": int(lux),

        }
        time.sleep(5)

threading.Thread(target=sensors_loop, daemon=True).start()

# Flask connection for streamlit

app = Flask(__name__)

@app.get("/frame.jpg")
def frame_jpg():
    if latest_jpeg is None:
        return ("No frame yet", 503)
    return Response(latest_jpeg, mimetype="image/jpeg")

@app.get("/mjpeg")
def mjpeg():
    def gen():
        while True:
            if latest_jpeg is not None:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_jpeg + b"\r\n")
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/sensors")
def sensors():
    return jsonify({
        "time": sensors_state["time"],
        "temp_c": sensors_state["temp_c"],
        "humidity_pct": sensors_state["humidity_pct"],
        "brightness_lux": sensors_state["brightness_lux"],
        "fps": round(cam_fps, 2),
    })

@app.get("/")
def root():
    return "Pi Edge: /mjpeg, /frame.jpg, /sensors"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
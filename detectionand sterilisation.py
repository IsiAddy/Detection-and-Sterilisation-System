# Food Scanner - Raspberry Pi
# Detection (1.5 min) -> LoG blob count -> UV sterilisation (20 min) -> shutdown

import os
import cv2
import numpy as np
from datetime import datetime
from time import sleep
from skimage.feature import blob_log

from RPLCD.i2c import CharLCD
from gpiozero import LED, OutputDevice, Button, Buzzer
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)

uv_lamp    = LED(22)
blue_led   = LED(6)
lamp_relay = OutputDevice(16, active_high=False)
start_btn  = Button(5)
buzzer     = Buzzer(26)

lcd = CharLCD('PCF8574', 0x27, cols=16, rows=2)

VIDEO_DURATION = 90000   # 1.5 minutes in milliseconds
UV_DURATION    = 1200    # 20 minutes in seconds


def show_lcd(line1="", line2=""):
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string(line1[:16])
    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2[:16])


def beep():
    buzzer.on()
    sleep(1)
    buzzer.off()


def capture_image(filename="capture.jpg"):
    os.system(f"rpicam-jpeg -o {filename}")
    return filename


def capture_video():
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    filename  = f"detection_{timestamp}.mp4"
    os.system(f"rpicam-vid -t {VIDEO_DURATION} -o {filename}")


def set_relay_blue():
    lamp_relay.off()
    sleep(0.01)
    lamp_relay.on()
    sleep(0.03)
    lamp_relay.off()
    lamp_relay.on()
    sleep(0.3)


def run_excitation_lighting():
    blue_led.off()
    set_relay_blue()
    blue_led.on()
    capture_video()
    blue_led.off()


def run_uv_sterilisation():
    uv_lamp.on()
    sleep(UV_DURATION)
    uv_lamp.off()
#Laplacian of Gaussian

def detect_bacteria(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"Could not open image: {image_path}")
        return 0

    img        = cv2.resize(img, (512, 512))
    blue       = img[:, :, 0]
    blue_float = blue.astype(np.float32)

    #  isolate the tomato using Otsu threshold
    blurred = cv2.GaussianBlur(blue, (101, 101), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    tomato_mask = np.zeros_like(mask)
    tomato_mask[labels == largest_label] = 255
    tomato_mask = cv2.morphologyEx(tomato_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8), iterations=2)

    # subtract background to extract fluorescent spots
    background = cv2.GaussianBlur(blue_float, (51, 51), 0)
    signal     = np.clip(blue_float - background, 0, None)

    if signal.max() > 0:
        signal = signal / signal.max()

    fluor = (signal > 0.35).astype(np.uint8) * 255
    fluor = cv2.bitwise_and(fluor, fluor, mask=tomato_mask)
    fluor = cv2.morphologyEx(fluor, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # count bacteria blobs using LoG detection
    norm  = fluor.astype(np.float32) / 255.0
    blobs = blob_log(norm, min_sigma=1, max_sigma=4, num_sigma=12, threshold=0.02)
    count = len(blobs)

    output = img.copy()
    for y, x, r in blobs:
        cv2.circle(output, (int(x), int(y)), int(r * 1.5), (0, 255, 0), 2)

    cv2.imwrite("result.jpg", output)
    print(f"Blobs detected: {count}")

    return count


def run_detection():
    show_lcd("Detection", "Please wait...")
    run_excitation_lighting()
    capture_image("capture.jpg")
    count = detect_bacteria("capture.jpg")
    show_lcd("Detected:", str(count))
    sleep(3)
    return count


def run_sterilisation():
    show_lcd("Sterilisation", "UV running...")
    run_uv_sterilisation()
    beep()
    show_lcd("Completed", "")
    sleep(3)


# Press button to begin - runs once then shuts down
show_lcd("System ready", "Press button")

while True:
    if start_btn.is_pressed:
        run_detection()
        run_sterilisation()
        show_lcd("Shutting down", "")
        sleep(2)
        os.system("sudo shutdown now")
        break

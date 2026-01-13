import socket
import numpy as np
from flask import Flask, jsonify
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from collections import deque
import threading

# ---------- FLASK ----------
app = Flask(__name__)

# ---------- WIFI CONFIG ----------
ESP_IP = "10.121.199.88"
PORT = 3333

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((ESP_IP, PORT))
sock.settimeout(0.01)

# ---------- PARAMETERS ----------
MAX_POINTS = 200
SIGMA = 2

POS_THRESHOLD = 5
NEG_THRESHOLD = -5

CUTOFF_FREQ = 2.0
SAMPLING_RATE = 20
SG_WINDOW = 11
SG_POLYORDER = 2
SAMPLE_TOLERANCE = 5

# ---------- FILTER ----------
def lowpass_filter(data, cutoff=CUTOFF_FREQ, fs=SAMPLING_RATE, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# ---------- STORAGE ----------
roll_data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)
stretch_data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)

wave_count = 0
prev_above_threshold = False

sample_count = 0
valley_count = 0
counted_valley_abs = set()

latest_map = "Not available"

buffer = ""

# ---------- MAIN LOOP ----------
def esp_listener():
    global wave_count, prev_above_threshold, buffer
    global sample_count, valley_count, counted_valley_abs
    global latest_map

    while True:
        try:
            data = sock.recv(1024).decode(errors='ignore')
            buffer += data

            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                # ---- MAP ----
                if line.startswith("MAP:"):
                    latest_map = line[4:]

                # ---- DATA ----
                elif line.startswith("DATA:"):
                    try:
                        payload = line[5:]
                        roll, stretch = payload.split(",")

                        roll = float(roll)
                        stretch = int(stretch)

                        roll_data.append(roll)
                        stretch_data.append(stretch)
                        sample_count += 1

                        # ---- ROLL COUNT ----
                        if roll > POS_THRESHOLD and not prev_above_threshold:
                            wave_count += 1
                            prev_above_threshold = True
                        elif roll < NEG_THRESHOLD:
                            prev_above_threshold = False

                    except:
                        pass

        except:
            pass

        # ---------- STRETCH PROCESS ----------
        if len(stretch_data) >= SG_WINDOW:
            raw = np.array(stretch_data)

            lpf = lowpass_filter(raw)
            filtered = savgol_filter(lpf, SG_WINDOW, SG_POLYORDER)

            inverted = -filtered
            valleys, _ = find_peaks(inverted, distance=40, prominence=20)
            crests, _ = find_peaks(filtered, distance=40, prominence=20)

            valid_valleys = []
            for v in valleys:
                left_crests = [c for c in crests if c < v]
                if left_crests:
                    last_crest = max(left_crests)
                    depth = filtered[last_crest] - filtered[v]
                    if depth >= 70:
                        valid_valleys.append(v)

            abs_indices = [
                sample_count - (MAX_POINTS - 1 - v) for v in valid_valleys
            ]

            if sample_count >= MAX_POINTS:
                for abs_idx in abs_indices:
                    if not any(abs(abs_idx - c) <= SAMPLE_TOLERANCE
                               for c in counted_valley_abs):
                        valley_count += 1
                        counted_valley_abs.add(abs_idx)


# ---------- API ----------
@app.route("/data")
def send_data():
    return jsonify({
        "roll_count": wave_count,
        "valley_count": valley_count,
        "map_link": latest_map
    })


# ---------- START ----------
if __name__ == "__main__":
    threading.Thread(target=esp_listener, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)

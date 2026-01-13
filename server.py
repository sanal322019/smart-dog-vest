from flask import Flask, request, jsonify
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks
from collections import deque
import os   # ✅ Added

app = Flask(__name__)

# -------- PARAMETERS --------
MAX_POINTS = 200
POS_THRESHOLD = 5
NEG_THRESHOLD = -5
SG_WINDOW = 11
SG_POLYORDER = 2
SAMPLE_TOLERANCE = 5

roll_data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)
stretch_data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)

wave_count = 0
prev_above = False
sample_count = 0
valley_count = 0
counted = set()
latest_map = "N/A"

# -------- FILTER --------
def lowpass(data, cutoff=2, fs=20, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, data)

# -------- ESP DATA RECEIVE --------
@app.route("/upload", methods=["POST"])
def upload():

    global wave_count, prev_above, sample_count
    global valley_count, counted, latest_map

    d = request.json
    roll = float(d["roll"])
    stretch = int(d["stretch"])

    if d["map"] != "":
        latest_map = d["map"]

    roll_data.append(roll)
    stretch_data.append(stretch)
    sample_count += 1

    # ---- ROLL COUNT ----
    if roll > POS_THRESHOLD and not prev_above:
        wave_count += 1
        prev_above = True
    elif roll < NEG_THRESHOLD:
        prev_above = False

    # ---- STRETCH PROCESS ----
    if len(stretch_data) >= SG_WINDOW:

        raw = np.array(stretch_data)
        filt = savgol_filter(lowpass(raw), SG_WINDOW, SG_POLYORDER)

        inv = -filt
        valleys, _ = find_peaks(inv, distance=40, prominence=20)
        crests, _ = find_peaks(filt, distance=40, prominence=20)

        valid = []
        for v in valleys:
            lc = [c for c in crests if c < v]
            if lc:
                depth = filt[max(lc)] - filt[v]
                if depth >= 70:
                    valid.append(v)

        abs_i = [
            sample_count - (MAX_POINTS - 1 - v)
            for v in valid
        ]

        if sample_count >= MAX_POINTS:
            for i in abs_i:
                if not any(abs(i - c) <= SAMPLE_TOLERANCE for c in counted):
                    valley_count += 1
                    counted.add(i)

    return jsonify({"status": "ok"})

# -------- API FOR WEBSITE --------
@app.route("/data")
def data():
    return jsonify({
        "roll_count": wave_count,
        "valley_count": valley_count,
        "map": latest_map
    })

# -------- WEBSITE --------
@app.route("/")
def home():
    return """
    <h1>🐶 Smart Dog Vest</h1>
    <h2>Roll Count: <span id=r>0</span></h2>
    <h2>Valley Count: <span id=v>0</span></h2>
    <a id=m target=_blank>Open Map</a>

    <script>
    setInterval(()=>{
      fetch('/data').then(r=>r.json()).then(d=>{
        r.innerText=d.roll_count
        v.innerText=d.valley_count
        m.href=d.map
      })
    },1000)
    </script>
    """

# -------- IMPORTANT CLOUD FIX --------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Render port
    app.run(host="0.0.0.0", port=port)

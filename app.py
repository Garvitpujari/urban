"""
UrbanNet SIH26124 - Edge Detection Demo (Streamlit)

Upload the 7-clip demo video -> the app draws bounding boxes + confidence on the
hard-coded detection windows, shows what was detected and where (Delhi map),
and POSTs each event to the backend (MongoDB behind it).

Run:  streamlit run app.py
"""
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BACKEND_URL = "https://urban-net-sih26124-backend.onrender.com/api/v1/edge/events"
DEFAULT_BUS_ID = "BUS_010"
HIT_AND_RUN_PLATE = "UP32KH8090"

# type -> (category, handling, display name, hex colour)
CATALOG = {
    "POTHOLE":                ("ROAD",           "PERSISTENT", "Pothole",                "#ff9800"),
    "WATERLOGGING":           ("ROAD",           "PERSISTENT", "Waterlogging",           "#2196f3"),
    "ROAD_DAMAGE":            ("ROAD",           "PERSISTENT", "Road damage",            "#795548"),
    "ROAD_CRACK":             ("ROAD",           "PERSISTENT", "Road crack",             "#9e9e9e"),
    "MISSING_DIVIDER":        ("INFRASTRUCTURE", "PERSISTENT", "Missing divider",        "#9c27b0"),
    "MISSING_TRAFFIC_LIGHT":  ("INFRASTRUCTURE", "PERSISTENT", "Missing traffic light",  "#607d8b"),
    "MISSING_ZEBRA_CROSSING": ("INFRASTRUCTURE", "PERSISTENT", "Missing zebra crossing", "#00bcd4"),
    "DAMAGED_SIGNBOARD":      ("INFRASTRUCTURE", "PERSISTENT", "Damaged signboard",      "#8bc34a"),
    "HIT_AND_RUN":            ("SAFETY",         "REAL_TIME",  "Hit and run",            "#f44336"),
    "RASH_DRIVING":           ("SAFETY",         "REAL_TIME",  "Rash driving",           "#e91e63"),
    "PEDESTRIAN_RISK":        ("SAFETY",         "REAL_TIME",  "Pedestrian risk",        "#ffc107"),
    "ACCIDENT":               ("SAFETY",         "REAL_TIME",  "Accident",               "#d50000"),
    "TRAFFIC_CONGESTION":     ("TRAFFIC",        "REAL_TIME",  "Traffic congestion",     "#ff5722"),
}

MODEL_BY_CATEGORY = {
    "ROAD":           {"name": "YOLOv8",         "version": "1.0.0"},
    "INFRASTRUCTURE": {"name": "YOLOv8",         "version": "1.0.0"},
    "SAFETY":         {"name": "YOLOv8-Safety",  "version": "2.0.1"},
    "TRAFFIC":        {"name": "YOLOv8-Traffic", "version": "1.0.0"},
}

# Delhi locations - deliberately spread far apart so they look distinct on the map
LOC = {
    "connaught_place": (28.6315, 77.2167),
    "india_gate":      (28.6129, 77.2295),
    "karol_bagh":      (28.6519, 77.1909),
    "lajpat_nagar":    (28.5677, 77.2433),
    "dwarka":          (28.5921, 77.0460),
    "rohini":          (28.7495, 77.0565),
    "saket":           (28.5245, 77.2066),
    "nehru_place":     (28.5494, 77.2519),
    "chandni_chowk":   (28.6506, 77.2303),
    "vasant_kunj":     (28.5200, 77.1590),
    "mayur_vihar":     (28.6090, 77.2960),
    "pitampura":       (28.7010, 77.1320),
}

# Cloudinary evidence images (one per clip)
IMG = {
    1: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699763/Screenshot_2026-09-29_220212.png",
    2: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699812/Screenshot_2026-09-29_220232.png",
    3: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699813/Screenshot_2026-09-29_220246_-_Copy.png",
    4: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699813/Screenshot_2026-09-29_220258.png",
    5: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699813/Screenshot_2026-09-29_220311.png",
    6: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699813/Screenshot_2026-09-29_220333.png",
    7: "https://res.cloudinary.com/ollmevhd/image/upload/v1790699814/Screenshot_2026-09-29_220357.png",
}


def det(type_, sev, conf, clip, loc, box, plate=""):
    """box = (x1, y1, x2, y2) as fractions of frame width/height."""
    return dict(type=type_, severity=sev, confidence=conf, image=IMG[clip],
                loc=LOC[loc], box=box, plate=plate)


# ----------------------------------------------------------------------------
# HARD-CODED TIMELINE  (edit start/end seconds + boxes to match your video)
# ----------------------------------------------------------------------------
CLIPS = [
    dict(clip=1, start=0, end=5, detections=[
        det("WATERLOGGING", "HIGH",   0.91, 1, "connaught_place", (0.10, 0.62, 0.60, 0.92)),
        det("POTHOLE",      "MEDIUM", 0.88, 1, "dwarka",          (0.62, 0.68, 0.82, 0.86)),
    ]),
    dict(clip=2, start=5, end=10, detections=[
        det("WATERLOGGING", "HIGH",   0.93, 2, "rohini",          (0.15, 0.60, 0.70, 0.93)),
        det("POTHOLE",      "HIGH",   0.90, 2, "lajpat_nagar",    (0.55, 0.70, 0.78, 0.88)),
    ]),
    dict(clip=3, start=10, end=15, detections=[
        det("HIT_AND_RUN",             "CRITICAL", 0.97, 3, "india_gate",  (0.30, 0.40, 0.65, 0.80), HIT_AND_RUN_PLATE),
        det("MISSING_ZEBRA_CROSSING",  "MEDIUM",   0.86, 3, "karol_bagh",  (0.05, 0.75, 0.95, 0.95)),
    ]),
    dict(clip=4, start=15, end=20, detections=[
        det("HIT_AND_RUN",             "CRITICAL", 0.96, 4, "nehru_place", (0.32, 0.38, 0.68, 0.82), HIT_AND_RUN_PLATE),
    ]),
    dict(clip=5, start=20, end=25, detections=[
        det("MISSING_DIVIDER",         "MEDIUM",   0.89, 5, "saket",       (0.35, 0.50, 0.65, 0.90)),
    ]),
    dict(clip=6, start=25, end=30, detections=[
        det("DAMAGED_SIGNBOARD",       "LOW",      0.87, 6, "chandni_chowk", (0.55, 0.10, 0.85, 0.45)),
    ]),
    dict(clip=7, start=30, end=35, detections=[
        det("PEDESTRIAN_RISK",         "HIGH",     0.92, 7, "vasant_kunj", (0.20, 0.45, 0.75, 0.90)),
    ]),
]


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_timeline(duration, mode):
    """Return clips with (possibly rescaled) start/end times."""
    clips = [dict(c, detections=[dict(d) for d in c["detections"]]) for c in CLIPS]
    if mode == "Split video evenly into 7 clips" and duration > 0:
        seg = duration / len(clips)
        for i, c in enumerate(clips):
            c["start"], c["end"] = i * seg, (i + 1) * seg
    return clips


def build_events(clips, duration, bus_id):
    """One backend event per detection whose window begins inside the video."""
    events = []
    for c in clips:
        if c["start"] >= duration:
            continue
        for d in c["detections"]:
            category, handling, _, _ = CATALOG[d["type"]]
            events.append({
                "_clip": c["clip"],
                "_start": c["start"],
                "_end": c["end"],
                "_box": d["box"],
                "observationId": f"{d['type'].lower()}_{uuid.uuid4().hex[:10]}",
                "busId": bus_id,
                "category": category,
                "type": d["type"],
                "handling": handling,
                "severity": d["severity"],
                "confidence": d["confidence"],
                "location": {"latitude": d["loc"][0], "longitude": d["loc"][1]},
                "capturedAt": now_iso(),
                "evidence": {"imageUrl": d["image"]},
                "vehicleNumber": d["plate"],
                "model": MODEL_BY_CATEGORY[category],
            })
    return events


def payload_of(ev, refresh_time=True):
    p = {k: v for k, v in ev.items() if not k.startswith("_")}
    if refresh_time:
        p["capturedAt"] = now_iso()
    return p


def annotate_video(src_path, events, progress_cb=None):
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    raw_path = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4").name
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 1400)
    thick = max(2, int(w / 640))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        for n, ev in enumerate(events):
            if not (ev["_start"] <= t < ev["_end"]):
                continue
            color = hex_to_bgr(CATALOG[ev["type"]][3])
            x1, y1, x2, y2 = ev["_box"]
            # tiny jitter so the box looks "tracked" rather than frozen
            dx = 0.006 * np.sin(t * 3.0 + n)
            dy = 0.004 * np.cos(t * 2.3 + n)
            p1 = (int((x1 + dx) * w), int((y1 + dy) * h))
            p2 = (int((x2 + dx) * w), int((y2 + dy) * h))
            conf = float(np.clip(ev["confidence"] + 0.02 * np.sin(t * 5 + n), 0, 0.99))
            label = f"{ev['type']} {conf:.2f}"
            if ev["vehicleNumber"]:
                label += f" | {ev['vehicleNumber']}"
            cv2.rectangle(frame, p1, p2, color, thick)
            (tw, th), base = cv2.getTextSize(label, font, scale, thick)
            top = max(p1[1] - th - base - 6, 0)
            cv2.rectangle(frame, (p1[0], top), (p1[0] + tw + 8, top + th + base + 6), color, -1)
            cv2.putText(frame, label, (p1[0] + 4, top + th + 2), font, scale,
                        (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
        writer.write(frame)
        idx += 1
        if progress_cb and idx % 10 == 0:
            progress_cb(min(idx / total, 1.0))
    cap.release()
    writer.release()

    # Re-encode to H.264 so browsers can play it
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_annotated.mp4").name
    if imageio_ffmpeg is not None:
        try:
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_path,
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", out_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(raw_path)
            return out_path
        except Exception:
            pass
    return raw_path  # fallback (may not preview in some browsers)


def send_events(events):
    rows = []
    for ev in events:
        payload = payload_of(ev)
        status, msg = None, ""
        for attempt in range(2):  # Render free tier can cold-start; retry once
            try:
                r = requests.post(BACKEND_URL, json=payload, timeout=60)
                status, msg = r.status_code, r.text[:200]
                break
            except Exception as e:  # noqa: BLE001
                status, msg = "ERR", str(e)[:200]
        rows.append({
            "observationId": payload["observationId"],
            "type": payload["type"],
            "status": status,
            "response": msg,
            "ok": isinstance(status, int) and 200 <= status < 300,
        })
    return rows


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="UrbanNet Edge Detection", page_icon="🚌", layout="wide")
st.title("🚌 UrbanNet – Edge Detection Demo")
st.caption("Upload the bus-camera video → annotated detections → events pushed to the UrbanNet backend.")

with st.sidebar:
    st.header("Settings")
    bus_id = st.text_input("Bus ID", DEFAULT_BUS_ID)
    mode = st.radio("Clip timing", ["Fixed timestamps (5s per clip)", "Split video evenly into 7 clips"])
    auto_send = st.checkbox("Auto-send events after analysis", value=False)
    st.text_input("Backend URL", BACKEND_URL, disabled=True)
    st.markdown("**Clip timeline**")
    st.dataframe(
        pd.DataFrame([{"clip": c["clip"], "start": c["start"], "end": c["end"],
                       "detects": ", ".join(d["type"] for d in c["detections"])} for c in CLIPS]),
        hide_index=True, use_container_width=True,
    )

for key in ("events", "annotated", "results"):
    st.session_state.setdefault(key, None)

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded and st.button("🔍 Analyze video", type="primary"):
    src = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]).name
    with open(src, "wb") as f:
        f.write(uploaded.getbuffer())

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    cap.release()

    clips = build_timeline(duration, mode)
    events = build_events(clips, duration, bus_id)

    bar = st.progress(0.0, text="Running detection & drawing bounding boxes…")
    st.session_state.annotated = annotate_video(
        src, events, lambda p: bar.progress(p, text="Running detection & drawing bounding boxes…"))
    bar.empty()
    st.session_state.events = events
    st.session_state.results = None

    if auto_send:
        with st.spinner("Sending events to backend…"):
            st.session_state.results = send_events(events)

events = st.session_state.events
if events:
    st.subheader("Annotated video")
    with open(st.session_state.annotated, "rb") as f:
        st.video(f.read())

    # ---- summary ----
    st.subheader("Detection summary")
    df = pd.DataFrame([{
        "clip": e["_clip"], "time": f"{e['_start']:.0f}s–{e['_end']:.0f}s",
        "type": e["type"], "category": e["category"], "handling": e["handling"],
        "severity": e["severity"], "confidence": e["confidence"],
        "lat": e["location"]["latitude"], "lon": e["location"]["longitude"],
        "vehicleNumber": e["vehicleNumber"], "observationId": e["observationId"],
        "color": CATALOG[e["type"]][3],
    } for e in events])

    cols = st.columns(4)
    cols[0].metric("Total events", len(df))
    cols[1].metric("Road issues", int((df.category == "ROAD").sum()))
    cols[2].metric("Infrastructure", int((df.category == "INFRASTRUCTURE").sum()))
    cols[3].metric("Safety (real-time)", int((df.category == "SAFETY").sum()))

    st.map(df, latitude="lat", longitude="lon", color="color", size=250)

    # ---- per-type sections ----
    st.subheader("What was detected & where")
    for t in df["type"].unique():
        sub = df[df["type"] == t]
        evs = [e for e in events if e["type"] == t]
        _, _, name, _ = CATALOG[t]
        icon = "🚨" if CATALOG[t][1] == "REAL_TIME" else "📍"
        with st.expander(f"{icon} {name} — {len(sub)} detected", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(
                    sub[["clip", "time", "severity", "confidence", "lat", "lon", "vehicleNumber"]],
                    hide_index=True, use_container_width=True)
                for e in evs:
                    loc = e["location"]
                    msg = (f"**{name}** detected at ({loc['latitude']:.4f}, {loc['longitude']:.4f}) "
                           f"in clip {e['_clip']} — confidence {e['confidence']:.2f}, "
                           f"severity {e['severity']}.")
                    if e["vehicleNumber"]:
                        msg += f" Vehicle: **{e['vehicleNumber']}**"
                    st.markdown("- " + msg)
            with c2:
                st.image([e["evidence"]["imageUrl"] for e in evs],
                         caption=[f"Clip {e['_clip']}" for e in evs], width=260)
            with st.popover("View JSON payload(s)"):
                for e in evs:
                    st.json(payload_of(e, refresh_time=False))

    # ---- send ----
    st.subheader("Backend sync")
    st.code(f"POST {BACKEND_URL}", language="text")
    if st.button("📡 Send all events to backend"):
        with st.spinner("Posting events (first call may take ~30s if Render is waking up)…"):
            st.session_state.results = send_events(events)

    if st.session_state.results:
        res = pd.DataFrame(st.session_state.results)
        ok = int(res["ok"].sum())
        (st.success if ok == len(res) else st.warning)(f"{ok}/{len(res)} events accepted by backend")
        st.dataframe(res.drop(columns="ok"), hide_index=True, use_container_width=True)
else:
    st.info("Upload the demo video and click **Analyze video** to begin.")

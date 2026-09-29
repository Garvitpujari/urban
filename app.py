"""
UrbanNet SIH26124 - Edge Detection Demo (Streamlit)

Upload the 7-clip demo video -> the app lists what was detected and where,
and POSTs each event (with lat/long + dummy Delhi address) to the backend.

Run:  streamlit run app.py
"""
import os
import tempfile
import uuid
from datetime import datetime, timezone

import cv2
import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
BACKEND_URL = "https://urban-net-sih26124-backend.onrender.com/api/v1/edge/events"
BUS_ID = "BUS_010"
HIT_AND_RUN_PLATE = "UP32KH8090"

# type -> (category, handling, display name)
CATALOG = {
    "POTHOLE":                ("ROAD",           "PERSISTENT", "Pothole"),
    "WATERLOGGING":           ("ROAD",           "PERSISTENT", "Waterlogging"),
    "ROAD_DAMAGE":            ("ROAD",           "PERSISTENT", "Road damage"),
    "ROAD_CRACK":             ("ROAD",           "PERSISTENT", "Road crack"),
    "MISSING_DIVIDER":        ("INFRASTRUCTURE", "PERSISTENT", "Missing divider"),
    "MISSING_TRAFFIC_LIGHT":  ("INFRASTRUCTURE", "PERSISTENT", "Missing traffic light"),
    "MISSING_ZEBRA_CROSSING": ("INFRASTRUCTURE", "PERSISTENT", "Missing zebra crossing"),
    "DAMAGED_SIGNBOARD":      ("INFRASTRUCTURE", "PERSISTENT", "Damaged signboard"),
    "HIT_AND_RUN":            ("SAFETY",         "REAL_TIME",  "Hit and run"),
    "RASH_DRIVING":           ("SAFETY",         "REAL_TIME",  "Rash driving"),
    "PEDESTRIAN_RISK":        ("SAFETY",         "REAL_TIME",  "Pedestrian risk"),
    "ACCIDENT":               ("SAFETY",         "REAL_TIME",  "Accident"),
    "TRAFFIC_CONGESTION":     ("TRAFFIC",        "REAL_TIME",  "Traffic congestion"),
}

MODEL_BY_CATEGORY = {
    "ROAD":           {"name": "YOLOv8",         "version": "1.0.0"},
    "INFRASTRUCTURE": {"name": "YOLOv8",         "version": "1.0.0"},
    "SAFETY":         {"name": "YOLOv8-Safety",  "version": "2.0.1"},
    "TRAFFIC":        {"name": "YOLOv8-Traffic", "version": "1.0.0"},
}

# Delhi locations (lat, lon, address) - coordinates match the named landmarks
LOC = {
    "shahdara":      (28.6735, 77.2890, "Shahdara Flyover, GT Road, Shahdara, Delhi"),
    "rohini":        (28.7209, 77.1073, "Outer Ring Road, near Rithala Metro Station, Rohini, Delhi"),
    "karol_bagh":    (28.6435, 77.1885, "Pusa Road, near Karol Bagh Metro Station, Delhi"),
    "nehru_place":   (28.5494, 77.2519, "Nehru Place Flyover, Outer Ring Road, Delhi"),
    "saket":         (28.5284, 77.2192, "Press Enclave Marg, near Saket District Centre, Delhi"),
    "chandni_chowk": (28.6562, 77.2312, "Chandni Chowk Road, near Town Hall, Old Delhi"),
    "vasant_kunj":   (28.5300, 77.1560, "Nelson Mandela Marg, near Sector D, Vasant Kunj, Delhi"),
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


def det(type_, sev, conf, clip, loc, plate=""):
    return dict(type=type_, severity=sev, confidence=conf, image=IMG[clip],
                loc=LOC[loc], plate=plate)


# ----------------------------------------------------------------------------
# HARD-CODED TIMELINE (5 seconds per clip)
# ----------------------------------------------------------------------------
CLIPS = [
    dict(clip=1, start=0, end=5, detections=[
        det("WATERLOGGING", "HIGH",   0.91, 1, "shahdara"),
        det("POTHOLE",      "MEDIUM", 0.88, 1, "shahdara"),
    ]),
    dict(clip=2, start=5, end=10, detections=[
        det("WATERLOGGING", "HIGH",   0.93, 2, "rohini"),
    ]),
    dict(clip=3, start=10, end=15, detections=[
        det("MISSING_ZEBRA_CROSSING", "MEDIUM",   0.86, 3, "karol_bagh"),
    ]),
    dict(clip=4, start=15, end=20, detections=[
        det("HIT_AND_RUN",            "CRITICAL", 0.96, 4, "nehru_place", HIT_AND_RUN_PLATE),
    ]),
    dict(clip=5, start=20, end=25, detections=[
        det("MISSING_DIVIDER",        "MEDIUM",   0.89, 5, "saket"),
    ]),
    dict(clip=6, start=25, end=30, detections=[
        det("DAMAGED_SIGNBOARD",      "LOW",      0.87, 6, "chandni_chowk"),
    ]),
    dict(clip=7, start=30, end=35, detections=[
        det("PEDESTRIAN_RISK",        "HIGH",     0.92, 7, "vasant_kunj"),
    ]),
]


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def video_duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frames / fps


def build_events(duration):
    """One backend event per detection whose clip begins inside the video."""
    events = []
    for c in CLIPS:
        if c["start"] >= duration:
            continue
        for d in c["detections"]:
            category, handling, _ = CATALOG[d["type"]]
            events.append({
                "_clip": c["clip"],
                "_start": c["start"],
                "_end": c["end"],
                "observationId": f"{d['type'].lower()}_{uuid.uuid4().hex[:10]}",
                "busId": BUS_ID,
                "category": category,
                "type": d["type"],
                "handling": handling,
                "severity": d["severity"],
                "confidence": d["confidence"],
                "location": {
                    "latitude": d["loc"][0],
                    "longitude": d["loc"][1],
                    "address": d["loc"][2],
                },
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


def send_events(events):
    rows = []
    for ev in events:
        payload = payload_of(ev)
        status, msg = None, ""
        for _ in range(2):  # Render free tier can cold-start; retry once
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
st.caption("Upload the bus-camera video → detections listed → events pushed to the UrbanNet backend.")

st.session_state.setdefault("events", None)
st.session_state.setdefault("results", None)

uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    st.video(uploaded)

if uploaded and st.button("🔍 Analyze video", type="primary"):
    src = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]).name
    with open(src, "wb") as f:
        f.write(uploaded.getbuffer())
    with st.spinner("Running detection…"):
        st.session_state.events = build_events(video_duration(src))
    st.session_state.results = None

events = st.session_state.events
if events:
    st.subheader("Detection summary")
    df = pd.DataFrame([{
        "clip": e["_clip"], "time": f"{e['_start']:.0f}s–{e['_end']:.0f}s",
        "type": e["type"], "category": e["category"], "handling": e["handling"],
        "severity": e["severity"], "confidence": e["confidence"],
        "address": e["location"]["address"],
        "lat": e["location"]["latitude"], "lon": e["location"]["longitude"],
        "vehicleNumber": e["vehicleNumber"], "observationId": e["observationId"],
    } for e in events])

    cols = st.columns(4)
    cols[0].metric("Total events", len(df))
    cols[1].metric("Road issues", int((df.category == "ROAD").sum()))
    cols[2].metric("Infrastructure", int((df.category == "INFRASTRUCTURE").sum()))
    cols[3].metric("Safety (real-time)", int((df.category == "SAFETY").sum()))

    st.subheader("What was detected & where")
    for t in df["type"].unique():
        sub = df[df["type"] == t]
        evs = [e for e in events if e["type"] == t]
        name = CATALOG[t][2]
        icon = "🚨" if CATALOG[t][1] == "REAL_TIME" else "📍"
        with st.expander(f"{icon} {name} — {len(sub)} detected", expanded=True):
            c1, c2 = st.columns([1, 1])
            with c1:
                show_cols = ["severity", "confidence", "address", "lat", "lon"]
                if t == "HIT_AND_RUN":
                    show_cols.append("vehicleNumber")
                st.dataframe(sub[show_cols], hide_index=True, use_container_width=True)
                for e in evs:
                    loc = e["location"]
                    msg = (f"**{name}** detected at **{loc['address']}** "
                           f"({loc['latitude']:.4f}, {loc['longitude']:.4f}) "
                           f"— confidence {e['confidence']:.2f}, "
                           f"severity {e['severity']}.")
                    if e["vehicleNumber"]:
                        msg += f" Vehicle: **{e['vehicleNumber']}**"
                    st.markdown("- " + msg)
            with c2:
                st.image([e["evidence"]["imageUrl"] for e in evs],
                         caption=[e["location"]["address"] for e in evs], width=260)
            with st.popover("View JSON payload(s)"):
                for e in evs:
                    st.json(payload_of(e, refresh_time=False))

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

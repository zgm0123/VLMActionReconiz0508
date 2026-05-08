import os
import sys
import json
import subprocess
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
app.jinja_options = {
    'variable_start_string': '[[',
    'variable_end_string': ']]',
    'comment_start_string': '[#',
    'comment_end_string': '#]',
}
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_VIDEO_DIR = os.path.join(BASE_DIR, "sample_video")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "minicpm_v45")

DEFAULT_LABELS = [
    "追球",
    "拉球",
    "犬咬",
]


def get_videos():
    videos = []
    if not os.path.exists(SAMPLE_VIDEO_DIR):
        return videos
    files = set(os.listdir(SAMPLE_VIDEO_DIR))
    for f in sorted(files):
        if not f.lower().endswith('.mp4'):
            continue
        if f.endswith('_h264.mp4'):
            original = f[:-len('_h264.mp4')] + '.mp4'
            if original in files:
                continue
            name = f[:-len('_h264.mp4')]
            display_file = original
        else:
            name = os.path.splitext(f)[0]
            display_file = f
        result_path = os.path.join(RESULTS_DIR, f"{name}_v45.json")
        inferred = os.path.exists(result_path)
        videos.append({
            "name": f"sample_video/{display_file}",
            "video_name": name,
            "inferred": inferred
        })
    return videos


def get_video_results(video_name):
    result_path = os.path.join(RESULTS_DIR, f"{video_name}_v45.json")
    if not os.path.exists(result_path):
        return [], ""

    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    actions = []
    segment_results = data.get("segment_results", [])
    for i, seg in enumerate(segment_results):
        for action, times in seg.items():
            if isinstance(times, dict) and times.get("start") is not None:
                start = times["start"]
                end = times["end"]
                duration = round(end - start, 1) if end is not None else 0
                actions.append({
                    "action": action,
                    "status": "detected",
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                    "segment_id": i + 1
                })
    raw_outputs = data.get("raw_outputs", "")
    return actions, raw_outputs


@app.route("/")
def index():
    return render_template("index_minicpm.html")


@app.route("/api/videos")
def api_videos():
    return jsonify(get_videos())


@app.route("/api/labels")
def api_labels():
    return jsonify(DEFAULT_LABELS)


@app.route("/api/results/<video_name>")
def api_results(video_name):
    actions, raw_outputs = get_video_results(video_name)
    return jsonify({"actions": actions, "raw_outputs": raw_outputs})


@app.route("/api/infer", methods=["POST"])
def api_infer():
    data = request.json
    video = data.get("video", "")
    labels = data.get("labels", [])
    if not video or not labels:
        return jsonify({"error": "Missing video or labels"}), 400

    video_path = os.path.join(BASE_DIR, video)
    name, ext = os.path.splitext(video_path)
    h264_path = f"{name}_h264{ext}"
    if os.path.exists(h264_path):
        video_path = h264_path
    labels_str = ",".join(labels)
    output_name = os.path.splitext(os.path.basename(video))[0]
    output_path = os.path.join(RESULTS_DIR, f"{output_name}_v45.json")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cmd = [
        sys.executable, "minicpm_native/infer_video.py",
        "--video", video_path,
        "--actions", labels_str,
        "--detailed",
        "--segment-duration", "5",
        "--output", output_path,
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=3600, env=env)
        return jsonify({"success": True, "stdout": result.stdout, "stderr": result.stderr})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/video/<path:filename>")
def serve_video(filename):
    name, ext = os.path.splitext(filename)
    h264_path = os.path.join(BASE_DIR, f"{name}_h264{ext}")
    if os.path.exists(h264_path):
        return send_from_directory(BASE_DIR, f"{name}_h264{ext}")
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)

from flask import Flask, jsonify, send_file
from flask_cors import CORS

from backend.state import state
from backend.runner import start_background_run

app = Flask(__name__)
CORS(app)


@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "status": state["status"],
        "task": state["task"],
        "prompt": state["prompt"]
    })


@app.route("/run", methods=["POST"])
def run_task():
    if state["status"] == "running":
        return jsonify({"message": "Already running"}), 400

    start_background_run()
    return jsonify({"message": "Started"})


@app.route("/frame", methods=["GET"])
def get_frame():
    try:
        return send_file(state["frame_path"], mimetype="image/png")
    except Exception:
        return jsonify({"error": "No frame yet"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


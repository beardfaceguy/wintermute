from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)

MAP_LIST_HOST = os.getenv("MAP_LIST_HOST", "0.0.0.0")
MAP_LIST_PORT = int(os.getenv("MAP_LIST_PORT", "6011"))

@app.route('/list', methods=['POST'])
def list_directory():
    data = request.get_json()
    path = data.get("test")

    if not path or not isinstance(path, str):
        return jsonify({"error": "Invalid or missing path"}), 400

    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404

    try:
        output = subprocess.check_output(["ls", path], stderr=subprocess.STDOUT, text=True)
        files = [f for f in output.strip().split("\n") if f]
        return jsonify({"path": path, "contents": files})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ls failed", "details": e.output}), 500

if __name__ == '__main__':
    app.run(host=MAP_LIST_HOST, port=MAP_LIST_PORT)


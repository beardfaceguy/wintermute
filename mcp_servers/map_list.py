from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)

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
        files = output.strip().split("\n")
        return jsonify({"path": path, "contents": files})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "ls failed", "details": e.output}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6011)


from flask import Flask, request, jsonify, send_file
from image_generator import generate_image
import os

app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    """Generate image using external function."""
    try:
        data = request.get_json()
        prompt = data.get("prompt", "A beautiful landscape")

        # Generate image
        image_path = generate_image(prompt)
        filename = os.path.basename(image_path)

        return jsonify({
            "status": "success",
            "filename": filename,
            "url": f"/static/{filename}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/view/<filename>")
def view_image(filename):
    """Serve image from static folder."""
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/png")
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
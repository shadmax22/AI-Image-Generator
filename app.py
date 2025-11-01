from flask import Flask, request, jsonify, send_file, send_from_directory
from image_generator import generate_image
from PIL import Image
import os
import io

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Serve the main index.html file for the UI."""
    # This assumes index.html is in the same directory as app.py
    return send_from_directory(os.path.dirname(__file__), "index.html")
# -----------------

@app.route("/generate", methods=["POST"])
def generate():
    """
    POST /generate
    Content-Type: multipart/form-data
    Fields:
      - prompt: text (required)
      - image: file (optional)
    """
    try:
        prompt = request.form.get("prompt", "")
        if not prompt:
            return jsonify({"status": "error", "message": "Prompt is required"}), 400

        uploaded_file = request.files.get("image")
        init_image = None

        if uploaded_file:
            print(f"📥 Received image: {uploaded_file.filename}")
            image_bytes = uploaded_file.read()
            init_image = Image.open(io.BytesIO(image_bytes))

        result_path = generate_image(prompt, init_image)
        filename = os.path.basename(result_path)

        return jsonify({
            "status": "success",
            "filename": filename,
            "url": f"/static/{filename}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/images", methods=["GET"])
def list_images():
    """
    GET /images
    Lists all generated images in /static directory.
    """
    static_dir = "static"
    if not os.path.exists(static_dir):
        return jsonify({"images": []})

    # Get all files that end with .png, .jpg, .jpeg
    files = [
        f for f in os.listdir(static_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    image_urls = [f"/static/{f}" for f in sorted(files)]

    return jsonify({
        "count": len(image_urls),
        "images": image_urls
    })


@app.route("/view/<filename>")
def view_image(filename):
    """Serve a specific image from static."""
    file_path = os.path.join("static", filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/png")
    return jsonify({"error": "File not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
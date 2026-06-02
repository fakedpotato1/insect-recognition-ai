import base64
import logging

from flask import Flask
from flask import jsonify
from flask import request

from flask_cors import CORS

from config import Config
from model_service import predict


app = Flask(__name__)
app.config.from_object(Config)

CORS(app)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    }), 200


@app.route("/detect", methods=["POST"])
def detect():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is required"
            }), 400

        image = data.get("image")

        if not image:
            return jsonify({
                "error": "Image field is required"
            }), 400

        try:
            base64.b64decode(
                image,
                validate=True
            )
        except Exception:
            return jsonify({
                "error": "Invalid base64 image"
            }), 400

        result = predict(image)
        return jsonify(result), 200

    except NotImplementedError as e:

        logging.error(str(e))

        return jsonify({
            "error": str(e)
        }), 501

    except Exception as e:

        logging.exception(e)

        return jsonify({
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=app.config["PORT"],
        debug=True
    )
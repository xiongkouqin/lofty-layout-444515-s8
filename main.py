from flask import Flask, request, jsonify
from model import get_emotion_from_bytes  # Assuming the model file is saved as model.py

app = Flask(__name__)

@app.route('/')
def home():
    return "Emotion Detection API is running. Send a POST request to /predict to detect emotions."


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image:
        try:
            image_bytes = image.read()
            emotion, scores = get_emotion_from_bytes(image_bytes)
            # 将 scores 转换为标准 Python 类型
            scores = list(map(float, scores))
            return jsonify({'emotion': emotion, 'scores': scores})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid image file'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6789)

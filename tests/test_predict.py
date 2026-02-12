import tempfile
import os
from PIL import Image
from src.inference.predict import predict_image


def test_prediction_output():

    with tempfile.TemporaryDirectory() as tmpdir:

        image_path = os.path.join(tmpdir, "test.jpg")

        img = Image.new("RGB", (224, 224))
        img.save(image_path)

        result = predict_image(image_path)

        assert "predicted_class" in result
        assert "probabilities" in result

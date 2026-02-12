from src.inference.predict import predict_image
import os


def test_prediction_output():

    # Use any small image from dataset
    sample_image = None

    for root, dirs, files in os.walk("data/raw"):
        for file in files:
            if file.endswith(".jpg"):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break

    result = predict_image(sample_image)

    assert "predicted_class" in result
    assert "probabilities" in result

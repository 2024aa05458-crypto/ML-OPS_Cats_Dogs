import os
import tempfile
from PIL import Image
from src.data.preprocess import get_data_loaders


def test_dataloaders_creation():

    with tempfile.TemporaryDirectory() as tmpdir:

        # Create fake class folders
        os.makedirs(os.path.join(tmpdir, "cats"))
        os.makedirs(os.path.join(tmpdir, "dogs"))

        # Create small dummy images
        img = Image.new("RGB", (224, 224))

        img.save(os.path.join(tmpdir, "cats", "cat1.jpg"))
        img.save(os.path.join(tmpdir, "dogs", "dog1.jpg"))

        train_loader, val_loader, test_loader = get_data_loaders(tmpdir, batch_size=1)

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

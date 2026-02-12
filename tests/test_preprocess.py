from src.data.preprocess import get_data_loaders


def test_dataloaders_creation():
    train_loader, val_loader, test_loader = get_data_loaders("data/raw", batch_size=4)

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

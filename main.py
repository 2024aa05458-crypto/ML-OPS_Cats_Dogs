from src.data.preprocess import get_data_loaders

train_loader, val_loader, test_loader = get_data_loaders("data/raw")

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Test batches:", len(test_loader))

from autoencoder_model import FileAutoEncoder, ByteLogitsHead, AutoEncoderLoss
from dataset import FileAutoEncoderDataset

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


NUM_EPOCHES = 20
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def test_model(model, criterion, loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress = tqdm(loader, desc="Testing", leave=False)
        for input, target, mask in progress:
            input = input.to(device)
            target = target.to(device)
            mask = mask.to(device)

            output = model(input)
            loss = criterion(output, target, mask)

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def train_epoch(model, optimizer, criterion, loader, epoch):
    model.train()
    total_loss = 0

    progress = tqdm(loader, desc=f"Training Epoch {epoch}", leave=False)

    for input, target, mask in progress:
        input = input.to(device)
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        output = model(input)
        loss = criterion(output, target, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def main():
    train_dataset = FileAutoEncoderDataset()
    test_dataset = FileAutoEncoderDataset(is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )

    model = FileAutoEncoder(
        16,
        [
            [16, 32, 11, 1, 5],
            [32, 64, 5, 1, 2],
            [64, 128, 3, 1, 1],
            [128, 256, 3, 2, 1],
        ],
        [
            [512, 128, 3, 2, 1, 1],  # kernel=4 для компенсации: (5119-1)*2 + 4 = 10240
            [256, 64, 3, 1, 1, 0],
            [128, 32, 5, 1, 2, 0],
            [64, 16, 11, 1, 5, 0]
        ],
        ByteLogitsHead(16),
        is_gelu=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = AutoEncoderLoss()

    for epoch in range(NUM_EPOCHES):
        train_loss = train_epoch(
            model, optimizer, criterion, train_loader, epoch
        )

        test_loss = test_model(
            model, criterion, test_loader
        )

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.6f}, "
            f"Test Loss = {test_loss:.6f}"
        )


if __name__ == "__main__":
    main()

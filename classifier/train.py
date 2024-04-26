import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os


def save_model(model, optimizer, epoch, loss, filepath="model_checkpoint.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, filepath)


def train_model(model, criterion, optimizer, num_epochs=25, dataloaders=None, dataset_sizes=None, device="cuda:0"):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # モデルの保存
            if phase == "valid" and epoch_loss < best_loss:
                best_loss = epoch_loss
                file_path = f"results/resnet50/model_epoch_{epoch}.pth"
                save_model(model, optimizer, epoch, epoch_loss, filepath=file_path)
                print(f"Model saved: model_epoch_{epoch}.pth")

    return model


def evaluate_model(model, dataloader, criterion, device="cuda:0"):
    model.eval()  # モデルを評価モードに設定
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # 勾配計算を無効化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)

    total_loss = running_loss / total_samples
    total_accuracy = running_corrects.double() / total_samples

    print(f'Loss: {total_loss:.4f} Accuracy: {total_accuracy:.4f}')


def main():
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }
    data_dir = "/home/sugarl/VScode/team18/classifier/cook-ai-dataset"
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "valid", "test"]
    }
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ["train", "valid", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid", "test"]}
    class_names = image_datasets["train"].classes
    print(f'Number of classes: {len(class_names)}')
    print(f'Dataset sizes: {dataset_sizes}')
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    model = train_model(model, criterion, optimizer, num_epochs=50, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=device)
    evaluate_model(model, dataloaders["test"], criterion, device=device)


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

# Подготовка данных
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = ImageFolder(root='C:\LinearNN\Data', transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# Определение архитектуры нейронной сети
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(16, 3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Создание модели и определение функции потерь и оптимизатора
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 100

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)}')

print('Обучение завершено.')

test_dataset = ImageFolder(root='C:\LinearNN\Test', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

dataiter = iter(test_loader)
images, labels = next(dataiter)

# Визуализация предсказаний на тестовой выборке
model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

class_names = test_dataset.classes

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(np.transpose(images[i], (1, 2, 0)))
    ax.set_title(f'Предсказание: {class_names[predicted[i]]}\nИстинный класс: {class_names[labels[i]]}')
    ax.axis('off')

plt.show()

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Load CIFAR-10 with transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# CIFAR-10 class labels
classes = dataset.classes

# Get one batch
images, labels = next(iter(loader))

# Unnormalize if needed (since training normalize isn't applied here, it's okay)
def show_images(imgs, labels):
    fig, axs = plt.subplots(1, len(imgs), figsize=(15, 3))
    for i, (img, label) in enumerate(zip(imgs, labels)):
        img = img.permute(1, 2, 0)  # C, H, W → H, W, C
        axs[i].imshow(img)
        axs[i].set_title(classes[label])
        axs[i].axis('off')
    plt.tight_layout()
    plt.savefig('cifar10_sample.png')
    print("Plot saved as 'cifar10_sample.png'")

show_images(images, labels)
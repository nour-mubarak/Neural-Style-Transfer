import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd


# Define the text encoder network
class TextEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.tanh(x)
        return x


# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, image_paths, text_data, transform=None):
        self.image_paths = image_paths
        self.text_data = text_data
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        text = self.text_data[idx]
        if self.transform:
            img = self.transform(img)
        return img, text


# Define data transformation
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load skin cancer image paths and textual descriptions
image_paths = [
    "/home/nour/Coding-projects/GANs dark skin/challenge_data/jpeg/train"
]  # List of image file paths
text_data = ["challenge_data/train.csv"]  # List of textual descriptions

# Define batch size
batch_size = 32
text_dim = 100  # Define the value of text_dim
latent_dim = 256  # Define the value of latent_dim
image_dim = 3  # Define the value of image_dim

# Create dataset and dataloader
dataset = CustomDataset(image_paths, text_data, transform=transform)
lr = 0.001  # Define the learning rate
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the text encoder, generator, and discriminator
text_encoder = TextEncoder(input_dim=text_dim, hidden_dim=256, output_dim=latent_dim)
generator = Generator(latent_dim=latent_dim, output_dim=image_dim)
discriminator = models.resnet18(pretrained=True)  # Example discriminator

# Define optimizers
optimizer_text_encoder = torch.optim.Adam(text_encoder.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Define loss functions
adversarial_loss = nn.BCELoss()
image_reconstruction_loss = nn.MSELoss()

# Training loop
num_epochs = 10  # Define the number of epochs
for epoch in range(num_epochs):
    for i, (images, texts) in enumerate(dataloader):
        # Train the text encoder
        optimizer_text_encoder.zero_grad()
        text_embeddings = text_encoder(texts)

        # Train the generator
        optimizer_generator.zero_grad()
        generated_images = generator(text_embeddings)

        # Train the discriminator
        optimizer_discriminator.zero_grad()
        discriminator_output_real = discriminator(images)
        discriminator_output_fake = discriminator(generated_images.detach())

        # Update the models' parameters
        optimizer_text_encoder.step()
        optimizer_generator.step()
        optimizer_discriminator.step()

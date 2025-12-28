import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. Global Device Detection (Apple Silicon Support)
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Manifold_Alignment_GAN: Using Apple MPS (Metal Performance Shaders).")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Manifold_Alignment_GAN: Using CUDA.")
else:
    device = torch.device("cpu")
    print("Manifold_Alignment_GAN: Using CPU.")

# ==========================================
# 2. Network Architectures (Missing in previous step)
# ==========================================

class GeneratorNet(nn.Module):
    """
    Simple MLP Generator
    """
    def __init__(self, input_dim, output_dim):
        super(GeneratorNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.main(x)

class DiscriminatorNet(nn.Module):
    """
    Simple MLP Discriminator
    """
    def __init__(self, input_dim):
        super(DiscriminatorNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# ==========================================
# 3. Helper Functions
# ==========================================

def sample_data(data, batch_size):
    """
    Sample data from numpy array.
    """
    # Safety check: if batch_size > data size, sample with replacement or reduce batch size
    if batch_size > data.shape[0]:
        indices = np.random.choice(data.shape[0], size=batch_size, replace=True)
    else:
        indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
    return data[indices, :]

def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    return data

# ==========================================
# 4. Training Step Functions (FIXED)
# ==========================================

def train_discriminator(optimizer, discriminator, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # FIX: Send target to correct device
    error_real = nn.BCELoss()(prediction_real, ones_target(N).to(device))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # FIX: Send target to correct device
    error_fake = nn.BCELoss()(prediction_fake, zeros_target(N).to(device))
    error_fake.backward()
    
    optimizer.step()
    return error_real + error_fake

def train_generator(optimizer, discriminator, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    
    # Calculate error and backpropagate
    # FIX: Send target to correct device
    error = nn.BCELoss()(prediction, ones_target(N).to(device))
    error.backward()
    
    optimizer.step()
    return error

# ==========================================
# 5. Main Training Loop (FIXED)
# ==========================================

def train(generator, discriminator, batch_size, source_tech, target_tech, num_epochs, 
          g_learning_rate, d_learning_rate, checkpoint_epoch, techs, path_prefix, path_suffix):

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_learning_rate)
    g_optimizer = optim.SGD(generator.parameters(), lr=g_learning_rate)
    
    # Ensure models are on the correct device
    generator.to(device)
    discriminator.to(device)

    # Determine safe batch size
    actual_batch_size = min(batch_size, source_tech.shape[0], target_tech.shape[0])
    # Calculate iterations to cover roughly one epoch worth of data
    iterations_per_epoch = max(1, source_tech.shape[0] // actual_batch_size)

    for epoch in range(num_epochs):
        # Progress check
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} / {num_epochs}")

        g_loss = 0
        d_loss = 0
        
        for _ in range(iterations_per_epoch):
            # 1. Train Discriminator
            real_data = sample_data(target_tech, actual_batch_size)
            fake_data_source = sample_data(source_tech, actual_batch_size)
            
            # FIX: Send data to device
            real_data = torch.tensor(real_data).float().to(device)
            source = torch.tensor(fake_data_source).float().to(device)
            
            # Generate fake data
            fake_data = generator(source).detach()
            
            # Train D
            d_error = train_discriminator(d_optimizer, discriminator, real_data, fake_data)
            d_loss += d_error.item()

            # 2. Train Generator
            # Generate fake data
            fake_data = generator(source)
            
            # Train G
            g_error = train_generator(g_optimizer, discriminator, fake_data)
            g_loss += g_error.item()

        # Save Checkpoints
        if epoch % checkpoint_epoch == 0 and epoch != 0:
            path = "{}/Models/{}_to_{}_Generator_{}_{}.pt".format(
                path_prefix, techs[0], techs[1], epoch, path_suffix
            )
            # Ensure folder exists (redundancy check)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            
            torch.save(generator, path)
            
    return generator
import functools
import math
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

from os.path import expanduser
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from data import get_dataset, NumpyDataset2d, load_cut_data
from model import VAE, masked_mse, reparameterize, gaussian_kl

batch_size = 48
beta = 1  # related to \beta-VAE?
residual = False
in_memory = True
return_2d = True
voxel_standardize = True

if return_2d:
    train, test, mask = load_cut_data(voxel_standardize=voxel_standardize)
    train_dataset = NumpyDataset2d(train)
    test_dataset = NumpyDataset2d(test)
else:
    train_dataset, test_dataset, mask = get_dataset(in_memory=in_memory,
                                                    return_2d=return_2d)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)

embedding_dim = 32
model = VAE(input_2d=return_2d, embedding_dim=embedding_dim)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
mask = mask.to(device)

# TODO: switching from cuda:0 to cuda:1 returns error from summary
# if return_2d:
#     summary(model, (1, 91, 109))
# else:
#     summary(model, (1, 91, 109, 91))

loss_function = functools.partial(masked_mse, mask=mask)

optimizer = Adam(model.parameters(), lr=1e-3, amsgrad=True)
# Adaptively change learning rate on plateau
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                           mode='min', patience=5,
                                           threshold_mode='abs')
n_epochs = 40
total_loss = 0
n_batch = math.ceil(len(train_dataset) / batch_size)
mean = torch.zeros_like(train_dataset[0])
# Compute mean
if residual:
    length = 0
    for this_data in train_loader:
        length += this_data.shape[0]
        mean += this_data.sum(dim=0)
    mean /= length
mean = mean.to(device)

# for plotting ELBO train & val loss
list_train_elbo = []
list_val_elbo = []
latents = []
latents_reparam = []
list_rec = []

reparam = False

for epoch in range(n_epochs):
    epoch_batch = 0
    verbose_loss = 0
    verbose_penalty = 0
    verbose_batch = 0
    epoch_train_elbo = 0
    epoch_val_elbo = 0

    for this_data in train_loader:
        model.train()
        model.zero_grad()
        this_data[this_data >= 1] = 1
        this_data = this_data.to(device)
        this_data -= mean[None, ...]
        mu, log_var = model.encoder(this_data)
        penalty = gaussian_kl(mu, log_var)
        if reparam:
            latent = reparameterize(mu, log_var)
            latents.append(latent)
        else:
            latent = mu
            latents_reparam.append(latent)
        rec = model.decoder(latent)

        penalty *= beta
        loss = loss_function(rec, this_data)
        elbo = loss + penalty
        elbo.backward()
        optimizer.step()
        verbose_loss += loss.item()
        verbose_penalty += penalty.item()
        epoch_batch += 1
        verbose_batch += 1
        if epoch_batch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_batch = 0
                val_loss = 0
                val_penalty = 0
                for this_test_data in test_loader:
                    this_test_data = this_test_data.to(device)
                    this_test_data -= mean[None, ...]
                    rec, this_val_penalty = model(this_test_data)
                    this_val_penalty *= beta
                    this_val_loss = loss_function(rec, this_test_data)
                    val_loss += this_val_loss.item()
                    val_penalty += this_val_penalty.item()
                    val_batch += 1
            val_loss /= val_batch
            val_penalty /= val_batch
            verbose_loss /= verbose_batch
            verbose_penalty /= verbose_batch
            train_elbo = verbose_loss + verbose_penalty
            val_elbo = val_loss + val_penalty

            print('Epoch %03i | batch %i/%i | '
                  'train_ELBO: %4e | ' 
                  'val_ELBO:%4e | '
                  % (epoch, epoch_batch, n_batch, train_elbo, val_elbo))
            verbose_batch = 0
            train_loss = 0
            penalty = 0
            epoch_train_elbo += train_elbo
            epoch_val_elbo += val_elbo
    epoch_train_elbo = epoch_train_elbo / n_batch
    epoch_val_elbo = epoch_val_elbo / n_batch
    list_train_elbo.append(epoch_train_elbo)
    list_val_elbo.append(epoch_val_elbo)
    scheduler.step(epoch_val_elbo)  # update learning rate
    state_dict = model.state_dict()

    if return_2d:
        name = '2D_vae_dilated_e_%03i_loss_%.4e.pkl' % (epoch, elbo)
    else:
        name = 'vae_dilated_e_%03i_loss_%.4e.pkl' % (epoch, elbo)

    # Reconstruct the image after training
    recs = []
    with torch.no_grad():
        model.eval()
        for test_data in test_loader:
            test_data = test_data.to(device)
            test_data -= mean[None, ...]
            rec, penalty = model(test_data)
            mu, log_var = model.encoder(test_data)
            if reparam:
                latent = reparameterize(mu, log_var)
            else:
                latent = mu
            rec = model.decoder(latent)
            recs.append(rec)

    rec = torch.cat(recs, dim=0)
    # mean = mean.to('cpu')
    rec += mean[None, ...]
    rec = rec.masked_fill_(mask[None, None, ...] ^ 1, 0.)
    rec = rec.cpu().numpy()
    list_rec.append(rec)

    torch.save((state_dict, mean),
               expanduser('./output/%s' % name))


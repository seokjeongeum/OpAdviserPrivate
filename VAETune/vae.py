from collections import OrderedDict
import os
from typing import Any
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


def train(model, train_loader, optimizer, epoch, quiet, grad_clip=None):
    model.train()

    if not quiet:
        pbar = tqdm(total=len(train_loader.dataset))
    losses = OrderedDict()
    for x in train_loader:
        out = model.loss(x)
        optimizer.zero_grad()
        out["loss"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f"Epoch {epoch}"
        for k, v in out.items():
            if k not in losses:
                losses[k] = []
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f", {k} {avg_loss:.4f}"

        if not quiet:
            pbar.set_description(desc)
            pbar.update(x.shape[0])
    if not quiet:
        pbar.close()
    return losses


def eval_loss(model, data_loader, quiet):
    model.eval()
    total_losses = OrderedDict()
    with torch.no_grad():
        for x in data_loader:
            out = model.loss(x)
            for k, v in out.items():
                total_losses[k] = total_losses.get(k, 0) + v.item() * x.shape[0]

        desc = "Test "
        for k in total_losses.keys():
            total_losses[k] /= len(data_loader.dataset)
            desc += f", {k} {total_losses[k]:.4f}"
        if not quiet:
            print(desc)
    return total_losses


def train_epochs(model, train_loader, test_loader, train_args, quiet=False):
    epochs, lr = train_args["epochs"], train_args["lr"]
    grad_clip = train_args.get("grad_clip", None)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = OrderedDict(), OrderedDict()
    best_loss = float("inf")  # Initialize with infinity
    best_model_path = (
        f"VAETune/dim{model.latent_dim}.pth"  # Path to save the best model
    )
    epoch = 0
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
    for epoch in range(epoch, epochs):
        model.train()
        train_loss = train(model, train_loader, optimizer, epoch, quiet, grad_clip)
        test_loss = eval_loss(model, test_loader, quiet)

        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = []
                test_losses[k] = []
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])
        if test_loss["loss"] < best_loss:
            best_loss = test_loss["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss["loss"],
                },
                best_model_path,
            )
    return train_losses, test_losses


class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hiddens=[]):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hiddens = hiddens

        model = []
        prev_h = np.prod(input_shape)
        for h in hiddens + [np.prod(output_shape)]:
            model.append(nn.Linear(prev_h, h))
            model.append(nn.ReLU())
            prev_h = h
        model.pop()
        self.net = nn.Sequential(*model)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x).view(b, *self.output_shape)


class FullyConnectedVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        enc_hidden_sizes=[512, 256],
        dec_hidden_sizes=[256, 512],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MLP(input_dim, 2 * latent_dim, enc_hidden_sizes)
        self.decoder = MLP(latent_dim, 2 * input_dim, dec_hidden_sizes)

    def loss(self, x, beta=1.0):
        # Encode to latent space
        mu_z, log_std_z = self.encoder(x).chunk(2, dim=1)
        z = torch.randn_like(mu_z) * log_std_z.exp() + mu_z

        # Decode back to input space
        mu_x, log_std_x = self.decoder(z).chunk(2, dim=1)
        std_x = torch.exp(log_std_x) + 1e-6
        x_recon = torch.normal(mu_x, std_x)

        # Reconstruction loss (normalized)
        categorical_recon_loss = 0
        i = 0
        cross_entropy_loss = nn.CrossEntropyLoss()
        with open("VAETune/categories") as f:
            for category in f:
                n = len(eval(eval(category)))
                categorical_recon_loss += cross_entropy_loss(
                    x_recon[:, i : i + n],
                    x[:, i : i + n],
                )
                i += n
        l1_loss = nn.L1Loss()
        numerical_recon_loss = l1_loss(x_recon[:, i:], x[:, i:])
        recon_loss = categorical_recon_loss + numerical_recon_loss

        # KL Divergence loss
        kl_loss = (
            -0.5
            * torch.sum(
                1 + 2 * log_std_z - mu_z.pow(2) - (2 * log_std_z).exp(), dim=1
            ).mean()
        )

        # Total VAE loss with beta weight on KL
        total_loss = recon_loss + beta * kl_loss

        return OrderedDict(loss=total_loss, recon_loss=recon_loss, kl_loss=kl_loss)

    def sample(self, n, noise=True):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)
            mu, log_std = self.decoder(z).chunk(2, dim=1)
            if noise:
                z = torch.randn_like(mu) * log_std.exp() + mu
            else:
                z = mu
        return z.cpu().numpy()


class CustomDataset(data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index) -> Any:
        return torch.tensor(self.dataframe.iloc[index].values.astype("float32"))


def plot_losses(train_losses, test_losses, latent_dim, save_dir="VAETune/plots"):
    """
    Plot and save training and testing losses for total loss, reconstruction loss, and KL loss.

    Parameters:
    - train_losses: OrderedDict containing lists of training losses (loss, recon_loss, kl_loss).
    - test_losses: OrderedDict containing lists of testing losses (loss, recon_loss, kl_loss).
    - latent_dim: The dimension of the latent space (for plot title).
    - save_dir: Directory to save plots. Defaults to "plots".
    """
    epochs = range(len(test_losses["loss"]))  # Test losses recorded once per epoch

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 3 subplots: Total Loss, Reconstruction Loss, KL Loss
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    loss_names = ["loss", "recon_loss", "kl_loss"]

    for i, loss_name in enumerate(loss_names):
        # Plot training and testing losses
        axs[i].plot(
            np.linspace(0, len(epochs), len(train_losses[loss_name])),
            train_losses[loss_name],
            label=f"Train {loss_name}",
            alpha=0.7,
        )
        axs[i].plot(
            np.arange(len(epochs)),
            test_losses[loss_name],
            label=f"Test {loss_name}",
            alpha=0.7,
        )

        axs[i].set_title(f"{loss_name.capitalize()} (Latent dim: {latent_dim})")
        axs[i].set_xlabel("Epoch")
        axs[i].set_ylabel("Loss")
        axs[i].legend()
        axs[i].grid(True)
        # axs[i].ylim(top=0)

    plt.tight_layout()

    # Save the plot to a file
    plot_filename = os.path.join(save_dir, f"loss_plot_latent{latent_dim}.png")
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")

    plt.close()  # Close the figure to free up memory


if __name__ == "__main__":
    df = CustomDataset(pd.read_csv("VAETune/transformed.csv", index_col=0))
    train_data, test_data = train_test_split(df)
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)
    for latent_dim in [1, 20, 290]:
        model = FullyConnectedVAE(290, latent_dim)
        train_losses, test_losses = train_epochs(
            model,
            train_loader,
            test_loader,
            dict(epochs=200, lr=1e-3),
            quiet=False,
        )
        plot_losses(train_losses, test_losses, latent_dim)

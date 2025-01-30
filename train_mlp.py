import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import os
import numpy as np
import os.path as osp
import functools
from utils.pre_run import OptimOptions, load_all_form_checkpoint
from utils.visualization import motion2bvh_rot
from motion_class import DynamicData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_bvh_from_W(output_path, args, g_ema, mean_latent, predicted_W, motion_statics, normalisation_data, device):
    """
    Generates a BVH file from the predicted W+ latent vector.
    """
    predicted_W = torch.tensor(predicted_W, device=device, dtype=torch.float32)
    if predicted_W.shape != (1, 14, 512):
        raise ValueError(f"Incorrect W+ shape: {predicted_W.shape}, expected [1, 14, 512]")

    with torch.no_grad():
        generated_motion, _, _ = g_ema([predicted_W], truncation=0.7, truncation_latent=mean_latent, input_is_latent=True)
    
    generated_motion_data = DynamicData(generated_motion, motion_statics, use_velocity=args.use_velocity)
    mean_tensor = torch.tensor(normalisation_data["mean"], dtype=torch.float32, device=device)
    std_tensor = torch.tensor(normalisation_data["std"], dtype=torch.float32, device=device)
    generated_motion_data = generated_motion_data.un_normalise(mean_tensor, std_tensor)
    
    save_bvh = functools.partial(motion2bvh_rot)
    save_bvh(generated_motion_data, output_path)
    print(f"Generated BVH file saved to: {output_path}")


# ========================
# 1. Define the Encoder Model
# ========================
class MotionEncoder(nn.Module):
    def __init__(self, motion_dim, latent_dim, hidden_dim=2048, num_layers=8, dropout=0.3):
        super(MotionEncoder, self).__init__()
        layers = []
        in_features = motion_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim * 14))
        self.encoder = nn.Sequential(*layers)
        self.skip_connection = nn.Linear(motion_dim, latent_dim * 14)

    def forward(self, motion):
        batch_size = motion.shape[0]

        # Ensure motion is reshaped to (batch_size, motion_dim)
        motion = motion.view(batch_size, -1)  # Flatten the input properly
        encoded_output = self.encoder(motion)
        skip_output = self.skip_connection(motion)
        W_pred = encoded_output + skip_output
        return W_pred.view(batch_size, 14, 512)



# ========================
# 2. Generate Training Data with Diversity Enhancement
# ========================
def generate_training_data(g_ema, mean_latent, num_samples, motion_statics, device, latent_dim):
    X_train, Y_train = [], []

    for _ in tqdm(range(num_samples), desc="Generating training data"):
        W = torch.randn(1, latent_dim, device=device) * 1.5  # Widen sampling range
        W += 0.05 * torch.randn_like(W)  # Add small perturbation
        W = W.unsqueeze(1).repeat(1, 14, 1)
        
        with torch.no_grad():
            motion, _, _ = g_ema([W], truncation=0.7, truncation_latent=mean_latent, input_is_latent=True)
        
        X_train.append(motion.cpu().numpy())  # (1, 4, 23, 64)
        Y_train.append(W.cpu().numpy())  # (1, 14, 512)

    # Convert lists to numpy arrays
    X_train = np.array(X_train, dtype=np.float32).squeeze()  # Shape: (num_samples, 4, 23, 64)
    Y_train = np.array(Y_train, dtype=np.float32).squeeze()  # Shape: (num_samples, 14, 512)

    # Reshape X_train to (num_samples, 5888) before returning
    X_train = X_train.reshape(X_train.shape[0], -1)  # (num_samples, 4 * 23 * 64)

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)


# ========================
# 3. Train the Encoder with Improved Techniques
# ========================
def combined_loss(W_pred, W):
    criterion = nn.MSELoss()
    cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
    mse = criterion(W_pred, W)
    W_pred_flat = W_pred.view(W_pred.shape[0], -1)
    W_flat = W.view(W.shape[0], -1)
    cos = cosine_loss(W_pred_flat, W_flat, torch.ones(W.shape[0]).to(W.device))
    return mse + 0.1 * cos

def train_encoder(encoder, g_ema, mean_latent, motion_statics, normalisation_data, 
                  X_train, Y_train, target_motion, num_epochs=5000, batch_size=512, 
                  lr=0.005, save_path="encoder.pth", device="cpu", 
                  output_bvh_dir="generated_motions_mlp_modified", args=None):
    
    dataset = data.TensorDataset(X_train, Y_train)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_epochs * len(dataloader), pct_start=0.1)

    
    encoder.train()
    os.makedirs(output_bvh_dir, exist_ok=True)
    losses = []  # Store loss values
    
    for epoch in tqdm(range(num_epochs), desc="Training Encoder"):
        epoch_loss = 0.0
        for motion, W in dataloader:
            motion, W = motion.to(device), W.to(device)
            optimizer.zero_grad()
            W_pred = encoder(motion)
            loss = combined_loss(W_pred, W)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # Store average loss for the epoch
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        # Log loss every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.6f}")

        # Save generated motion every 100 epochs
        if epoch % 100 == 0:
            encoder.eval()
            with torch.no_grad():
                predicted_W = encoder(target_motion).detach().cpu().numpy()
                generate_bvh_from_W(osp.join(output_bvh_dir, f"motion_epoch_{epoch}.bvh"),
                                    args, g_ema, mean_latent, predicted_W, 
                                    motion_statics, normalisation_data, device)
            encoder.train()

    torch.save(encoder.state_dict(), save_path)
    print(f"Encoder saved to {save_path}")

    # Save loss logs
    loss_log_path = osp.join(output_bvh_dir, "training_losses.npy")
    np.save(loss_log_path, np.array(losses))
    print(f"Training losses saved to {loss_log_path}")



# ========================
# 4. Main Function - Load Checkpoint and Run Training
# ========================
def main(args_not_parsed):
    parser = OptimOptions()
    args = parser.parse_args(args_not_parsed)
    g_ema, _, motion_data, mean_latent, motion_statics, normalisation_data, _ = load_all_form_checkpoint(args.ckpt, args, return_motion_data=True)
    
    target_motion = motion_data[[args.target_idx]]
    target_motion = torch.tensor(motion_data, dtype=torch.float32).to(device)
    target_motion = target_motion.permute(0, 2, 1, 3)
    
    num_samples = 10000
    X_train, Y_train = generate_training_data(g_ema, mean_latent, num_samples, motion_statics, device, args.latent)

    print("X_train shape:", X_train.shape)  # Should be (num_samples, 4, 23, 64)
    print("Y_train shape:", Y_train.shape)  # Should be (num_samples, 14, 512)
    print("Motion shape before encoder:", target_motion[[args.target_idx]].shape)

    encoder = MotionEncoder(4 * 23 * 64, args.latent).to(device)
    
    train_encoder(encoder, g_ema, mean_latent, motion_statics, normalisation_data, 
              X_train, Y_train, target_motion[[args.target_idx]].reshape(1, -1), num_epochs=5000, 
              batch_size=2048, device=device, args=args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
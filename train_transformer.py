import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import os
import sys
from utils.pre_run import OptimOptions, load_all_form_checkpoint  # Import from existing project
import os.path as osp
import os
import torch
import functools
from tqdm import tqdm
from utils.visualization import motion2bvh_rot
from utils.pre_run import OptimOptions, load_all_form_checkpoint
from motion_class import DynamicData


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def generate_bvh_from_W(output_path, args, g_ema, mean_latent, predicted_W, motion_statics, normalisation_data, device):
    """
    Generates a BVH file from the predicted W+ latent vector.
    """
    # Convert predicted W+ to tensor
    predicted_W = torch.tensor(predicted_W, device=device, dtype=torch.float32)

    # Ensure W+ shape is [1, 14, 512]
    if predicted_W.shape != (1, 14, 512):
        raise ValueError(f"Incorrect W+ shape: {predicted_W.shape}, expected [1, 14, 512]")

    # Generate motion from predicted W+
    with torch.no_grad():
        generated_motion, _, _ = g_ema([predicted_W], truncation=0.7, truncation_latent=mean_latent, input_is_latent=True)

    # Convert to DynamicData for processing
    generated_motion_data = DynamicData(generated_motion, motion_statics, use_velocity=args.use_velocity)

    # Un-normalize motion
    mean_tensor = torch.tensor(normalisation_data["mean"], dtype=torch.float32, device=device)
    std_tensor = torch.tensor(normalisation_data["std"], dtype=torch.float32, device=device)
    generated_motion_data = generated_motion_data.un_normalise(mean_tensor, std_tensor)

    # Output path for BVH file
    # os.makedirs(args.out_path, exist_ok=True)
    output_bvh_path = osp.join(output_path)

    # Save BVH file
    save_bvh = functools.partial(motion2bvh_rot)
    save_bvh(generated_motion_data, output_bvh_path)

    print(f"Generated BVH file saved to: {output_bvh_path}")




# Define the Encoder Model
class TransformerMotionEncoder(nn.Module):
    def __init__(self, motion_dim, latent_dim, num_heads=8, num_layers=6, hidden_dim=512):
        super(TransformerMotionEncoder, self).__init__()
        self.embedding = nn.Linear(motion_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, latent_dim * 14)  # Output: [batch, 14 * 512]
    
    def forward(self, motion):
        batch_size = motion.shape[0]
        motion = motion.reshape(batch_size, -1)  # Flatten input
        motion_emb = self.embedding(motion).unsqueeze(1)  # Add sequence dimension
        encoded_motion = self.transformer_encoder(motion_emb).squeeze(1)  # Apply Transformer
        W_pred = self.fc_out(encoded_motion)
        return W_pred.view(batch_size, 14, 512)  # Reshape output to [batch_size, 14, 512]


# Generate Training Data
def generate_training_data(g_ema, mean_latent, num_samples, motion_statics, device, latent_dim):
    X_train, Y_train = [], []
    
    for _ in tqdm(range(num_samples), desc="Generating training data"):
        # Sample random W+ vector
        W = torch.randn(1, latent_dim, device=device)  # Shape: [1, 512]
        
        W = W.unsqueeze(1).repeat(1, 14, 1)

        # Generate motion from W
        with torch.no_grad():
            motion, _, _ = g_ema([W], truncation=0.7, truncation_latent=mean_latent, input_is_latent=True)

        X_train.append(motion.cpu().numpy())
        Y_train.append(W.cpu().numpy())
    
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)


# Train the Encoder
def train_encoder(args, encoder, g_ema, mean_latent, motion_statics, normalisation_data, 
                  X_train, Y_train, target_motion, num_epochs=1000, batch_size=32, 
                  lr=0.001, save_path="transformer_encoder.pth", device="cpu", 
                  output_bvh_dir="generated_motions"):
    dataset = data.TensorDataset(X_train, Y_train)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(encoder.parameters(), lr=lr)
    encoder.train()
    
    os.makedirs(output_bvh_dir, exist_ok=True)
    
    # Convert target motion to tensor if it's not already
    target_motion = torch.tensor(target_motion, dtype=torch.float32).to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for motion, W in dataloader:
            motion, W = motion.to(device), W.to(device)
            optimizer.zero_grad()
            W_pred = encoder(motion)
            loss = criterion(W_pred, W)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
        
        # Generate motion from target motion every 100 epochs
        if epoch % 100 == 0:
            encoder.eval()
            with torch.no_grad():
                predicted_W = encoder(target_motion.unsqueeze(0)).detach().cpu().numpy()  # Ensure batch dimension
                
            output_bvh_path = osp.join(output_bvh_dir, f"motion_epoch_{epoch}.bvh")
            generate_bvh_from_W(output_bvh_path, args, g_ema, mean_latent, predicted_W, motion_statics, normalisation_data, device)
            print(f"Generated motion saved at {output_bvh_path}")
    
    torch.save(encoder.state_dict(), save_path)
    print(f"Transformer Encoder saved to {save_path}")



# Load Encoder for Inference
def load_encoder(motion_dim, latent_dim, model_path, device):
    encoder = TransformerMotionEncoder(motion_dim, latent_dim)
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.to(device)
    encoder.eval()
    return encoder

# Inference - Predict W from Motion
def infer_W(encoder, target_motion):
    encoder.eval()
    with torch.no_grad():
        W_pred = encoder(target_motion.unsqueeze(0))
    return W_pred

# Main Function - Load Checkpoint and Run Training
def main(args_not_parsed):
    parser = OptimOptions()
    args = parser.parse_args(args_not_parsed)

    # Load checkpoint and required data
    g_ema, _, motion_data, mean_latent, motion_statics, normalisation_data, args = load_all_form_checkpoint(
        args.ckpt, args, return_motion_data=True
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Define dimensions
    motion_dim = 4 * 23 * 64
    latent_dim = args.latent

    # Initialize Transformer Encoder
    encoder = TransformerMotionEncoder(motion_dim, latent_dim).to(device)

    # Generate training data
    num_samples = 10000
    X_train, Y_train = generate_training_data(g_ema, mean_latent, num_samples, motion_statics, device, latent_dim)

    # Select target motion
    target_motion = torch.tensor(motion_data[args.target_idx], dtype=torch.float32).to(device)

    # Train the encoder with all necessary parameters
    train_encoder(
        args, encoder, g_ema, mean_latent, motion_statics, normalisation_data, 
        X_train.to(device), Y_train.to(device), target_motion, 
        num_epochs=1000, save_path="transformer_encoder.pth", device=device
    )

    # Load trained model for inference
    trained_encoder = load_encoder(motion_dim, latent_dim, "transformer_encoder.pth", device)

    # Predict W+ for the target motion
    predicted_W = trained_encoder(target_motion.unsqueeze(0)).detach().cpu().numpy()

    print("Predicted W shape:", predicted_W.shape)

    # Generate BVH file from the predicted W+
    generate_bvh_from_W(args.out_path, args, g_ema, mean_latent, predicted_W, motion_statics, normalisation_data, device)



if __name__ == "__main__":
    main(sys.argv[1:])
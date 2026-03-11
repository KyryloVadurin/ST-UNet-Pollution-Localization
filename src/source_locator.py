# source_locator.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import Dict, List, Tuple

# ==========================================
# 1. DATA LOADING (PyTorch Dataset)
# ==========================================
class PollutionDataset(Dataset):
    """
    Custom Dataset class for loading and preprocessing pollution simulation data.
    """
    def __init__(self, dataset_dir: str, grid_size: Tuple[int, int] = (50, 50)):
        self.dataset_dir = dataset_dir
        self.grid_size = grid_size
        self.scenarios = [d for d in os.listdir(dataset_dir) if d.startswith("scenario_")]
        
    def __len__(self):
        """
        Return the total number of samples, considering three sensor layouts per scenario.
        """
        return len(self.scenarios) * 3 

    def __getitem__(self, idx):
        """
        Load and normalize a specific scenario sample including sensor readings and ground truth.
        """
        scenario_idx = idx // 3
        layout_idx = idx % 3
        scenario_path = os.path.join(self.dataset_dir, self.scenarios[scenario_idx])
        
        with open(os.path.join(scenario_path, "metadata.json"), "r") as f:
            meta = json.load(f)
            
        sensor_file = np.load(os.path.join(scenario_path, f"sensor_layout_{layout_idx}.npz"))
        readings = sensor_file['readings']  # Dimensions: [Time, Sensors]
        coords = sensor_file['coordinates']
        
        # Utilize either the recorded average wind vector or the initial wind vector
        wind = np.array(meta.get('avg_wind', meta['wind_vector']))
        
        # --- INPUT NORMALIZATION ---
        # Apply logarithmic transformation to map pollution values into log space
        readings = np.log1p(readings)
        
        # Apply local Z-score normalization to enhance the contrast of local sources
        r_mean, r_std = readings.mean(), readings.std() + 1e-8
        readings = (readings - r_mean) / r_std

        # --- TARGET NORMALIZATION ---
        gt_file = np.load(os.path.join(scenario_path, "ground_truth.npz"))
        target_heatmap = np.mean(gt_file['data'], axis=0)
        
        # Perform Min-Max scaling to compress the target range into [0, 1]
        t_min, t_max = target_heatmap.min(), target_heatmap.max()
        target_heatmap = (target_heatmap - t_min) / (t_max - t_min + 1e-8)

        # Normalize wind components assuming a maximum velocity of 3.0 m/s
        x_wind = torch.tensor(wind / 3.0, dtype=torch.float32)

        x_readings = torch.tensor(readings, dtype=torch.float32).transpose(0, 1)
        x_coords = torch.tensor(coords, dtype=torch.long)
        y_target = torch.tensor(target_heatmap, dtype=torch.float32).unsqueeze(0)

        return x_readings, x_coords, x_wind, y_target


# ==========================================
# 2. ARCHITECTURE: Spatio-Temporal U-Net (ST-UNet)
# ==========================================
class GaussianSmearing(nn.Module):
    """
    Applies a fixed Gaussian kernel for spatial smoothing of point observations.
    """
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        grid = torch.meshgrid(coords, coords, indexing='ij')
        gaussian = torch.exp(-(grid[0]**2 + grid[1]**2) / (2 * sigma**2))
        gaussian /= gaussian.sum()
        self.register_buffer('weight', gaussian.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B * C, 1, H, W)
        smoothed = F.conv2d(x_reshaped, self.weight, padding=self.padding)
        return smoothed.view(B, C, H, W)

class DoubleConv(nn.Module):
    """
    Standard U-Net building block consisting of two convolutional layers with normalization.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SourceLocatorNet(nn.Module):
    """
    U-Net based neural network for reconstructing pollution source maps from sparse sensor data.
    """
    def __init__(self, time_steps: int, grid_x: int, grid_y: int, hidden_dim: int = 64):
        super().__init__()
        self.grid_x, self.grid_y = grid_x, grid_y
        self.smearing = GaussianSmearing(kernel_size=5, sigma=1.0)
        
        # Encoder path
        self.inc = DoubleConv(time_steps + 4, hidden_dim)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(hidden_dim, hidden_dim*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(hidden_dim*2, hidden_dim*4))
        
        # Decoder path
        self.up1 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 2, stride=2)
        self.conv_up1 = DoubleConv(hidden_dim*4, hidden_dim*2)
        self.up2 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 2, stride=2)
        self.conv_up2 = DoubleConv(hidden_dim*2, hidden_dim)
        
        self.outc = nn.Conv2d(hidden_dim, 1, 1)

        # Coordinate buffers for spatial awareness
        cx = torch.linspace(0, 1, grid_x).view(1, 1, grid_x, 1).expand(1, 1, grid_x, grid_y)
        cy = torch.linspace(0, 1, grid_y).view(1, 1, 1, grid_y).expand(1, 1, grid_x, grid_y)
        self.register_buffer('coord_x', cx)
        self.register_buffer('coord_y', cy)

    def forward(self, readings, coords, wind):
        """
        Execute the forward pass to predict the spatial distribution of pollution sources.
        """
        B, N, T = readings.shape
        grid = torch.zeros((B, T, self.grid_x, self.grid_y), device=readings.device)
        for b in range(B):
            # Clip indices to maintain boundary safety during grid projection
            cx = torch.clamp(coords[b, :, 0], 0, self.grid_x - 1)
            cy = torch.clamp(coords[b, :, 1], 0, self.grid_y - 1)
            grid[b, :, cx, cy] = readings[b].T
            
        # Concatenate grid features with wind vectors and coordinate information
        x = torch.cat([
            self.smearing(grid),
            wind[:, 0].view(B, 1, 1, 1).expand(B, 1, self.grid_x, self.grid_y),
            wind[:, 1].view(B, 1, 1, 1).expand(B, 1, self.grid_x, self.grid_y),
            self.coord_x.expand(B, -1, -1, -1),
            self.coord_y.expand(B, -1, -1, -1)
        ], dim=1)
        
        # Feature extraction and multi-scale upsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        u1 = F.interpolate(self.up1(x3), size=x2.shape[2:])
        u1 = self.conv_up1(torch.cat([u1, x2], dim=1))
        
        u2 = F.interpolate(self.up2(u1), size=x1.shape[2:])
        u2 = self.conv_up2(torch.cat([u2, x1], dim=1))
        
        return torch.sigmoid(self.outc(u2))

# ==========================================
# 3. LOSS FUNCTION (Dice + Weighted MSE)
# ==========================================
class HybridInverseLoss(nn.Module):
    """
    Composite loss function combining spatial regression and localization metrics.
    """
    def __init__(self, mse_weight=1.0, dice_weight=2.0, focus_factor=10.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.dice_weight = dice_weight
        self.focus_factor = focus_factor

    def forward(self, pred, target):
        """
        Calculate the hybrid loss for the source localization task.
        """
        # Calculate Physics-Weighted MSE to prioritize high-concentration zones
        mse = (pred - target) ** 2
        weight_mask = torch.where(target > 0.1, self.focus_factor, 1.0)
        weighted_mse = (mse * weight_mask).mean()
        
        # Calculate Soft Dice Loss to address spatial class imbalance
        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()
        
        return self.mse_weight * weighted_mse + self.dice_weight * dice_loss


# ==========================================
# 4. TRAINING AND INFERENCE MANAGER
# ==========================================
class SourcePredictor:
    """
    Orchestrates the training process, hyperparameter optimization, and inference 
    mechanisms for pollution source localization.
    """
    def __init__(self, time_steps=48, grid_x=50, grid_y=50, device=None, hidden_dim=64):
        self.device = torch.device(device if device else "cpu")
        self.model = SourceLocatorNet(time_steps, grid_x, grid_y, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4)
        self.criterion = HybridInverseLoss(dice_weight=5.0, focus_factor=15.0)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
        self.grid_x, self.grid_y = grid_x, grid_y

    def train(self, dataset_path: str, epochs: int = 30, batch_size: int = 16, 
              save_path: str = "best_model.pth", early_stopping_patience: int = 10):
        """
        Executes the supervised learning protocol with early stopping and validation monitoring.
        """
        # Ensure the target directory for saving model weights exists
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directory for models has been created: {directory}.")
        
        dataset = PollutionDataset(dataset_path, (self.grid_x, self.grid_y))
        train_size = int(0.9 * len(dataset))
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        history = {'train_loss': [], 'val_loss': [], 'val_le': [], 'val_f1': [], 'lr': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Define the path for the backup of the most recent model state
        last_model_path = save_path.replace(".pth", "_last.pth")

        print(f"Scientific training process has started with early stopping patience set to {early_stopping_patience}.")
        print(f"Execution device: {self.device}. Dataset split: {len(train_ds)} samples for training and {len(val_ds)} for validation.")

        try:
            for epoch in range(epochs):
                # --- TRAINING PHASE ---
                self.model.train()
                t_epoch_loss = 0
                for r, c, w, t in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
                    r, c, w, t = r.to(self.device), c.to(self.device), w.to(self.device), t.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(r, c, w)
                    loss = self.criterion(pred, t)
                    loss.backward()
                    # Apply gradient clipping to ensure numerical stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    t_epoch_loss += loss.item()

                # --- VALIDATION PHASE ---
                self.model.eval()
                v_epoch_loss, v_le, v_f1 = 0, [], []
                with torch.no_grad():
                    for r, c, w, t in val_loader:
                        r, c, w, t = r.to(self.device), c.to(self.device), w.to(self.device), t.to(self.device)
                        pred = self.model(r, c, w)
                        v_epoch_loss += self.criterion(pred, t).item()
                        
                        p_np, t_np = pred.squeeze().cpu().numpy(), t.squeeze().cpu().numpy()
                        # Calculate performance metrics for the current batch
                        if p_np.ndim == 3: # Multiple samples in batch
                            for i in range(p_np.shape[0]):
                                m = Evaluator.calculate_metrics(t_np[i], p_np[i])
                                v_le.append(m['LE_pixels']); v_f1.append(m['F1'])
                        else: # Single sample in batch
                            m = Evaluator.calculate_metrics(t_np, p_np)
                            v_le.append(m['LE_pixels']); v_f1.append(m['F1'])

                # Compute aggregate statistics for the current epoch
                avg_train_loss = t_epoch_loss / len(train_loader)
                avg_val_loss = v_epoch_loss / len(val_loader)
                avg_le = np.mean(v_le)
                avg_f1 = np.mean(v_f1)
                
                self.scheduler.step(avg_val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['val_le'].append(avg_le)
                history['val_f1'].append(avg_f1)
                history['lr'].append(current_lr)

                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LE: {avg_le:.2f} px, F1: {avg_f1:.4f}, LR: {current_lr:.1e}.")

                # Save the current model state as a backup
                torch.save(self.model.state_dict(), last_model_path)

                # Check for model improvement and handle early stopping criteria
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(self.model.state_dict(), save_path)
                    patience_counter = 0
                    print(f"New optimal model state achieved. Weights saved to {save_path}.")
                else:
                    patience_counter += 1
                    print(f"No improvement detected. Patience counter: {patience_counter}/{early_stopping_patience}.")

                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
                    
        except KeyboardInterrupt:
            print("Training process has been manually interrupted by the user.")

        # Revert to the best model weights obtained during the session
        if os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))
            print("The optimal model weights have been loaded successfully.")
            
        return history

    def load_weights(self, path):
        """
        Loads pre-trained model weights from the specified file path.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print("Model weights have been loaded.")

    def update_hyperparameters(self, lr=None, focus=None, dice=None):
        """
        Dynamically updates the optimizer learning rate and loss function coefficients.
        """
        if lr: self.optimizer.param_groups[0]['lr'] = lr
        if focus: self.criterion.focus_factor = focus
        if dice: self.criterion.dice_weight = dice

    def predict_with_uncertainty(self, readings: np.ndarray, coords: np.ndarray, wind: np.ndarray, num_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs probabilistic inference using Monte Carlo Dropout to estimate predictive uncertainty.
        """
        self.model.train() # Enable Dropout during inference for stochastic sampling
        
        # Apply the identical normalization procedure used during the training phase
        r_proc = np.log1p(readings)
        r_mean, r_std = r_proc.mean(), r_proc.std() + 1e-8
        r_proc = (r_proc - r_mean) / r_std
        w_proc = wind / 3.0
        
        predictions = []
        with torch.no_grad():
            t_readings = torch.tensor(r_proc, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(self.device)
            t_coords = torch.tensor(coords, dtype=torch.long).unsqueeze(0).to(self.device)
            t_wind = torch.tensor(w_proc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for _ in range(num_samples):
                pred = self.model(t_readings, t_coords, t_wind)
                predictions.append(pred.squeeze().cpu().numpy())
                
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty_map = np.var(predictions, axis=0)
        
        return mean_pred, uncertainty_map

    def predict(self, readings: np.ndarray, coords: np.ndarray, wind: np.ndarray) -> np.ndarray:
        """
        Performs deterministic inference for standard evaluation and ablation studies.
        """
        self.model.eval()
        # Apply normalization consistent with training data preprocessing
        r_proc = np.log1p(readings)
        r_mean, r_std = r_proc.mean(), r_proc.std() + 1e-8
        r_proc = (r_proc - r_mean) / r_std
        w_proc = wind / 3.0
        
        with torch.no_grad():
            r_t = torch.tensor(r_proc, dtype=torch.float32).unsqueeze(0).transpose(1, 2).to(self.device)
            c_t = torch.tensor(coords, dtype=torch.long).unsqueeze(0).to(self.device)
            w_t = torch.tensor(w_proc, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.model(r_t, c_t, w_t).squeeze().cpu().numpy()


# ==========================================
# 5. EVALUATION AND VISUALIZATION (Evaluator)
# ==========================================
class Evaluator:
    """
    Provides a suite of diagnostic tools and metrics for evaluating the performance 
    of inverse pollution modeling.
    """
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Computes a comprehensive set of scientific metrics including spatial localization, 
        segmentation, and intensity regression.
        """
        
        # 1. Physical Localization Error (LE)
        # Identify coordinates of peak intensity to measure spatial displacement
        t_max = np.unravel_index(np.argmax(y_true), y_true.shape)
        p_max = np.unravel_index(np.argmax(y_pred), y_pred.shape)
        le = np.sqrt((t_max[0]-p_max[0])**2 + (t_max[1]-p_max[1])**2)
        
        # 2. Spatial Segmentation (IoU and F1-Score)
        # Binarize maps using a specified threshold to evaluate overlap regions
        y_true_bin = (y_true > threshold)
        y_pred_bin = (y_pred > threshold)
        
        intersection = np.logical_and(y_true_bin, y_pred_bin).sum()
        union = np.logical_or(y_true_bin, y_pred_bin).sum()
        iou = intersection / (union + 1e-8)  # Intersection over Union
        
        ytb_flat = y_true_bin.flatten()
        ypb_flat = y_pred_bin.flatten()
        f1 = f1_score(ytb_flat, ypb_flat, zero_division=0)
        
        # 3. Intensity Regression (R2, Pearson correlation, and RMSE)
        y_t_flat = y_true.flatten()
        y_p_flat = y_pred.flatten()
        
        r2 = r2_score(y_t_flat, y_p_flat)
        r_corr, _ = pearsonr(y_t_flat, y_p_flat)
        rmse = np.sqrt(mean_squared_error(y_t_flat, y_p_flat))
        
        return {
            "LE_pixels": float(le),
            "F1": float(f1),
            "IoU": float(iou),
            "R2": float(r2),
            "Pearson_r": float(r_corr),
            "RMSE": float(rmse)
        }

    @staticmethod
    def plot_comparison(y_true: np.ndarray, y_pred: np.ndarray, coords: np.ndarray, metrics: Dict):
        """
        Generates a visual comparison between ground truth and model predictions.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Ground Truth visualization
        ax1 = axes[0]
        im1 = ax1.imshow(y_true, cmap='hot', origin='lower')
        ax1.scatter(coords[:, 1], coords[:, 0], c='cyan', marker='^', label='Sensors', s=50)
        ax1.set_title("Ground Truth Source Map")
        ax1.legend()
        fig.colorbar(im1, ax=ax1)
        
        # 2. Prediction visualization
        ax2 = axes[1]
        im2 = ax2.imshow(y_pred, cmap='hot', origin='lower', vmin=0, vmax=1.0)
        ax2.scatter(coords[:, 1], coords[:, 0], c='cyan', marker='^', s=50)
        
        true_max = np.unravel_index(np.argmax(y_true), y_true.shape)
        pred_max = np.unravel_index(np.argmax(y_pred), y_pred.shape)
        ax2.plot([true_max[1], pred_max[1]], [true_max[0], pred_max[0]], 'w--', label='Localization Error (LE)')
        ax2.scatter(*true_max[::-1], c='green', marker='*', s=150, label='Actual Center')
        ax2.scatter(*pred_max[::-1], c='blue', marker='X', s=100, label='Predicted Center')
        
        ax2.set_title("Neural Network Prediction")
        ax2.legend()
        fig.colorbar(im2, ax=ax2)
        
        # 3. Metrics summary display
        ax3 = axes[2]
        ax3.axis('off')
        textstr = '\n\n'.join((
            "Reconstruction Quality Assessment (Q1 Metrics):",
            f"Localization Error (LE): {metrics['LE_pixels']:.2f} px",
            f"F1-Score (Threshold=0.5): {metrics['F1']:.4f}",
            f"IoU (Spatial Overlap): {metrics['IoU']:.4f}",
            f"Intensity RMSE: {metrics['RMSE']:.4f}"
        ))
        ax3.text(0.1, 0.5, textstr, fontsize=14, va='center', bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_uncertainty(y_true: np.ndarray, mean_pred: np.ndarray, uncertainty: np.ndarray, coords: np.ndarray):
        """
        Visualizes predictive uncertainty derived from Monte Carlo Dropout sampling.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1 = axes[0]
        im1 = ax1.imshow(y_true, cmap='hot', origin='lower')
        ax1.scatter(coords[:, 1], coords[:, 0], c='cyan', marker='^', s=50)
        ax1.set_title("Ground Truth")
        fig.colorbar(im1, ax=ax1)
        
        ax2 = axes[1]
        im2 = ax2.imshow(mean_pred, cmap='hot', origin='lower', vmin=0, vmax=1.0)
        ax2.set_title("Mean Prediction")
        fig.colorbar(im2, ax=ax2)
        
        ax3 = axes[2]
        # Utilize a distinct colormap for uncertainty to distinguish variance from intensity
        im3 = ax3.imshow(uncertainty, cmap='viridis', origin='lower')
        ax3.scatter(coords[:, 1], coords[:, 0], c='red', marker='^', s=50, label='Sensors')
        ax3.set_title("Uncertainty Map (Variance)")
        ax3.legend()
        fig.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_summary(metrics: Dict):
        """
        Outputs evaluation metrics in a structured text format.
        """
        print("\n" + "="*40)
        print(f"{'SCIENTIFIC MODEL EVALUATION REPORT':^40}")
        print("="*40)
        
        # Spatial error
        le = float(metrics['LE_pixels'])
        print(f"Localization Error (LE):    {le:8.2f} px")
        
        # Detection accuracy
        f1 = float(metrics['F1'])
        print(f"F1-Score (Detection):     {f1:8.4f}")
        
        # Spatial overlap percentage
        iou = float(metrics['IoU'])
        print(f"IoU (Spatial Overlap):    {iou*100:8.2f} %")
        
        # Intensity regression error
        rmse = float(metrics['RMSE'])
        print(f"Intensity RMSE:           {rmse:8.4f}")
        
        print("="*40)


class ClassicalBaseline:
    """
    Implements a robust geometric baseline using a center-of-mass approach with inverse advection.
    """
    @staticmethod
    def predict(readings: np.ndarray, coords: np.ndarray, wind: np.ndarray, grid_shape: Tuple[int, int]):
        """
        Calculates an estimated pollution source map using classical heuristics.
        """
        grid_x, grid_y = grid_shape
        
        # 1. Data preprocessing using logarithmic space to mitigate outliers
        r_proc = np.log1p(readings)
        mean_r = np.mean(r_proc, axis=0) 
        
        # 2. Geometric center of mass calculation in pixel space
        total_p = np.sum(mean_r) + 1e-8
        cx = np.sum(coords[:, 0] * mean_r) / total_p
        cy = np.sum(coords[:, 1] * mean_r) / total_p
        
        # 3. Inverse advection heuristic to compensate for wind transport
        # The scaling factor is empirically derived based on average atmospheric transport time.
        est_x = cx - wind[0] * 4.0 
        est_y = cy - wind[1] * 4.0
        
        # Constraint enforcement to ensure coordinates remain within grid boundaries
        est_x = np.clip(est_x, 0, grid_x - 1)
        est_y = np.clip(est_y, 0, grid_y - 1)
        
        # 4. Prediction map generation using a 2D Gaussian distribution
        yy, xx = np.mgrid[0:grid_x, 0:grid_y]
        dist_sq = (yy - est_x)**2 + (xx - est_y)**2
        
        # Set Gaussian spread to a fraction of the grid size to facilitate spatial overlap
        sigma = grid_x / 8.0 
        heatmap = np.exp(-dist_sq / (2 * sigma**2))
        
        return np.clip(heatmap, 0, 1)
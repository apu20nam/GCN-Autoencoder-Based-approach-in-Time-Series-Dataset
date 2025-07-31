import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load ECG dataset
print("Loading ECG dataset...")
df = pd.read_csv("http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv", header=None)
data = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

print(f"Dataset shape: {data.shape}")
print(f"Normal samples: {np.sum(labels == 0)}, Anomaly samples: {np.sum(labels == 1)}")

# Improved normalization using StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split dataset with stratification
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)


def create_graphs(samples, k_neighbors=5):
    """Create graphs with better connectivity patterns"""
    data_list = []
    seq_len = samples.shape[1]
    
    for sample in samples:
        # Node features: each time step becomes a node
        x = torch.tensor(sample, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        
        # Create multiple types of edges for better connectivity
        edges = []
        
        # 1. Sequential connections (time series order)
        for i in range(seq_len - 1):
            edges.extend([[i, i + 1], [i + 1, i]])
        
        # 2. Skip connections (every k steps)
        skip = max(1, seq_len // 20)  # adaptive skip based on sequence length
        for i in range(seq_len - skip):
            edges.extend([[i, i + skip], [i + skip, i]])
        
        # 3. Local neighborhood connections (k-nearest temporal neighbors)
        for i in range(seq_len):
            for j in range(max(0, i - k_neighbors//2), min(seq_len, i + k_neighbors//2 + 1)):
                if i != j:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Add self loops for better gradient flow
        edge_index, _ = add_self_loops(edge_index, num_nodes=seq_len)
        
        graph = Data(x=x, edge_index=edge_index)
        data_list.append(graph)
    
    return data_list

print("Creating improved graph representations...")
train_graphs = create__graphs(train_data)
test_graphs = create_graphs(test_data)

# Create data loaders for better batch processing
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)


class ImprovedGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, latent_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

class ImprovedDecoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(latent_dim, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.lin1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.lin2(x)))
        x = self.dropout(x)
        x = self.lin3(x)
        return x

class ImprovedGNNAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super().__init__()
        self.encoder = ImprovedGCNEncoder(in_channels, hidden_channels, latent_dim)
        self.decoder = ImprovedDecoder(latent_dim, hidden_channels, in_channels)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        x_hat = self.decoder(z)
        return x_hat, z

# Training setup with better hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ImprovedGNNAutoEncoder(
    in_channels=1, 
    hidden_channels=64, 
    latent_dim=16
).to(device)

# Better optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Combined loss function
def combined_loss(x_hat, x_true, z, lambda_reg=1e-4):
    # Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x_true)
    
    # L2 regularization on latent representations
    reg_loss = torch.mean(torch.sum(z**2, dim=1))
    
    return recon_loss + lambda_reg * reg_loss

def train_epoch():
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        x_hat, z = model(batch)
        loss = combined_loss(x_hat, batch.x, z)
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def evaluate():
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_hat, z = model(batch)
            loss = F.mse_loss(x_hat, batch.x)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# Training loop with validation
print("Starting training...")
train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(1, 51):  # More epochs
    train_loss = train_epoch()
    val_loss = evaluate()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')

# Load best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Compute reconstruction errors for anomaly detection
recon_errors = []
true_labels = []

print("Computing reconstruction errors...")
with torch.no_grad():
    for i, (graph, label) in enumerate(zip(test_graphs, test_labels)):
        graph = graph.to(device)
        # Create a batch with single graph
        batch = Data(x=graph.x, edge_index=graph.edge_index, batch=torch.zeros(graph.x.size(0), dtype=torch.long).to(device))
        
        x_hat, _ = model(batch)
        loss = F.mse_loss(x_hat, graph.x, reduction='mean').item()
        recon_errors.append(loss)
        true_labels.append(label)

recon_errors = np.array(recon_errors)
true_labels = np.array(true_labels)

# Visualization
plt.subplot(1, 2, 2)
plt.hist(recon_errors[true_labels == 0], bins=50, alpha=0.7, label='Normal', color='blue')
plt.hist(recon_errors[true_labels == 1], bins=50, alpha=0.7, label='Anomaly', color='red')
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.legend()
plt.title("Reconstruction Error Distribution")
plt.yscale('log')  # Log scale for better visualization
plt.tight_layout()
plt.show()

# Improved threshold selection using multiple percentiles
percentiles = [90, 95, 97, 99]
print("\n" + "="*50)
print("ANOMALY DETECTION RESULTS")
print("="*50)

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

for p in percentiles:
    threshold = np.percentile(recon_errors[true_labels == 0], p)
    preds = (recon_errors > threshold).astype(int)
    
    print(f"\nThreshold (Normal {p}th percentile): {threshold:.6f}")
    print("-" * 40)
    print(classification_report(true_labels, preds, target_names=['Normal', 'Anomaly']))
    
    if len(np.unique(preds)) > 1:  # Only compute AUC if we have both classes
        auc_score = roc_auc_score(true_labels, recon_errors)
        print(f"AUC Score: {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    print(f"Confusion Matrix:\n{cm}")

print(f"\nReconstruction Error Statistics:")
print(f"Normal samples - Mean: {np.mean(recon_errors[true_labels == 0]):.6f}, Std: {np.std(recon_errors[true_labels == 0]):.6f}")
print(f"Anomaly samples - Mean: {np.mean(recon_errors[true_labels == 1]):.6f}, Std: {np.std(recon_errors[true_labels == 1]):.6f}")

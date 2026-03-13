import torch                       # Core PyTorch library
import torch.nn as nn                # Neural network layers and modules
import torch.nn.functional as F      # Activation functions and other operations

class TemporalBiLSTM(nn.Module):
    """
    A Bidirectional LSTM (BiLSTM) model for capturing temporal inconsistencies
    across a sequence of video frame features (embeddings).
    
    This model assumes the input features are already extracted by a spatial CNN (like ResNet-50).
    Input Shape: (batch_size, sequence_length, feature_dim)
    Example: (B, 30, 2048)
    """

    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2, num_classes=1):
        """
        Initializes the temporal model architecture.
        
        Args:
            input_dim (int):  Dimension of the input spatial embeddings (default 2048 for ResNet-50).
            hidden_dim (int): Number of hidden units in each LSTM layer.
            num_layers (int): Number of recurrent layers to stack.
            num_classes (int): Number of output units (1 for binary classification: Real vs Fake).
        """
        super(TemporalBiLSTM, self).__init__()
        
        # The LSTM layer: processes the sequence from both directions (bidirectional=True)
        # This helps capture dependencies that might flow forward or backward in time.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,         # Input/output tensors are (B, L, H)
            bidirectional=True,       # Processes sequence forward and backward
            dropout=0.3 if num_layers > 1 else 0  # Regularization to prevent overfitting
        )
        
        # The output of a BiLSTM is 2 * hidden_dim because it concatenates both directions
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),   # Compact the 1024-dim signal
            nn.ReLU(),                        # Non-linear activation
            nn.Dropout(0.3),                  # Further regularization
            nn.Linear(256, num_classes)       # Final logit for classification
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, sequence_length, 2048)
            
        Returns:
            torch.Tensor: The output logit.
        """
        # x shape: (batch, seq_len, input_dim)
        
        # Passing sequence through the BiLSTM
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # We take the features from the LAST time step of the forward pass
        # and the FIRST time step of the backward pass (which is the actual last step of info).
        # Alternatively, common practice for binary classification is to just use the 
        # final hidden state or pool across time. Here we'll use Global Average Pooling across time.
        
        # Global Average Pooling (avg across the seq_len dimension)
        # This reduces (B, L, 1024) -> (B, 1024)
        out = torch.mean(lstm_out, dim=1)
        
        # Pass the pooled representation through the classifier head
        logits = self.fc(out)
        
        return logits

def get_model(input_dim=2048, device='cpu'):
    """
    Utility function to initialize the model and move it to the target device.
    """
    model = TemporalBiLSTM(input_dim=input_dim)
    model = model.to(device)
    return model

if __name__ == "__main__":
    # Test block: create a dummy batch to verify the forward pass works
    # Batch size: 8, Sequence Length: 30 frames, Feature Dim: 2048
    dummy_input = torch.randn(8, 30, 2048)
    model = TemporalBiLSTM()
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (logits): {output.shape}")
    print("Temporal model architecture verification successful.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import time
import os
import warnings

warnings.filterwarnings("ignore")
# Uncomment the following line if you have a specific GPU to use
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 1. Core Autoformer Components
class SeriesDecomp(nn.Module):
    """
    Splits a time series into seasonal and trend components.
    y = seasonal + trend
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        # Using AvgPool1d as a moving average filter
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Sequence_Length, Features]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Seasonal component, Trend component)
        """
        # Pad the start of the sequence for the moving average
        # Padding is (kernel_size - 1) on the left side
        padding = (self.kernel_size - 1, 0)
        # Permute to [Batch, Features, Sequence_Length] for pooling
        x_permuted = x.permute(0, 2, 1)
        
        # Calculate trend component (moving average)
        # The output of avg_pool will have a smaller sequence length, which is what we want.
        trend_init = self.avg_pool(F.pad(x_permuted, padding, 'replicate'))
        trend = trend_init.permute(0, 2, 1) # Back to [Batch, Sequence_Length, Features]
        
        # Calculate seasonal component
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationLayer(nn.Module):
    """
    Computes the Auto-Correlation mechanism as described in the Autoformer paper.
    This is an efficient, vectorized implementation.
    """
    def __init__(self, d_model, n_heads, top_k_factor=5):
        super(AutoCorrelationLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k_factor = top_k_factor

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B, L, _ = query.shape
        _, S, _ = key.shape
        
        # Project and reshape for multi-head
        Q = self.query_projection(query).view(B, L, self.n_heads, self.d_head)
        K = self.key_projection(key).view(B, S, self.n_heads, self.d_head)
        V = self.value_projection(value).view(B, S, self.n_heads, self.d_head)

        # Use FFT to compute correlations for all lags at once
        # This is the "Efficient Computation" part of the paper
        q_fft = torch.fft.rfft(Q.permute(0, 2, 3, 1).contiguous(), n=L)
        k_fft = torch.fft.rfft(K.permute(0, 2, 3, 1).contiguous(), n=L)
        
        # Element-wise product in frequency domain is correlation in time domain
        autocorr_fft = q_fft * torch.conj(k_fft)
        autocorr = torch.fft.irfft(autocorr_fft, n=L)
        autocorr = autocorr.permute(0, 3, 1, 2) # [B, L, n_heads, d_head]

        # Select top-k delays
        top_k = int(self.top_k_factor * np.log(L))
        top_k_vals, top_k_indices = torch.topk(autocorr, top_k, dim=1) # [B, k, n_heads, d_head]

        # Apply softmax to the correlation values
        weights = F.softmax(top_k_vals, dim=1)

        # Aggregate values based on the selected delays (time delay aggregation)
        aggregated_values = torch.zeros_like(Q)
        for i in range(top_k):
            # Roll the value tensor by the selected delay
            delay = top_k_indices[:, i, :, :] # [B, n_heads, d_head]
            
            # Since torch.roll doesn't support batch-wise rolling, we have to iterate or use a more complex method.
            # For simplicity and minimal change, a loop over heads is acceptable.
            rolled_V = torch.roll(V, shifts=-int(delay[0,0,0]), dims=1) # Simplified roll for example
            
            # A more robust way would be to gather based on indices, but roll is the paper's term.
            # Using a simplified roll for this example. A production implementation would need a batched roll.
            
            # The paper's `Roll(V, tau)` is a bit ambiguous in a batched context.
            # A practical interpretation is to use the indices to gather values.
            # Here, we will simplify by just rolling the whole batch by the mean delay.
            # Note: This is a simplification. A perfect implementation needs a batched, per-head roll.
            mean_delay = torch.mean(delay.float()).int().item()
            rolled_V = torch.roll(V, shifts=-mean_delay, dims=1)
            
            # Weight and sum
            aggregated_values += rolled_V * weights[:, i, :, :].unsqueeze(1)
        
        # Concatenate heads and project output
        aggregated_values = aggregated_values.reshape(B, L, -1)
        output = self.out_projection(aggregated_values)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        # x is [Batch, Seq_Len, d_model]
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        x = self.dropout(self.conv2(x)).permute(0, 2, 1)
        return x


# 2. Encoder and Decoder Architecture (Optimized)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, decomp_kernel_size, dropout, top_k_factor, activation='relu'):
        super(EncoderLayer, self).__init__()
        self.autocorrelation = AutoCorrelationLayer(d_model, n_heads, top_k_factor)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.decomp1 = SeriesDecomp(decomp_kernel_size)
        self.decomp2 = SeriesDecomp(decomp_kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Auto-correlation block
        new_x = self.autocorrelation(x, x, x)
        res = x + self.dropout(new_x)
        res, _ = self.decomp1(res) # Decompose and only keep seasonal part
        res = self.norm1(res)

        # Feed-forward block
        new_x = self.feed_forward(res)
        res = res + self.dropout(new_x)
        res, _ = self.decomp2(res) # Decompose and only keep seasonal part
        res = self.norm2(res)
        
        return res


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, decomp_kernel_size, dropout, top_k_factor, activation='relu'):
        super(DecoderLayer, self).__init__()
        self.self_autocorrelation = AutoCorrelationLayer(d_model, n_heads, top_k_factor)
        self.cross_autocorrelation = AutoCorrelationLayer(d_model, n_heads, top_k_factor)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
        self.decomp1 = SeriesDecomp(decomp_kernel_size)
        self.decomp2 = SeriesDecomp(decomp_kernel_size)
        self.decomp3 = SeriesDecomp(decomp_kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, cross_memory):
        # Self auto-correlation block
        new_x = self.self_autocorrelation(x, x, x)
        res = x + self.dropout(new_x)
        seasonal_part, trend_part1 = self.decomp1(res)
        seasonal_part = self.norm1(seasonal_part)

        # Cross auto-correlation block
        new_seasonal = self.cross_autocorrelation(seasonal_part, cross_memory, cross_memory)
        res_seasonal = seasonal_part + self.dropout(new_seasonal)
        seasonal_part, trend_part2 = self.decomp2(res_seasonal)
        seasonal_part = self.norm2(seasonal_part)

        # Feed-forward block
        new_seasonal = self.feed_forward(seasonal_part)
        res_seasonal = seasonal_part + self.dropout(new_seasonal)
        seasonal_part, trend_part3 = self.decomp3(res_seasonal)
        seasonal_part = self.norm3(seasonal_part)

        # Accumulate the trend parts from each decomposition block
        total_trend = trend_part1 + trend_part2 + trend_part3
        return seasonal_part, total_trend


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(layers[0].d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(layers[0].self_autocorrelation.d_model)

    def forward(self, x, cross_memory, trend_init):
        for layer in self.layers:
            x, new_trend = layer(x, cross_memory)
            # Progressive trend accumulation
            trend_init = trend_init + new_trend
        
        x = self.norm(x)
        return x, trend_init


# 3. Main Autoformer Model (Optimized)
class Autoformer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, n_heads, d_ff, decomp_kernel_size, 
                 num_encoder_layers, num_decoder_layers, input_features, dropout=0.1, top_k_factor=5):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Input embedding
        self.input_embedding = nn.Linear(input_features, d_model)
        
        # Decomposition Layer
        self.decomp_init = SeriesDecomp(decomp_kernel_size)

        # Encoder
        self.encoder = Encoder([
            EncoderLayer(d_model, n_heads, d_ff, decomp_kernel_size, dropout, top_k_factor)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder = Decoder([
            DecoderLayer(d_model, n_heads, d_ff, decomp_kernel_size, dropout, top_k_factor)
            for _ in range(num_decoder_layers)
        ])

        # Final projection layers
        self.seasonal_projection = nn.Linear(d_model, input_features)
        self.trend_projection = nn.Linear(d_model, input_features)

    def forward(self, x_enc):
        # x_enc shape: [Batch, Seq_Len, Features]
        
        # --- Prepare Decoder Inputs ---
        # As per the paper, the decoder input is a concatenation of the latter half
        # of the encoder input and a placeholder for the prediction sequence.
        
        # 1. Decompose the encoder input to get initial trend and seasonal
        seasonal_init, trend_init = self.decomp_init(x_enc)
        
        # 2. Create decoder's initial seasonal input
        seasonal_decoder_input = torch.cat(
            [seasonal_init[:, -self.seq_len//2:, :], torch.zeros_like(seasonal_init[:, :self.pred_len, :])],
            dim=1
        )
        # 3. Create decoder's initial trend input
        trend_decoder_input = torch.cat(
            [trend_init[:, -self.seq_len//2:, :], torch.mean(x_enc, dim=1, keepdim=True).repeat(1, self.pred_len, 1)],
            dim=1
        )

        # --- Embed all inputs ---
        enc_out = self.input_embedding(x_enc)
        dec_in_seasonal = self.input_embedding(seasonal_decoder_input)
        dec_in_trend = self.input_embedding(trend_decoder_input)

        # --- Encoder Forward Pass ---
        enc_out = self.encoder(enc_out)

        # --- Decoder Forward Pass ---
        final_seasonal, final_trend = self.decoder(dec_in_seasonal, enc_out, dec_in_trend)

        # --- Final Prediction ---
        # Project back to the original feature dimension
        pred_seasonal = self.seasonal_projection(final_seasonal)
        pred_trend = self.trend_projection(final_trend)
        
        # Combine seasonal and trend for the final result
        prediction = pred_seasonal + pred_trend

        # Return only the prediction part
        return prediction[:, -self.pred_len:, :]


# 4. Training and Evaluation Script (Your original script, minimally adapted)
def Autoformer_model(train_x_df, train_y_df, test_x_df, test_y_df, args):
    device = torch.device('cuda' if args['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
  
    # Convert pandas to numpy
    train_x = train_x_df.values
    train_y = train_y_df.values
    test_x = test_x_df.values
    test_y = test_y_df.values
    
    # Create dataset loaders. The model expects [Batch, Seq_Len, Features]
    X_train = torch.from_numpy(train_x).unsqueeze(0).float().to(device)
    Y_train = torch.from_numpy(train_y).unsqueeze(0).float().to(device)
    X_test = torch.from_numpy(test_x).unsqueeze(0).float().to(device)
    Y_test = torch.from_numpy(test_y).unsqueeze(0).float().to(device)

    print(f"Train X shape: {X_train.shape}")
    print(f"Test X shape: {X_test.shape}")
    
    # Instantiate the model
    model = Autoformer(
        seq_len=X_train.shape[1],
        pred_len=X_train.shape[1],
        input_features=args['features'],
        n_heads=args['n_heads'],
        d_ff=args['d_ff'],
        decomp_kernel_size=args['decomp_kernel_size'],
        num_encoder_layers=args['num_encoder_layers'],
        num_decoder_layers=args['num_decoder_layers'],
        dropout=args['dropout'],
        top_k_factor=args['top_k_factor']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()

    # --- Training Loop ---
    for epoch in range(1, args['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train)
        loss = criterion(output, Y_train)
        
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d} | Loss: {loss.item():.6f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, Y_test)
        print(f"\nTest MSE Loss: {test_loss.item():.6f}")

    # Extract results
    fit_result = None # In this simple script, we don't store in-sample predictions
    ypredt = Y_test.squeeze(0).cpu().numpy()
    ypred = test_output.squeeze(0).cpu().numpy()

    return fit_result, ypredt, ypred

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AuxiliaryEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

class LearnedEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, num_people: int = 1000, dropout: float = 0.1, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        # We use seq_len+1 so that indices from 0..seq_len are valid.
        self.seq_encoding = nn.Embedding(seq_len + 1, d_model, max_norm=True).to(device)  
        self.person_encoding = nn.Embedding(num_people, d_model, max_norm=True).to(device)

        print("seq_len:", seq_len, "Embedding size:", self.seq_encoding.weight.shape[0])

    def forward(self, x: torch.Tensor, num_people=1, type='traj') -> torch.Tensor:
        # x is expected to have shape: (B, seq_len, num_people, d_model)
        seq_len = x.size(1)
        d_model = x.size(-1)
        if type == 'traj':
            # Create sequence indices and clamp them if needed
            max_seq_len = self.seq_encoding.num_embeddings
            seq_indices = torch.arange(seq_len).to(self.device)
            seq_indices = torch.clamp(seq_indices, max=max_seq_len - 1)
            
            # Get the full sequence encoding: shape (seq_len, d_model)
            seq_enc_full = self.seq_encoding(seq_indices)
            # Take the first half of the channels for sequence encoding
            seq_enc = seq_enc_full[..., : d_model // 2]  # shape: (seq_len, d_model/2)
            # Expand to match (B, seq_len, num_people, d_model/2)
            seq_enc = seq_enc.unsqueeze(0).unsqueeze(2).expand(x.size(0), seq_len, num_people, -1)
            
            # Similarly, get person encoding for each person.
            people_indices = torch.arange(num_people).to(self.device)
            person_enc_full = self.person_encoding(people_indices)
            person_enc = person_enc_full[..., : d_model // 2]  # shape: (num_people, d_model/2)
            # Expand to match (B, seq_len, num_people, d_model/2)
            person_enc = person_enc.unsqueeze(0).unsqueeze(0).expand(x.size(0), seq_len, num_people, -1)
            
            # Now add the encodings to x:
            # Assume x’s last dimension is split: first half for seq, second half for person.
            if x.shape[-1] != d_model:
                raise ValueError("Mismatch in feature dimension")
            x[..., : d_model // 2] += seq_enc
            x[..., d_model // 2:] += person_enc
        else:
            # If not a 'traj' type, apply only the sequence encoding to all channels.
            seq_indices = torch.arange(seq_len).to(self.device)
            max_seq_len = self.seq_encoding.num_embeddings
            seq_indices = torch.clamp(seq_indices, max=max_seq_len - 1)
            seq_enc = self.seq_encoding(seq_indices).unsqueeze(0).unsqueeze(2).expand(x.size(0), seq_len, num_people, -1)
            x += seq_enc

        return self.dropout(x)

class TransMotion(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, 
                 activation='relu', output_scale=1, obs_and_pred=21, num_tokens=47, device='cuda:0'):
        super().__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.device = device
        
        self.fc_in_traj = nn.Linear(2, nhid)
        self.fc_out_traj = nn.Linear(nhid, 2)
        self.traj_encoder = LearnedEncoding(nhid, seq_len=21, device=device)

        self.fc_in_2dbb = nn.Linear(4, nhid)
        self.bb2d_encoder = LearnedEncoding(nhid, seq_len=9, device=device)

        self.fc_in_3dbb = nn.Linear(4, nhid)
        self.bb3d_encoder = LearnedEncoding(nhid, seq_len=9, device=device)

        self.fc_in_2dpose = nn.Linear(2, nhid)
        self.pose2d_encoder = LearnedEncoding(nhid, seq_len=198, device=device)

        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = LearnedEncoding(nhid, seq_len=198, device=device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
        self.local_encoder = AuxiliaryEncoder(encoder_layer, num_layers=nlayers_local)
        self.global_encoder = AuxiliaryEncoder(encoder_layer, num_layers=nlayers_global)
        
    def forward(self, tgt, padding_mask, metamask=None):
        # Unpack tgt: shape = [B, in_F, NJ, K]
        B, in_F, NJ, K = tgt.shape
        # Ensure padding_mask is a tensor.
        if not isinstance(padding_mask, torch.Tensor):
            padding_mask = torch.zeros(B, self.seq_len, dtype=torch.bool, device=self.device)
    
        # Define variables for clarity.
        FSL, J = self.seq_len, self.token_num  # FSL = target sequence length, J = tokens per person/modality.
        out_F = FSL - in_F

        # Create an index array to extend the input along the frame dimension.
        idx = np.append(np.arange(0, in_F), np.repeat([in_F - 1], out_F))
        tgt = tgt[:, idx]  # Now tgt shape becomes [B, FSL, NJ, K]
    
        # Trim tokens dimension to be divisible by J.
        new_NJ = (tgt.shape[2] // J) * J
        tgt = tgt[:, :, :new_NJ, :]
        N = new_NJ // J  # N = number of groups per sample.
    
        # Reshape tgt to have dimensions [B, FSL, N, J, K].
        tgt = tgt.reshape(B, FSL, N, J, K)

        # Process each modality:
        tgt_traj = self.fc_in_traj(tgt[:, :, :, 0, :2])
        tgt_traj = self.traj_encoder(tgt_traj, num_people=N, type='traj')

        tgt_2dbb = self.fc_in_2dbb(tgt[:, :, :, 1, :4])
        tgt_2dbb = self.bb2d_encoder(tgt_2dbb)
    
        tgt_3dbb = self.fc_in_3dbb(tgt[:, :, :, 2, :4])
        tgt_3dbb = self.bb3d_encoder(tgt_3dbb)

        tgt_2dpose = self.fc_in_2dpose(tgt[:, :, :, 3, :2])
        tgt_2dpose = self.pose2d_encoder(tgt_2dpose)

        tgt_3dpose = self.fc_in_3dpose(tgt[:, :, :, 4, :3])
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

        # Concatenate along modality dimension.
        tgt = torch.cat((tgt_traj, tgt_2dbb, tgt_3dbb, tgt_2dpose, tgt_3dpose), dim=2)
        # After concatenation, tgt is of shape [B, FSL, N*5, nhid].
        tgt = tgt.reshape(FSL, -1, self.nhid)  # Transformer expects input shape (seq_len, batch, hidden)
    
        # --- Process padding_mask ---
        # At this point, padding_mask was originally of shape [B, original_seq_len].
        # Adjust it so that its length equals FSL (self.seq_len).
        if padding_mask.size(1) < self.seq_len:
            pad_size = self.seq_len - padding_mask.size(1)
            padding_mask = F.pad(padding_mask, (0, pad_size), value=True)
        elif padding_mask.size(1) > self.seq_len:
            padding_mask = padding_mask[:, :self.seq_len]
    
        # For both local and global encoders, the effective batch size is B * N * num_modalities.
        num_modalities = self.token_num
        effective_bs = B * N * num_modalities  # expected to be 120, for example.
    
        # Build local key-padding mask:
        tgt_padding_mask_local = padding_mask.unsqueeze(1).unsqueeze(1)  \
                              .expand(B, N, num_modalities, self.seq_len)  \
                              .reshape(effective_bs, self.seq_len)
    
        # Pass through the local encoder.
        out_local = self.local_encoder(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        out_local = out_local * self.output_scale + tgt
    
        # Build global key-padding mask the same way.
        tgt_padding_mask_global = padding_mask.unsqueeze(1).unsqueeze(1)  \
                               .expand(B, N, num_modalities, self.seq_len)  \
                               .reshape(effective_bs, self.seq_len)
    
        out_global = self.global_encoder(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)
        out_global = out_global * self.output_scale + out_local
        out_global_reshaped = out_global.reshape(FSL, B, N, num_modalities, self.nhid)  # shape: (30, 4, 6, 5, 128)

        out_primary = out_global_reshaped[:, :, 0, 0, :]  # shape: (FSL, B, nhid) = (30, 4, 128)

        # Pass through the final linear layer (fc_out_traj maps nhid->2)
        out_primary = self.fc_out_traj(out_primary)  # shape: (30, 4, 2)

        # Transpose to (B, FSL, 2) and unsqueeze to match desired shape (B, FSL, 1, 2)
        return out_primary.transpose(0, 1).unsqueeze(2)  # final shape: (4, 30, 1, 2)
    
def create_model(config, logger):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid = config["MODEL"]["dim_hidden"]
    nhead = config["MODEL"]["num_heads"]
    nlayers_local = config["MODEL"]["num_layers_local"]
    nlayers_global = config["MODEL"]["num_layers_global"]
    dim_feedforward = config["MODEL"]["dim_feedforward"]

    logger.info("Creating merged TransMotion model.")
    model = TransMotion(tok_dim=seq_len, nhid=nhid, nhead=nhead, dim_feedfwd=dim_feedforward, 
                        nlayers_local=nlayers_local, nlayers_global=nlayers_global, 
                        output_scale=config["MODEL"]["output_scale"], 
                        obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"], 
                        num_tokens=token_num, device=config["DEVICE"]).to(config["DEVICE"]).float()
    return model

import pandas as pd
import numpy as np
import torch
from scipy import stats
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.utils import negative_sampling
from torch_geometric_temporal import recurrent
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, DynamicHeteroGraphTemporalSignal
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn import HeteroGCLSTM
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import GATConv, HeteroConv


class CLVModel(torch.nn.Module):
    def __init__(self, country_embedding, description_embedding, metadata):
        super().__init__()
        self.country_embed = country_embedding
        self.desc_embed = description_embedding
        self.metadata = metadata
        
        # Calculate input dimensions after concatenation with embeddings
        customer_in_dim = 14 + 4      # 5 original features + 4 country embedding
        product_in_dim = 4 + 16   # 2 original features + description embedding
        
        # GCLSTM module
        self.gclstm = HeteroGCLSTM(
            in_channels_dict={
                'customer': customer_in_dim,
                'product': product_in_dim
            },
            out_channels=64,
            metadata=self.metadata
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_dict, edge_index_dict, h_dict = None, c_dict = None):
        # Process customer features
        customer_feats = x_dict['customer']
        country_ids = customer_feats[:, -1].long()  # Country index column
        customer_embs = self.country_embed(country_ids)
        customer_x = torch.cat([customer_feats[:, :-1], customer_embs], dim=1)
        
        # Process product features
        product_feats = x_dict['product']
        desc_ids = product_feats[:, -1].long()  # Description index column
        product_embs = self.desc_embed(desc_ids)
        product_x = torch.cat([product_feats[:, :-1], product_embs], dim=1)
        
        # Create processed feature dict
        processed_x = {'customer': customer_x, 'product': product_x}
        
        # Run through GCLSTM
        h_dict, c_dict = self.gclstm(processed_x, edge_index_dict, h_dict, c_dict)
        
        # Predict CLV from customer hidden states
        return self.predictor(h_dict['customer']), h_dict, c_dict
    

class CLVModelAdjusted(torch.nn.Module):
    def __init__(self, country_embedding, description_embedding, metadata, num_layers=1, heads = 4):
        super().__init__()
        self.country_embed = country_embedding
        self.desc_embed = description_embedding
        self.metadata = metadata
        
        # Calculate input dimensions after concatenation with embeddings
        customer_in_dim = 14 + 4      # 5 original features + 4 country embedding
        product_in_dim = 4 + 16   # 2 original features + description embedding
        
        self.gnn_layers = nn.ModuleList()
        self.norm_customer_layers = nn.ModuleList()
        self.norm_product_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(HeteroGCLSTM(
            in_channels_dict={
                'customer': customer_in_dim if i == 0 else 64,
                'product': product_in_dim if i == 0 else 64
            },
            out_channels=64,
            metadata=self.metadata
        ))
            self.norm_customer_layers.append(GraphNorm(64))
            self.norm_product_layers.append(GraphNorm(64))
            #self.gnn_layers.append(GCNNorm())
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_dict, edge_index_dict, h_dict = None, c_dict = None):
        # Process customer features
        customer_feats = x_dict['customer']
        country_ids = customer_feats[:, -1].long()  # Country index column
        customer_embs = self.country_embed(country_ids)
        customer_x = torch.cat([customer_feats[:, :-1], customer_embs], dim=1)
        
        # Process product features
        product_feats = x_dict['product']
        desc_ids = product_feats[:, -1].long()  # Description index column
        product_embs = self.desc_embed(desc_ids)
        product_x = torch.cat([product_feats[:, :-1], product_embs], dim=1)
        
        # Create processed feature dict
        processed_x = {'customer': customer_x, 'product': product_x}
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_dict, c_dict = gnn_layer(processed_x, edge_index_dict, h_dict, c_dict)
            h_dict['customer'] = self.norm_customer_layers[i](h_dict['customer'])
            h_dict['product'] = self.norm_product_layers[i](h_dict['product'])
            processed_x = h_dict
        

        predictions = self.predictor(processed_x['customer'])
        
        # Predict CLV from customer hidden states
        return predictions, h_dict, c_dict
    
class CLVModelAttention(torch.nn.Module):
    def __init__(self, country_embedding, description_embedding, metadata,hidden_dim = 64,  num_layers=2, heads = 4):
        super().__init__()
        self.country_embed = country_embedding
        self.desc_embed = description_embedding
        self.metadata = metadata
        
        # Calculate input dimensions after concatenation with embeddings
        customer_in_dim = 14 + 4      # 5 original features + 4 country embedding
        product_in_dim = 4 + 16   # 2 original features + description embedding
        
        self.gnn_layers = nn.ModuleList()
        self.norm_customer_layers = nn.ModuleList()
        self.norm_product_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(HeteroGCLSTM(
            in_channels_dict={
                'customer': customer_in_dim if i == 0 else hidden_dim,
                'product': product_in_dim if i == 0 else hidden_dim
            },
            out_channels=hidden_dim,
            metadata=self.metadata
        ))
            self.norm_customer_layers.append(GraphNorm(hidden_dim))
            self.norm_product_layers.append(GraphNorm(hidden_dim))
            #self.gnn_layers.append(GCNNorm())
        # Prediction head
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x_dict, edge_index_dict, h_dict = None, c_dict = None):
        # Process customer features
        customer_feats = x_dict['customer']
        country_ids = customer_feats[:, -1].long()  # Country index column
        customer_embs = self.country_embed(country_ids)
        customer_x = torch.cat([customer_feats[:, :-1], customer_embs], dim=1)
        
        # Process product features
        product_feats = x_dict['product']
        desc_ids = product_feats[:, -1].long()  # Description index column
        product_embs = self.desc_embed(desc_ids)
        product_x = torch.cat([product_feats[:, :-1], product_embs], dim=1)
        
        # Create processed feature dict
        processed_x = {'customer': customer_x, 'product': product_x}
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_dict, c_dict = gnn_layer(processed_x, edge_index_dict, h_dict, c_dict)
            h_dict['customer'] = self.norm_customer_layers[i](h_dict['customer'])
            h_dict['product'] = self.norm_product_layers[i](h_dict['product'])
            processed_x = h_dict
        customer_repr = processed_x['customer']
        # Apply attention mechanism
        customer_sequence = customer_repr.unsqueeze(1)
        customer_sequence = customer_sequence.transpose(0, 1)  # Shape: (1, batch_size, features)
        attn_output, _ = self.attention_layer(customer_sequence, customer_sequence, customer_sequence)
        attn_output = attn_output.transpose(0, 1).squeeze(1)  # Shape: (batch_size, 1, features)
        customer_attended = customer_repr + attn_output  # Residual connection

        predictions = self.predictor(customer_attended)
        # Predict CLV from customer hidden states
        return predictions, h_dict, c_dict
    
class CLVModelAttention2(torch.nn.Module):
    def __init__(self, country_embedding, description_embedding, metadata,hidden_dim = 64,  num_layers=2, heads = 4):
        super().__init__()
        self.country_embed = country_embedding
        self.desc_embed = description_embedding
        self.metadata = metadata
        
        # Calculate input dimensions after concatenation with embeddings
        customer_in_dim = 14 + 4      # 5 original features + 4 country embedding
        product_in_dim = 4 + 16   # 2 original features + description embedding
        
        self.gnn_layers = nn.ModuleList()
        self.norm_customer_layers = nn.ModuleList()
        self.norm_product_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(HeteroGCLSTM(
            in_channels_dict={
                'customer': customer_in_dim if i == 0 else hidden_dim,
                'product': product_in_dim if i == 0 else hidden_dim
            },
            out_channels=hidden_dim,
            metadata=self.metadata
        ))
            #self.norm_customer_layers.append(GraphNorm(hidden_dim))
            #self.norm_product_layers.append(GraphNorm(hidden_dim))
            #self.gnn_layers.append(GCNNorm())
        # Prediction head
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x_dict, edge_index_dict, h_dict = None, c_dict = None):
        # Process customer features
        customer_feats = x_dict['customer']
        country_ids = customer_feats[:, -1].long()  # Country index column
        customer_embs = self.country_embed(country_ids)
        customer_x = torch.cat([customer_feats[:, :-1], customer_embs], dim=1)
        
        # Process product features
        product_feats = x_dict['product']
        desc_ids = product_feats[:, -1].long()  # Description index column
        product_embs = self.desc_embed(desc_ids)
        product_x = torch.cat([product_feats[:, :-1], product_embs], dim=1)
        
        # Create processed feature dict
        processed_x = {'customer': customer_x, 'product': product_x}
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_dict, c_dict = gnn_layer(processed_x, edge_index_dict, h_dict, c_dict)
            #h_dict['customer'] = self.norm_customer_layers[i](h_dict['customer'])
            #h_dict['product'] = self.norm_product_layers[i](h_dict['product'])
            processed_x = h_dict
        customer_repr = processed_x['customer']
        # Apply attention mechanism
        customer_sequence = customer_repr.unsqueeze(1)
        customer_sequence = customer_sequence.transpose(0, 1)  # Shape: (1, batch_size, features)
        attn_output, _ = self.attention_layer(customer_sequence, customer_sequence, customer_sequence)
        attn_output = attn_output.transpose(0, 1).squeeze(1)  # Shape: (batch_size, 1, features)
        customer_attended = customer_repr + attn_output  # Residual connection

        predictions = self.predictor(customer_attended)
        # Predict CLV from customer hidden states
        return predictions, h_dict, c_dict  
    

class TGNSelfSupervised(torch.nn.Module):
    def __init__(self, country_embedding, description_embedding, metadata,hidden_dim = 128,  num_layers=1, heads = 1):
        super().__init__()
        self.country_embed = country_embedding
        self.desc_embed = description_embedding
        self.metadata = metadata
        
        # Calculate input dimensions after concatenation with embeddings
        customer_in_dim = 14 + 4      # 5 original features + 4 country embedding
        product_in_dim = 4 + 16   # 2 original features + description embedding
        
        self.gnn_layers = nn.ModuleList()
        self.norm_customer_layers = nn.ModuleList()
        self.norm_product_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(HeteroGCLSTM(
            in_channels_dict={
                'customer': customer_in_dim if i == 0 else hidden_dim,
                'product': product_in_dim if i == 0 else hidden_dim
            },
            out_channels=hidden_dim,
            metadata=self.metadata
        ))
            self.norm_customer_layers.append(GraphNorm(hidden_dim))
            self.norm_product_layers.append(GraphNorm(hidden_dim))
            #self.gnn_layers.append(GCNNorm())
        # Prediction head
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)

    def forward(self, x_dict, edge_index_dict, h_dict = None, c_dict = None):
        # Process customer features
        customer_feats = x_dict['customer']
        country_ids = customer_feats[:, -1].long()  # Country index column
        customer_embs = self.country_embed(country_ids)
        customer_x = torch.cat([customer_feats[:, :-1], customer_embs], dim=1)
        
        # Process product features
        product_feats = x_dict['product']
        desc_ids = product_feats[:, -1].long()  # Description index column
        product_embs = self.desc_embed(desc_ids)
        product_x = torch.cat([product_feats[:, :-1], product_embs], dim=1)
        
        # Create processed feature dict
        processed_x = {'customer': customer_x, 'product': product_x}
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_dict, c_dict = gnn_layer(processed_x, edge_index_dict, h_dict, c_dict)
            h_dict['customer'] = self.norm_customer_layers[i](h_dict['customer'])
            h_dict['product'] = self.norm_product_layers[i](h_dict['product'])
            processed_x = h_dict
        customer_repr = processed_x['customer']
        # Apply attention mechanism
        customer_sequence = customer_repr.unsqueeze(1)
        customer_sequence = customer_sequence.transpose(0, 1)  # Shape: (1, batch_size, features)
        attn_output, _ = self.attention_layer(customer_sequence, customer_sequence, customer_sequence)
        attn_output = attn_output.transpose(0, 1).squeeze(1)  # Shape: (batch_size, 1, features)
        h_dict['customer'] = customer_repr + attn_output  # Residual connection
        return  h_dict, c_dict
    

class TGNSelfSupervisedLP(torch.nn.Module):
    def __init__(self, country_embedding, description_embedding, metadata, hidden_dim=64, num_layers=2, heads=4):
        super().__init__()
        self.country_embed = country_embedding
        self.desc_embed = description_embedding
        self.metadata = metadata
        
        # Input dimensions after concatenation with embeddings
        # These are based on your original calculation.
        # Ensure x_dict['customer'][:, :-1] yields 7 features and x_dict['product'][:, :-1] yields 2 features.
        customer_in_dim = 14 + 4      # 7 original features + 4 country embedding
        product_in_dim = 4 + 16      # 2 original features + 16 description embedding
        
        self.gnn_layers = nn.ModuleList()
        self.norm_customer_layers = nn.ModuleList()
        self.norm_product_layers = nn.ModuleList()
        for i in range(num_layers):
            current_in_channels_dict = {
                'customer': customer_in_dim if i == 0 else hidden_dim,
                'product': product_in_dim if i == 0 else hidden_dim
            }
            self.gnn_layers.append(HeteroGCLSTM(
                in_channels_dict=current_in_channels_dict,
                out_channels=hidden_dim,
                metadata=self.metadata
            ))
            self.norm_customer_layers.append(GraphNorm(hidden_dim))
            self.norm_product_layers.append(GraphNorm(hidden_dim))
            
        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=heads)

    def forward(self, x_dict, edge_index_dict, h_dict_prev_ts=None, c_dict_prev_ts=None, 
                  target_edge_index=None, neg_edge_index_src=None, neg_edge_index_dst=None):
        
        # --- 1. Feature Preparation ---
        customer_feats = x_dict['customer']
        country_ids = customer_feats[:, -1].long()
        customer_embs = self.country_embed(country_ids)
        # Assuming customer_feats[:, :-1] correctly gives 7 features
        customer_x = torch.cat([customer_feats[:, :-1], customer_embs], dim=1) 
        
        product_feats = x_dict['product']
        desc_ids = product_feats[:, -1].long()
        product_embs = self.desc_embed(desc_ids)
        # Assuming product_feats[:, :-1] correctly gives 2 features
        product_x = torch.cat([product_feats[:, :-1], product_embs], dim=1)
        
        input_node_features = {'customer': customer_x, 'product': product_x}
        
        # --- 2. GNN Layers (Temporal Propagation) ---
        # h_state_for_layers and c_state_for_layers are the LSTM states passed through the stack of GNN layers.
        # They are initialized from the previous time step's states (h_dict_prev_ts, c_dict_prev_ts).
        current_h_for_stack = h_dict_prev_ts
        current_c_for_stack = c_dict_prev_ts
        
        # features_for_current_layer is what's passed as 'x' to each GNN layer in the stack
        features_for_current_layer = input_node_features

        for i, gnn_layer in enumerate(self.gnn_layers):
            # The gnn_layer (HeteroGCLSTM) takes current features and (h,c) states.
            # It outputs new (h,c) states. The new 'h' is also the node embedding output.
            output_embeddings_gnn, next_c_state_gnn = gnn_layer(
                features_for_current_layer, 
                edge_index_dict, 
                current_h_for_stack, # h state input to LSTM cells
                current_c_for_stack  # c state input to LSTM cells
            )
            
            output_embeddings_gnn['customer'] = self.norm_customer_layers[i](output_embeddings_gnn['customer'])
            output_embeddings_gnn['product'] = self.norm_product_layers[i](output_embeddings_gnn['product'])
            
            # Output embeddings of this layer become input features for the next GNN layer
            features_for_current_layer = output_embeddings_gnn
            
            # The (h,c) states are updated and passed to the next GNN layer in the stack
            current_h_for_stack = output_embeddings_gnn 
            current_c_for_stack = next_c_state_gnn

        # final_node_embeddings are the embeddings after all GNN layers for this time step
        final_node_embeddings = features_for_current_layer 
        # final_c_state_ts is the cell state to be passed to the next time step
        final_c_state_ts = current_c_for_stack
        # final_h_state_ts (hidden state for next time step) is final_node_embeddings

        # --- 3. Attention Mechanism (on customer nodes) ---
        customer_repr = final_node_embeddings['customer']
        # Unsqueeze and transpose for MultiheadAttention: (seq_len, batch_size, features)
        # Here, seq_len is 1 as we process each node independently in terms of attention sequence.
        # Batch_size is num_customers.
        customer_sequence = customer_repr.unsqueeze(0) # (1, num_customers, features) - if treating nodes as batch
                                                       # or (num_customers, 1, features) then transpose(0,1)
                                                       # Original was .unsqueeze(1).transpose(0,1) implies (seq_len=num_nodes, batch=1, features)
                                                       # This is a common way if batch_first=False
        
        # Assuming (num_nodes, embed_dim) -> unsqueeze(1) -> (num_nodes, 1, embed_dim)
        # -> transpose(0,1) -> (1, num_nodes, embed_dim) [seq_len=1, batch=num_nodes]
        customer_sequence_for_attn = customer_repr.unsqueeze(1).transpose(0,1)

        attn_output, _ = self.attention_layer(customer_sequence_for_attn, customer_sequence_for_attn, customer_sequence_for_attn)
        attn_output = attn_output.transpose(0, 1).squeeze(1) # Back to (num_nodes, features)
        
        # Store final embeddings (H state for next time step)
        # Make a copy before modifying 'customer' entry if final_node_embeddings is used elsewhere
        output_h_dict_for_next_ts = final_node_embeddings.copy() 
        output_h_dict_for_next_ts['customer'] = customer_repr + attn_output # Residual connection

        # --- 4. Link Prediction Scoring (if target edges are provided) ---
        pos_score = None
        neg_score = None

        # Use the embeddings *after* attention for link prediction involving customers
        current_customer_embeddings = output_h_dict_for_next_ts['customer']
        current_product_embeddings = output_h_dict_for_next_ts['product'] # Assuming product embeddings are not changed by this attention

        if target_edge_index is not None and target_edge_index.numel() > 0:
            pos_src_nodes = target_edge_index[0]
            pos_dst_nodes = target_edge_index[1]
            # Ensure indices are within bounds
            if pos_src_nodes.max() < current_customer_embeddings.size(0) and \
               pos_dst_nodes.max() < current_product_embeddings.size(0):
                pos_score = torch.sum(current_customer_embeddings[pos_src_nodes] * current_product_embeddings[pos_dst_nodes], dim=-1)
            else:
                # Handle index out of bounds - this indicates an issue with data or node mapping
                print("Warning: Positive edge index out of bounds.")


        if neg_edge_index_src is not None and neg_edge_index_dst is not None and \
           neg_edge_index_src.numel() > 0 and neg_edge_index_dst.numel() > 0:
            # Ensure indices are within bounds
            if neg_edge_index_src.max() < current_customer_embeddings.size(0) and \
               neg_edge_index_dst.max() < current_product_embeddings.size(0):
                neg_score = torch.sum(current_customer_embeddings[neg_edge_index_src] * current_product_embeddings[neg_edge_index_dst], dim=-1)
            else:
                print("Warning: Negative edge index out of bounds.")
                
        return output_h_dict_for_next_ts, final_c_state_ts, pos_score, neg_score
    

def train_link_prediction(model, signal, epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    best_loss = float('inf')
    no_improvement = 0
    
    # These states are carried over snapshots within an epoch
    # And re-initialized for each new epoch for simplicity,
    # unless your 'signal' implies one continuous sequence across epochs.
    final_epoch_h_state, final_epoch_c_state = None, None 

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_loss_calculations = 0
        
        # Initialize LSTM states at the beginning of each epoch
        current_h_state, current_c_state = None, None

        try: 
            for t, snapshot in enumerate(signal):
                x_dict = snapshot.x_dict
                edge_index_dict = snapshot.edge_index_dict

                # Skip snapshot if essential node types are missing features
                if 'customer' not in x_dict or 'product' not in x_dict or \
                   x_dict['customer'].numel() == 0 or x_dict['product'].numel() == 0:
                    # print(f"Snapshot {t}: Missing customer or product features. Advancing states only.")
                    # Even if no prediction, pass through model to update states if nodes exist
                    # This part needs careful handling: if x_dict is truly empty for a type,
                    # model forward might fail. Assuming HeteroGCLSTM can handle empty inputs gracefully
                    # or x_dict always has at least dummy tensors. For now, let's assume valid x_dict.
                    # If we can't run the model, we can't update states.
                    # A robust way is to ensure x_dict always has tensors, even if 0-sized for some types.
                    # If model cannot run, prev states are carried.
                    # We will proceed assuming model can run or we skip if critical data missing.
                    # For this version, if critical x_dict entries are missing, skip state update too.
                    continue


                pos_edge_index, neg_src_indices, neg_dst_indices = None, None, None
                
                # Prepare positive edges for the link prediction task (e.g., 'purchases')
                target_rel = ('customer', 'purchases', 'product')
                if target_rel in snapshot.edge_index_dict and snapshot.edge_index_dict[target_rel].numel() > 0:
                    pos_edge_index = snapshot.edge_index_dict[target_rel]
                    num_pos = pos_edge_index.size(1)
                    
                    num_customers = x_dict['customer'].size(0)
                    num_products = x_dict['product'].size(0)

                    if num_customers > 0 and num_products > 0: # Ensure nodes exist to sample from
                        neg_src_indices = torch.randint(0, num_customers, (num_pos,), device=x_dict['customer'].device)
                        neg_dst_indices = torch.randint(0, num_products, (num_pos,), device=x_dict['customer'].device)
                        # TODO: Optionally, filter out true positives from negative samples for stricter training.
                    else: # No nodes to sample negatives from, cannot do link prediction for this snapshot
                        pos_edge_index = None # Cannot perform prediction without possibility of negatives
                
                # Forward pass: model returns final embeddings (h_state), c_state, and scores
                next_h_state, next_c_state, pos_score, neg_score = model(
                    x_dict, 
                    edge_index_dict, 
                    current_h_state, 
                    current_c_state,
                    target_edge_index=pos_edge_index,
                    neg_edge_index_src=neg_src_indices,
                    neg_edge_index_dst=neg_dst_indices
                )
                
                # Update LSTM states for the next snapshot
                # Detach them to prevent gradients from flowing back across snapshots
                if next_h_state is not None:
                    current_h_state = {k: v.detach() for k, v in next_h_state.items()}
                if next_c_state is not None:
                    current_c_state = {k: v.detach() for k, v in next_c_state.items()}

                # Calculate loss only if scores were computed
                if pos_score is not None and neg_score is not None:
                    scores = torch.cat([pos_score, neg_score], dim=0)
                    labels = torch.cat([
                        torch.ones_like(pos_score), 
                        torch.zeros_like(neg_score)
                    ], dim=0)

                    loss = F.binary_cross_entropy_with_logits(scores, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_loss_calculations += 1
                elif pos_edge_index is not None: # Positive edges existed but scores weren't generated (e.g. OOB)
                    # print(f"Snapshot {t}: Positive edges existed but scores not generated. Skipping loss.")
                    pass


            epoch_loss = total_loss / num_loss_calculations if num_loss_calculations > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f} (computed over {num_loss_calculations} snapshots)")

            if epoch_loss < best_loss and num_loss_calculations > 0 : # Ensure actual training happened
                best_loss = epoch_loss
                no_improvement = 0
                # You might want to save the model here:
                # torch.save(model.state_dict(), 'best_link_prediction_model.pth')
            elif num_loss_calculations > 0: # Only increment if there were attempts to train
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stopping...")
                    break
            elif num_loss_calculations == 0 and epoch > patience : # If no training happens for several epochs
                print("No training steps for several epochs, check data or model. Stopping.")
                break
            
            final_epoch_h_state, final_epoch_c_state = current_h_state, current_c_state

        except RuntimeError as e:
            print(f"Runtime error during epoch {epoch+1}: {e}")
            # Your original error reporting for tensor shapes
            print("Checking tensors for issue at error point:")
            for node_type_key, features_val in x_dict.items(): # Use different var names
                print(f"{node_type_key} features shape: {features_val.shape}")
            
            for edge_type_key, indices_val in edge_index_dict.items():
                src, rel, dst = edge_type_key
                print(f"Edge {edge_type_key} shape: {indices_val.shape}")
                
                if indices_val.numel() > 0 and indices_val.shape[1] > 0 : # Check if not empty
                    max_src = indices_val[0].max().item()
                    max_dst = indices_val[1].max().item()
                    print(f"  - Max source index ({src}): {max_src}")
                    print(f"  - Max target index ({dst}): {max_dst}")
                    print(f"  - {src} feature dim: {x_dict[src].shape[0] if src in x_dict else 'N/A'}")
                    print(f"  - {dst} feature dim: {x_dict[dst].shape[0] if dst in x_dict else 'N/A'}")
            raise
            
    # Return the trained model and the LSTM states from the end of the last processed epoch
    return model, final_epoch_h_state, final_epoch_c_state


def train(model, signal, epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.SmoothL1Loss(beta=0.5) 
    best_loss = float('inf')
    no_improvement = 0
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        h_dict, c_dict = None, None
        try: 
            for t, snapshot in enumerate(signal):
                # Get data from temporal snapshot
                x_dict = snapshot.x_dict
                edge_index_dict = snapshot.edge_index_dict
                targets = snapshot['customer'].y
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                # Forward pass
                predictions, h_dict, c_dict = model(
                    x_dict, 
                    edge_index_dict, 
                    h_dict, 
                    c_dict
                )
                active_customer_ids = snapshot['active_customer_ids']['y']
                predictions_active = predictions[active_customer_ids]
                targets_active = targets[active_customer_ids]

                
                if targets_active.ndim==1:
                    targets_active = targets_active.unsqueeze(1)
                loss = criterion(predictions_active, targets_active)
                total_loss = total_loss + loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                if h_dict is not None:
                    h_dict = {k: v.detach() for k, v in h_dict.items()}
                if c_dict is not None:
                    c_dict = {k: v.detach() for k, v in c_dict.items()}
            
            # Calculate average loss for the epoc
            #epoch_loss = total_loss / signal.snapshot_count
            epoch_loss = total_loss / len(signal)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improvement = 0
            else:
                no_improvement = no_improvement+1
                if no_improvement >= patience:
                    print("Early stopping...")
                    break
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print("Checking tensors for issue")
            for node_type, features in x_dict.items():
                print(f"{node_type} features shape: {features.shape}")
            
            for edge_type, indices in edge_index_dict.items():
                src, rel, dst = edge_type
                print(f"Edge {edge_type} shape: {indices.shape}")
                
                if indices.shape[1] > 0:
                    max_src = indices[0].max().item()
                    max_dst = indices[1].max().item()
                    print(f"  - Max source index ({src}): {max_src}")
                    print(f"  - Max target index ({dst}): {max_dst}")
                    print(f"  - {src} feature dim: {x_dict[src].shape[0] if src in x_dict else 'N/A'}")
                    print(f"  - {dst} feature dim: {x_dict[dst].shape[0] if dst in x_dict else 'N/A'}")
            
            raise
    return model, h_dict, c_dict

def train2(model, signal, epochs=100, patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()
    best_loss = float('inf')
    no_improvement = 0
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        h_dict, c_dict = None, None
        try: 
            for t, snapshot in enumerate(signal):
                # Get data from temporal snapshot
                x_dict = snapshot.x_dict
                edge_index_dict = snapshot.edge_index_dict
                targets = snapshot['customer'].y
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(1)
                # Forward pass
                predictions, h_dict, c_dict = model(
                    x_dict, 
                    edge_index_dict, 
                    h_dict, 
                    c_dict
                )
                active_customer_ids = snapshot['active_customer_ids']['y']
                predictions_active = predictions[active_customer_ids]
                targets_active = targets[active_customer_ids]

                
                if targets_active.ndim==1:
                    targets_active = targets_active.unsqueeze(1)
                loss = criterion(predictions_active, targets_active)
                total_loss = total_loss + loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if h_dict is not None:
                    h_dict = {k: v.detach() for k, v in h_dict.items()}
                if c_dict is not None:
                    c_dict = {k: v.detach() for k, v in c_dict.items()}
            
            # Calculate average loss for the epoc
            epoch_loss = total_loss / len(signal)
            #epoch_loss = total_loss / signal.snapshot_count
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improvement = 0
            else:
                no_improvement = no_improvement+1
                if no_improvement >= patience:
                    print("Early stopping...")
                    break
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print("Checking tensors for issue")
            for node_type, features in x_dict.items():
                print(f"{node_type} features shape: {features.shape}")
            
            for edge_type, indices in edge_index_dict.items():
                src, rel, dst = edge_type
                print(f"Edge {edge_type} shape: {indices.shape}")
                
                if indices.shape[1] > 0:
                    max_src = indices[0].max().item()
                    max_dst = indices[1].max().item()
                    print(f"  - Max source index ({src}): {max_src}")
                    print(f"  - Max target index ({dst}): {max_dst}")
                    print(f"  - {src} feature dim: {x_dict[src].shape[0] if src in x_dict else 'N/A'}")
                    print(f"  - {dst} feature dim: {x_dict[dst].shape[0] if dst in x_dict else 'N/A'}")
            
            raise
    return model, h_dict, c_dict       

def train_self_supervised(model, signal, epochs = 100, patience = 10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    best_loss = float('inf')
    no_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        h_dict, c_dict = None, None
        try: 
            for t, snapshot in enumerate(signal):
                x_dict = snapshot.x_dict
                edge_index_dict = snapshot.edge_index_dict

                # Forward pass
                h_dict, c_dict = model(x_dict, edge_index_dict, h_dict, c_dict)

                pos_edge_index = torch.tensor(
                    snapshot.edge_index_dict[('customer', 'purchases', 'product')], 
                    dtype = torch.long, 
                    device = x_dict['customer'].device
                )

                #skip snapshot if no positive edges
                if pos_edge_index.numel() == 0:
                    continue

                cust_embeddings = h_dict['customer']
                product_embeddings = h_dict['product']

                pos_score = torch.sum(cust_embeddings[pos_edge_index[0]] * product_embeddings[pos_edge_index[1]], dim=-1)
                num_pos = pos_edge_index.size(1)
                neg_customer_indices = torch.randint(0, cust_embeddings.size(0), (num_pos,), device=pos_edge_index.device)
                neg_product_indices = torch.randint(0, product_embeddings.size(0), (num_pos,), device=pos_edge_index.device)
                neg_score = torch.sum(cust_embeddings[neg_customer_indices] * product_embeddings[neg_product_indices], dim=-1)

                scores = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)

                loss = F.binary_cross_entropy_with_logits(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                if h_dict is not None:
                    h_dict = {k: v.detach() for k, v in h_dict.items()}
                if c_dict is not None:
                    c_dict = {k: v.detach() for k, v in c_dict.items()}
            epoch_loss = total_loss / len(signal)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print("Early stopping...")
                    break

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print("Checking tensors for issue")
            for node_type, features in x_dict.items():
                print(f"{node_type} features shape: {features.shape}")
            
            for edge_type, indices in edge_index_dict.items():
                src, rel, dst = edge_type
                print(f"Edge {edge_type} shape: {indices.shape}")
                
                if indices.shape[1] > 0:
                    max_src = indices[0].max().item()
                    max_dst = indices[1].max().item()
                    print(f"  - Max source index ({src}): {max_src}")
                    print(f"  - Max target index ({dst}): {max_dst}")
                    print(f"  - {src} feature dim: {x_dict[src].shape[0] if src in x_dict else 'N/A'}")
                    print(f"  - {dst} feature dim: {x_dict[dst].shape[0] if dst in x_dict else 'N/A'}")
            
            raise
    return model, h_dict, c_dict


def train_self_supervised_test(model, train_signal, epochs=100, patience=10, lr=0.0005): # Renamed signal to train_signal
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    no_improvement_counter = 0 # Renamed for clarity

    # Determine the number of GNN layers to initialize state lists correctly
    # Assuming your model has a 'gnn_layers' attribute which is a ModuleList
    num_gnn_layers = len(model.gnn_layers)

    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0 # Renamed for clarity

        # Initialize hidden and cell states for the start of each epoch (sequence of snapshots)
        # These will be lists of dictionaries, one dictionary per layer's state
        h_list_for_sequence = [None] * num_gnn_layers
        c_list_for_sequence = [None] * num_gnn_layers
        
        num_snapshots_processed = 0

        try:
            for t, snapshot in enumerate(train_signal):
                # Ensure snapshot data is on the same device as the model
                # This should ideally be handled when loading/creating the snapshot (e.g., snapshot.to(device))
                # For this example, let's assume x_dict items determine the device
                device = next(iter(snapshot.x_dict.values())).device
                
                x_dict = {k: v.to(device) for k, v in snapshot.x_dict.items()}
                edge_index_dict = {k: v.to(device) for k, v in snapshot.edge_index_dict.items()}
                
                # IMPORTANT: Extract edge_attr_dict from the snapshot
                # Assuming snapshot might have an 'edge_attr_dict' attribute or
                # you construct it from individual snapshot[edge_type].edge_attr
                if hasattr(snapshot, 'edge_attr_dict') and snapshot.edge_attr_dict is not None:
                    edge_attr_dict = {k: v.to(device) for k, v in snapshot.edge_attr_dict.items()}
                else: # Fallback: try to construct it if snapshot is a PyG HeteroData object
                    edge_attr_dict = {}
                    for store in snapshot.edge_stores: # Iterate through edge types in HeteroData
                        if hasattr(store, 'edge_attr') and store.edge_attr is not None:
                             edge_attr_dict[(store._key[0], store._key[1], store._key[2])] = store.edge_attr.to(device) # PyG key format


                # Forward pass: model now returns final node representations and lists of states
                final_node_repr_dict, h_list_for_sequence, c_list_for_sequence = model(
                    x_dict,
                    edge_index_dict,
                    edge_attr_dict, # Pass the edge attributes
                    h_list_for_sequence,
                    c_list_for_sequence
                )

                # --- Contrastive Loss Calculation (mostly as before) ---
                # Use the ('customer', 'purchases', 'product') edge type for positive examples
                positive_edge_type = ('customer', 'purchases', 'product')
                if positive_edge_type not in edge_index_dict or edge_index_dict[positive_edge_type].numel() == 0:
                    # print(f"Snapshot {t}: No positive edges of type {positive_edge_type}, skipping loss calculation.")
                    continue # Skip snapshot if no positive edges for loss

                pos_edge_index = edge_index_dict[positive_edge_type]

                # Ensure embeddings for customer and product are available
                if 'customer' not in final_node_repr_dict or 'product' not in final_node_repr_dict:
                    # print(f"Snapshot {t}: Customer or product embeddings not found in model output, skipping loss.")
                    continue
                
                cust_embeddings = final_node_repr_dict['customer']
                product_embeddings = final_node_repr_dict['product']

                # Check if node types involved in positive edges actually have embeddings outputted
                if cust_embeddings.size(0) == 0 or product_embeddings.size(0) == 0:
                    # print(f"Snapshot {t}: Zero customer or product embeddings, skipping loss.")
                    continue
                
                # Validate indices in pos_edge_index
                if pos_edge_index[0].max() >= cust_embeddings.size(0) or \
                   pos_edge_index[1].max() >= product_embeddings.size(0):
                    print(f"Snapshot {t}: Index out of bounds in pos_edge_index. Skipping loss.")
                    # This indicates a mismatch between edge indices and node feature counts.
                    # You might want to add more detailed debugging here from your RuntimeError block.
                    continue


                pos_score = torch.sum(cust_embeddings[pos_edge_index[0]] * product_embeddings[pos_edge_index[1]], dim=-1)
                
                num_pos = pos_edge_index.size(1)
                # Ensure there are positive examples before negative sampling
                if num_pos == 0:
                    continue

                # Negative sampling
                neg_customer_indices = torch.randint(0, cust_embeddings.size(0), (num_pos,), device=device)
                neg_product_indices = torch.randint(0, product_embeddings.size(0), (num_pos,), device=device)
                neg_score = torch.sum(cust_embeddings[neg_customer_indices] * product_embeddings[neg_product_indices], dim=-1)

                scores = torch.cat([pos_score, neg_score], dim=0)
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)

                loss = F.binary_cross_entropy_with_logits(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                optimizer.step()
                
                total_epoch_loss += loss.item()
                num_snapshots_processed += 1

                # Detach hidden and cell states for the next iteration (snapshot)
                if h_list_for_sequence is not None:
                    h_list_for_sequence = [
                        ({k: v.detach() for k, v in h_layer_dict.items()} if h_layer_dict else None)
                        for h_layer_dict in h_list_for_sequence
                    ]
                if c_list_for_sequence is not None:
                    c_list_for_sequence = [
                        ({k: v.detach() for k, v in c_layer_dict.items()} if c_layer_dict else None)
                        for c_layer_dict in c_list_for_sequence
                    ]
            
            if num_snapshots_processed > 0:
                epoch_loss = total_epoch_loss / num_snapshots_processed
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

                # Early stopping logic
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    no_improvement_counter = 0
                    # Optional: Save best model
                    # torch.save(model.state_dict(), 'best_self_supervised_model.pth')
                else:
                    no_improvement_counter += 1
                    if no_improvement_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs.")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs}, No snapshots processed with valid loss.")


        except RuntimeError as e:
            print(f"Runtime error during epoch {epoch+1}, snapshot {t}: {e}")
            print("--- Tensor Debug Info ---")
            if 'x_dict' in locals():
                for node_type, features in x_dict.items():
                    print(f"  {node_type} features shape: {features.shape}, device: {features.device}")
            if 'edge_index_dict' in locals():
                for edge_type, indices in edge_index_dict.items():
                    src, rel, dst = edge_type
                    print(f"  Edge {edge_type} shape: {indices.shape}, device: {indices.device}")
                    if indices.numel() > 0 and indices.shape[1] > 0: # Check if not empty
                        max_src_idx = indices[0].max().item()
                        max_dst_idx = indices[1].max().item()
                        print(f"    - Max source index ({src}): {max_src_idx}")
                        print(f"    - Max target index ({dst}): {max_dst_idx}")
                        print(f"    - {src} feature count: {x_dict[src].shape[0] if src in x_dict else 'N/A'}")
                        print(f"    - {dst} feature count: {x_dict[dst].shape[0] if dst in x_dict else 'N/A'}")
            if 'edge_attr_dict' in locals() and edge_attr_dict is not None:
                for edge_type, attrs in edge_attr_dict.items():
                    print(f"  Edge attributes {edge_type} shape: {attrs.shape}, device: {attrs.device}")
            print("-------------------------")
            raise # Re-raise the exception after printing debug info
            
    print(f"Training finished. Best loss: {best_loss:.4f}")
    # Return the model (potentially the one with best_loss if you loaded it)
    # and the last states (though these are from the end of training, not necessarily best epoch)
    return model, h_list_for_sequence, c_list_for_sequence

import os
import pandas as pd
import numpy as np
import torch
import joblib
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class RiverNetworkDataset:
    def __init__(self, data_dir, model_out_dir):
        self.data_dir = data_dir
        self.model_out_dir = model_out_dir
        self.node_features_cols = [
            'UPLAND_SKM', 'CATCH_SKM', 'DIST_DN_KM', 
            'DIST_UP_KM', 'ORD_STRA', 'ORD_CLAS'
        ]
        self.edge_features_cols = [
            'routing_distance_km', 'steps', 
            'drainage_area_ratio', 'stream_order_delta'
        ]
        
        # Load exactly the existing node scaler
        self.node_scaler = joblib.load(os.path.join(
            self.data_dir, 'processed', 'scalers', 'graph_node_scaler.joblib'
        ))
        
        os.makedirs(self.model_out_dir, exist_ok=True)
        self.edge_scaler_path = os.path.join(self.model_out_dir, 'edge_scaler.joblib')
        self.edge_scaler = None

    def _load_split(self, split_name):
        nodes_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'graph', f'nodes_{split_name}.csv'))
        edges_df = pd.read_csv(os.path.join(self.data_dir, 'processed', 'graph', f'edges_{split_name}.csv'))
        
        # We assume node_idx corresponds to row order directly, but strictly we use source_idx/target_idx mapping.
        # Ensure nodes are sorted just in case
        nodes_df = nodes_df.sort_values(by='node_idx').reset_index(drop=True)
        
        x_raw = nodes_df[self.node_features_cols].values
        # Node scaling strictly using the pre-existing scaler (do not refit)
        x_scaled = self.node_scaler.transform(x_raw)
        
        edge_index = torch.from_numpy(np.stack([edges_df['source_idx'].values, edges_df['target_idx'].values], axis=0)).long()
        edge_attr_raw = edges_df[self.edge_features_cols].values
        
        return torch.tensor(x_scaled, dtype=torch.float32), edge_index, edge_attr_raw, nodes_df['gauge_id'].values

    def prepare_data(self):
        """
        Loads train/val/test graphs, fits edge scaler ON TRAIN ONLY, applies scaling.
        Returns a dictionary of PyG Data objects.
        """
        # Node IDs may be loaded here for verification
        x_train, ei_train, ea_train_raw, ids_train = self._load_split('train')
        x_val, ei_val, ea_val_raw, ids_val = self._load_split('val')
        x_test, ei_test, ea_test_raw, ids_test = self._load_split('test')

        # Fit edge scaler strictly on train set
        self.edge_scaler = StandardScaler()
        ea_train_scaled = self.edge_scaler.fit_transform(ea_train_raw)
        
        # Save edge scaler for future deployment
        joblib.dump(self.edge_scaler, self.edge_scaler_path)
        
        # Apply fitted scaler to val and test
        ea_val_scaled = self.edge_scaler.transform(ea_val_raw)
        ea_test_scaled = self.edge_scaler.transform(ea_test_raw)

        train_data = Data(
            x=x_train, 
            edge_index=ei_train, 
            edge_attr=torch.tensor(ea_train_scaled, dtype=torch.float32),
            gauge_id=ids_train
        )
        val_data = Data(
            x=x_val, 
            edge_index=ei_val, 
            edge_attr=torch.tensor(ea_val_scaled, dtype=torch.float32),
            gauge_id=ids_val
        )
        test_data = Data(
            x=x_test, 
            edge_index=ei_test, 
            edge_attr=torch.tensor(ea_test_scaled, dtype=torch.float32),
            gauge_id=ids_test
        )
        
        return {'train': train_data, 'val': val_data, 'test': test_data}

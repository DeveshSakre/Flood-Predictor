import os
import pytest
import torch
import joblib

from src.data.graph_dataset import RiverNetworkDataset
from src.models.gat_config import GATConfig
from src.models.river_gat import RiverGAT

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(REPO_ROOT / "data")
MODEL_DIR = str(REPO_ROOT / "models" / "gat")

@pytest.fixture
def dataset():
    loader = RiverNetworkDataset(DATA_DIR, MODEL_DIR)
    return loader.prepare_data()

@pytest.fixture
def config():
    return GATConfig()

@pytest.fixture
def model(config):
    return RiverGAT(config)

def test_graph_loading_and_counts(dataset):
    """Verify correct nodes and edges counts as per audit requirements."""
    train = dataset['train']
    val = dataset['val']
    test = dataset['test']
    
    assert train.x.shape[0] == 169, f"Expected 169 train nodes, got {train.x.shape[0]}"
    assert train.edge_index.shape[1] == 119, f"Expected 119 train edges, got {train.edge_index.shape[1]}"
    
    assert val.x.shape[0] == 36, f"Expected 36 validation nodes, got {val.x.shape[0]}"
    assert val.edge_index.shape[1] == 33, f"Expected 33 validation edges, got {val.edge_index.shape[1]}"
    
    assert test.x.shape[0] == 37, f"Expected 37 test nodes, got {test.x.shape[0]}"
    assert test.edge_index.shape[1] == 34, f"Expected 34 test edges, got {test.edge_index.shape[1]}"

def test_data_integrity(dataset):
    """Verify strictly 6 node features and 4 edge features."""
    for split in ['train', 'val', 'test']:
        data = dataset[split]
        assert data.x.shape[1] == 6, f"{split}: Node features should be 6"
        assert data.edge_attr.shape[1] == 4, f"{split}: Edge features should be 4"
        assert not torch.isnan(data.x).any(), f"{split}: NaN nodes detected"
        assert not torch.isnan(data.edge_attr).any(), f"{split}: NaN edges detected"

def test_no_overlap(dataset):
    """Verify that there is no ID overlap between train, val, and test splits."""
    train_ids = set(dataset['train'].gauge_id)
    val_ids = set(dataset['val'].gauge_id)
    test_ids = set(dataset['test'].gauge_id)
    
    assert train_ids.isdisjoint(val_ids), "Train and Validation sets overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and Test sets overlap!"
    assert val_ids.isdisjoint(test_ids), "Validation and Test sets overlap!"

def test_scaler_saved(dataset):
    assert os.path.exists(os.path.join(MODEL_DIR, 'edge_scaler.joblib')), "Edge scaler not saved"

def test_forward_pass(dataset, model, config):
    """Verify that the model successfully outputs the exact expected contract dims."""
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val', 'test']:
            data = dataset[split]
            emb = model(data.x, data.edge_index, data.edge_attr)
            
            assert emb.shape == (data.x.shape[0], config.spatial_embedding_dim), \
                   f"Embedding shape mismatch for {split}"
            assert not torch.isnan(emb).any(), f"NaNs in embedding for {split}"


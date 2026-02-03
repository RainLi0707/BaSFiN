
import sys
import os
import torch
import pytest

# Add code directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../code/BaSFiN')))

from BaS import NAC_BBB

def test_nac_model_initialization():
    """Test if NAC_BBB model initializes correctly."""
    n_player = 100
    model = NAC_BBB(n_player=n_player, team_size=5, device=torch.device('cpu'))
    assert model is not None
    assert model.n_player == n_player

def test_nac_model_forward_pass():
    """Test if a forward pass runs without error."""
    n_player = 100
    batch_size = 4
    team_size = 5
    model = NAC_BBB(n_player=n_player, team_size=team_size, device=torch.device('cpu'))
    
    # Create dummy data: [batch_size, 1 + 2*team_size + target] logic usually, 
    # but model forward expects [batch_size, ...] where indices are players
    # Based on BaS.py:
    # team_A = data[:, 1:1+self.team_size]
    # team_B = data[:, 1+self.team_size:]
    # So input should have at least 1 + 2*team_size columns if it slices like that.
    
    dummy_input = torch.randint(0, n_player, (batch_size, 1 + 2 * team_size + 1))
    
    # Forward pass
    prob, z = model(dummy_input, num_samples=2)
    
    assert prob.shape == (2, batch_size) # (num_samples, batch_size)
    assert z.shape == (2, batch_size)

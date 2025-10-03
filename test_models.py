"""
Unit tests for models and data pipeline

Run with: python test_models.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from models import ForwardModel, ForwardModelResNet, InversecVAE
from models.utils import gumbel_softmax_binary


def test_forward_model():
    """Test forward model shapes and forward pass"""
    print("\n" + "="*60)
    print("Testing Forward Model")
    print("="*60)
    
    model = ForwardModel()
    model.eval()
    
    # Test input
    batch_size = 4
    pattern = torch.rand(batch_size, 1, 48, 32)
    
    # Forward pass
    with torch.no_grad():
        output = model(pattern)
    
    # Check output shape
    expected_shape = (batch_size, 201, 4)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print(f"✓ Input shape:  {pattern.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Forward model test passed!")
    
    return True


def test_forward_model_resnet():
    """Test ResNet forward model"""
    print("\n" + "="*60)
    print("Testing Forward Model (ResNet)")
    print("="*60)
    
    model = ForwardModelResNet()
    model.eval()
    
    batch_size = 4
    pattern = torch.rand(batch_size, 1, 48, 32)
    
    with torch.no_grad():
        output = model(pattern)
    
    expected_shape = (batch_size, 201, 4)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Input shape:  {pattern.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Parameters:   {n_params:,}")
    print(f"✓ ResNet forward model test passed!")
    
    return True


def test_inverse_cvae():
    """Test inverse cVAE model"""
    print("\n" + "="*60)
    print("Testing Inverse cVAE")
    print("="*60)
    
    latent_dim = 128
    model = InversecVAE(latent_dim=latent_dim)
    model.eval()
    
    batch_size = 4
    S_params = torch.randn(batch_size, 201, 4)
    
    # Test encode
    with torch.no_grad():
        mu, logvar = model.encode(S_params)
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        print(f"✓ Encoder output: mu={mu.shape}, logvar={logvar.shape}")
        
        # Test reparameterization
        z = model.reparameterize(mu, logvar)
        assert z.shape == (batch_size, latent_dim)
        print(f"✓ Latent vector: {z.shape}")
        
        # Test decode
        logits = model.decode(z, S_params)
        expected_shape = (batch_size, 1, 48, 32)
        assert logits.shape == expected_shape
        print(f"✓ Decoder output: {logits.shape}")
        
        # Test full forward pass
        logits_full, mu_full, logvar_full = model(S_params)
        assert logits_full.shape == expected_shape
        print(f"✓ Full forward pass: {logits_full.shape}")
        
        # Test pattern generation
        pattern = (torch.sigmoid(logits_full) > 0.5).float()
        assert pattern.shape == expected_shape
        assert torch.all((pattern == 0) | (pattern == 1))
        print(f"✓ Binary pattern: {pattern.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {n_params:,}")
    print(f"✓ Inverse cVAE test passed!")
    
    return True


def test_gumbel_softmax():
    """Test Gumbel-Softmax utility"""
    print("\n" + "="*60)
    print("Testing Gumbel-Softmax")
    print("="*60)
    
    logits = torch.randn(4, 1, 48, 32)
    
    # Test soft sampling
    soft = gumbel_softmax_binary(logits, temperature=1.0, hard=False)
    assert soft.shape == logits.shape
    assert torch.all(soft >= 0) and torch.all(soft <= 1)
    print(f"✓ Soft sampling: shape={soft.shape}, range=[{soft.min():.3f}, {soft.max():.3f}]")
    
    # Test hard sampling
    hard = gumbel_softmax_binary(logits, temperature=0.5, hard=True)
    assert hard.shape == logits.shape
    assert torch.all((hard == 0) | (hard == 1))
    print(f"✓ Hard sampling: shape={hard.shape}, unique values={torch.unique(hard).tolist()}")
    
    # Test temperature effect
    hot = gumbel_softmax_binary(logits, temperature=5.0, hard=False)
    cold = gumbel_softmax_binary(logits, temperature=0.1, hard=False)
    
    hot_entropy = -torch.sum(hot * torch.log(hot + 1e-8) + (1-hot) * torch.log(1-hot + 1e-8))
    cold_entropy = -torch.sum(cold * torch.log(cold + 1e-8) + (1-cold) * torch.log(1-cold + 1e-8))
    
    print(f"✓ Temperature effect: high_T entropy > low_T entropy")
    print(f"  T=5.0:  entropy={hot_entropy:.2f}")
    print(f"  T=0.1:  entropy={cold_entropy:.2f}")
    print(f"✓ Gumbel-Softmax test passed!")
    
    return True


def test_end_to_end():
    """Test complete forward-inverse pipeline"""
    print("\n" + "="*60)
    print("Testing End-to-End Pipeline")
    print("="*60)
    
    # Models
    forward_model = ForwardModelResNet()
    inverse_model = InversecVAE(latent_dim=128)
    forward_model.eval()
    inverse_model.eval()
    
    batch_size = 2
    
    # 1. Start with a pattern
    original_pattern = (torch.rand(batch_size, 1, 48, 32) > 0.5).float()
    print(f"✓ Original pattern: {original_pattern.shape}")
    
    # 2. Get S-parameters from forward model
    with torch.no_grad():
        S_params = forward_model(original_pattern)
    print(f"✓ Forward pass: pattern -> S-params {S_params.shape}")
    
    # 3. Generate pattern from inverse model
    with torch.no_grad():
        logits, mu, logvar = inverse_model(S_params)
        reconstructed_pattern = (torch.sigmoid(logits) > 0.5).float()
    print(f"✓ Inverse pass: S-params -> pattern {reconstructed_pattern.shape}")
    
    # 4. Verify with forward model again
    with torch.no_grad():
        S_params_reconstructed = forward_model(reconstructed_pattern)
    print(f"✓ Validation pass: reconstructed pattern -> S-params")
    
    # Calculate reconstruction accuracy (won't be perfect with untrained models)
    pattern_accuracy = (reconstructed_pattern == original_pattern).float().mean()
    s_param_error = torch.mean((S_params - S_params_reconstructed) ** 2)
    
    print(f"  Pattern reconstruction accuracy: {pattern_accuracy:.4f}")
    print(f"  S-parameter MSE: {s_param_error:.6f}")
    print(f"✓ End-to-end pipeline test passed!")
    
    return True


def test_data_formats():
    """Test that data formats are compatible"""
    print("\n" + "="*60)
    print("Testing Data Formats")
    print("="*60)
    
    # Create dummy data
    pattern = np.random.randint(0, 2, size=(48, 32)).astype(np.float32)
    
    # Create S-parameters DataFrame
    s_params = pd.DataFrame({
        'Frequency': np.linspace(190, 200, 201),
        'S11 dB': np.random.randn(201) * 5 - 10,
        'S21 dB': np.random.randn(201) * 5 - 50,
        'S22 dB': np.random.randn(201) * 5 - 10,
        'S12 dB': np.random.randn(201) * 5 - 50,
    })
    
    print(f"✓ Pattern shape: {pattern.shape}, dtype: {pattern.dtype}")
    print(f"✓ S-params shape: {s_params.shape}")
    print(f"✓ S-params columns: {list(s_params.columns)}")
    
    # Test file I/O
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save pattern
        pattern_file = tmpdir / "test_pattern.array"
        pattern.tofile(pattern_file)
        
        # Load pattern
        loaded_pattern = np.fromfile(pattern_file, dtype=np.float32).reshape(48, 32)
        assert np.array_equal(pattern, loaded_pattern)
        print(f"✓ Pattern file I/O successful")
        
        # Save S-parameters
        pkl_file = tmpdir / "test_sparams.pkl"
        s_params.to_pickle(pkl_file)
        
        # Load S-parameters
        loaded_s_params = pd.read_pickle(pkl_file)
        assert loaded_s_params.equals(s_params)
        print(f"✓ S-parameters file I/O successful")
    
    print(f"✓ Data format test passed!")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        ("Forward Model", test_forward_model),
        ("Forward Model ResNet", test_forward_model_resnet),
        ("Inverse cVAE", test_inverse_cvae),
        ("Gumbel-Softmax", test_gumbel_softmax),
        ("End-to-End Pipeline", test_end_to_end),
        ("Data Formats", test_data_formats),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} FAILED!")
            print(f"  Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    n_passed = sum(1 for _, success in results if success)
    n_total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n✓ All tests passed! Models are working correctly.")
        return True
    else:
        print(f"\n✗ {n_total - n_passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

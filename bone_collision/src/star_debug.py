#!/usr/bin/env python3
"""
Debug script to find the correct STAR model attributes
"""

import torch
from star.pytorch.star import STAR

def debug_star_model():
    """Debug STAR model to find correct attribute names"""
    print("=" * 60)
    print("STAR MODEL DEBUG")
    print("=" * 60)
    
    # Initialize STAR model
    model = STAR(gender='neutral')
    print(f"STAR model type: {type(model)}")
    
    # Get all attributes
    print(f"\n=== All STAR Model Attributes ===")
    all_attrs = dir(model)
    
    # Filter for likely skinning weight attributes
    weight_attrs = []
    lbs_attrs = []
    matrix_attrs = []
    
    for attr in all_attrs:
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(model, attr)
                attr_type = type(value).__name__
                
                # Look for tensor attributes that might be skinning weights
                if isinstance(value, torch.Tensor):
                    shape = tuple(value.shape)
                    print(f"  {attr}: {attr_type} {shape}")
                    
                    # Check for likely skinning weight dimensions
                    if len(shape) == 2:
                        if shape[0] == 6890:  # STAR vertex count
                            weight_attrs.append((attr, shape))
                        elif 24 in shape:  # Joint count
                            matrix_attrs.append((attr, shape))
                
                # Look for LBS-related attributes
                if 'lbs' in attr.lower() or 'weight' in attr.lower():
                    lbs_attrs.append((attr, attr_type, getattr(value, 'shape', 'no shape')))
                    
            except Exception as e:
                print(f"  {attr}: Error accessing - {e}")
    
    print(f"\n=== Likely Skinning Weight Attributes ===")
    for attr, shape in weight_attrs:
        print(f"  {attr}: {shape}")
    
    print(f"\n=== LBS/Weight Related Attributes ===")
    for attr, attr_type, shape in lbs_attrs:
        print(f"  {attr}: {attr_type} {shape}")
    
    print(f"\n=== Matrix Attributes with Joint Dimension (24) ===")
    for attr, shape in matrix_attrs:
        print(f"  {attr}: {shape}")
    
    # Try common skinning weight attribute names
    common_names = [
        'lbs_weights', 'weights', 'skinning_weights', 'W', 'lbs_W',
        'J_regressor_prior', 'lbs_J_regressor', 'vertex_joint_selector'
    ]
    
    print(f"\n=== Testing Common Skinning Weight Names ===")
    for name in common_names:
        try:
            attr = getattr(model, name)
            print(f"  ✓ {name}: {type(attr)} {getattr(attr, 'shape', 'no shape')}")
        except AttributeError:
            print(f"  ✗ {name}: Not found")
    
    # Check the model's state_dict for weights
    print(f"\n=== Model State Dict Keys (with 'weight' or 'lbs') ===")
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if 'weight' in key.lower() or 'lbs' in key.lower():
            tensor = state_dict[key]
            print(f"  {key}: {tensor.shape}")
    
    print(f"\n=== Manual Inspection Suggestions ===")
    print("Try these in your Python REPL:")
    print("  model = STAR(gender='neutral')")
    print("  print([attr for attr in dir(model) if 'weight' in attr.lower()])")
    print("  print([key for key in model.state_dict().keys() if 'weight' in key])")
    print("  print(model.state_dict().keys())")

if __name__ == "__main__":
    debug_star_model()
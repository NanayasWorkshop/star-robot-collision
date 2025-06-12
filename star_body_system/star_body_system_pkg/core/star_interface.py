"""
STAR Model Interface
Handles STAR model loading and mesh/joint data extraction
"""

import numpy as np
import torch

try:
    from star.pytorch.star import STAR
    STAR_AVAILABLE = True
except ImportError:
    print("WARNING: STAR model not available. Install from https://github.com/ahmedosman/STAR")
    STAR_AVAILABLE = False


class STARInterface:
    """Interface to STAR model for mesh and joint data"""
    
    def __init__(self, gender='neutral'):
        if not STAR_AVAILABLE:
            raise ImportError("STAR model required")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = STAR(gender=gender)
        self.model.to(self.device)
        self.model.eval()
        
    def get_mesh_and_joints(self, pose_params=None):
        """
        Get mesh vertices and joint positions
        
        Args:
            pose_params: Pose parameters (72,) or None for neutral pose
            
        Returns:
            tuple: (vertices, joints) as numpy arrays
        """
        batch_size = 1
        if pose_params is None:
            pose_params = torch.zeros(batch_size, 72, device=self.device)
        elif isinstance(pose_params, np.ndarray):
            pose_params = torch.from_numpy(pose_params).float().to(self.device)
            if pose_params.dim() == 1:
                pose_params = pose_params.unsqueeze(0)
        
        shape_params = torch.zeros(batch_size, 10, device=self.device)
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        with torch.no_grad():
            try:
                result = self.model(pose_params, shape_params, trans)
                if isinstance(result, tuple):
                    vertices, joints = result
                    return vertices[0].cpu().numpy(), joints[0].cpu().numpy()
                else:
                    vertices = result
                    if hasattr(self.model, 'J_regressor'):
                        joints = torch.matmul(self.model.J_regressor, vertices)
                        return vertices[0].cpu().numpy(), joints[0].cpu().numpy()
                    else:
                        return vertices[0].cpu().numpy(), None
            except Exception as e:
                print(f"Error getting mesh data: {e}")
                return None, None
    
    def get_neutral_pose(self):
        """Get mesh and joints for neutral pose"""
        return self.get_mesh_and_joints()
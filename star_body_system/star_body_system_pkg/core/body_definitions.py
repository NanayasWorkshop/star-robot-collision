"""
Body Definitions
Contains anatomical definitions, joint names, bone connections, layer hierarchies, and containment mappings
Updated with optimized sphere parameters for tighter mesh fit
"""


class BodyDefinitions:
    """Anatomical definitions for multi-layer body representation"""
    
    # STAR joint names (24 joints)
    JOINT_NAMES = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]
    
    # Layer 2: 24 bone connections (detailed anatomical)
    DETAILED_BONES = [
        (0, 3, 'pelvis-spine1'),
        (3, 6, 'spine1-spine2'),  
        (6, 9, 'spine2-spine3'),
        (9, 12, 'spine3-neck'),
        (12, 15, 'neck-head'),
        
        (0, 1, 'pelvis-left_hip'),
        (1, 4, 'left_hip-left_knee'),
        (4, 7, 'left_knee-left_ankle'),
        (7, 10, 'left_ankle-left_foot'),
        
        (0, 2, 'pelvis-right_hip'),
        (2, 5, 'right_hip-right_knee'),
        (5, 8, 'right_knee-right_ankle'),
        (8, 11, 'right_ankle-right_foot'),
        
        (9, 13, 'spine3-left_collar'),
        (13, 16, 'left_collar-left_shoulder'),
        (16, 18, 'left_shoulder-left_elbow'),
        (18, 20, 'left_elbow-left_wrist'),
        (20, 22, 'left_wrist-left_hand'),
        
        (9, 14, 'spine3-right_collar'),
        (14, 17, 'right_collar-right_shoulder'),
        (17, 19, 'right_shoulder-right_elbow'),
        (19, 21, 'right_elbow-right_wrist'),
        (21, 23, 'right_wrist-right_hand'),
    ]
    
    # Layer 3: 9 simplified bone connections
    SIMPLE_BONES = [
        # Legs (4 capsules)
        (10, 4, 'left_foot-left_knee'),    # left lower leg
        (11, 5, 'right_foot-right_knee'),  # right lower leg  
        (4, 0, 'left_knee-pelvis'),        # left upper leg
        (5, 0, 'right_knee-pelvis'),       # right upper leg
        
        # Core (1 capsule)
        (0, 15, 'pelvis-head'),             # entire torso
        
        # Arms (4 capsules)
        (16, 18, 'left_shoulder-left_elbow'),   # left upper arm
        (17, 19, 'right_shoulder-right_elbow'), # right upper arm
        (18, 22, 'left_elbow-left_hand'),       # left forearm+hand
        (19, 23, 'right_elbow-right_hand'),     # right forearm+hand
    ]
    
    # HIERARCHICAL CONTAINMENT MAPPINGS
    
    # Layer 3 → Layer 2: Each Layer 3 capsule contains these Layer 2 capsules
    LAYER3_TO_LAYER2_CONTAINMENT = {
        'left_foot-left_knee': [
            'left_knee-left_ankle',
            'left_ankle-left_foot'
        ],
        'right_foot-right_knee': [
            'right_knee-right_ankle', 
            'right_ankle-right_foot'
        ],
        'left_knee-pelvis': [
            'pelvis-left_hip',
            'left_hip-left_knee'
        ],
        'right_knee-pelvis': [
            'pelvis-right_hip',
            'right_hip-right_knee'
        ],
        'pelvis-head': [
            'pelvis-spine1',
            'spine1-spine2',
            'spine2-spine3', 
            'spine3-neck',
            'neck-head',
            'spine3-left_collar',   # Collar connections belong to torso
            'spine3-right_collar'
        ],
        'left_shoulder-left_elbow': [
            'left_collar-left_shoulder',
            'left_shoulder-left_elbow'
        ],
        'right_shoulder-right_elbow': [
            'right_collar-right_shoulder', 
            'right_shoulder-right_elbow'
        ],
        'left_elbow-left_hand': [
            'left_elbow-left_wrist',
            'left_wrist-left_hand'
        ],
        'right_elbow-right_hand': [
            'right_elbow-right_wrist',
            'right_wrist-right_hand'
        ]
    }
    
    # Layer 2 → Layer 1: Each Layer 2 capsule contains spheres with matching bone name
    # This is handled dynamically by name matching (e.g., 'pelvis-spine1' contains all 'pelvis-spine1_X' spheres)
    
    # Anatomical properties for Layer 1 sphere generation - OPTIMIZED for tighter fit
    BONE_DEFINITIONS = {
        # Core/Torso - Smaller spheres, more density
        'pelvis-spine1': {'start_radius': 0.08, 'end_radius': 0.08, 'type': 'torso', 'min_overlap': 0.2},
        'spine1-spine2': {'start_radius': 0.08, 'end_radius': 0.07, 'type': 'torso', 'min_overlap': 0.2},
        'spine2-spine3': {'start_radius': 0.07, 'end_radius': 0.07, 'type': 'torso', 'min_overlap': 0.2},
        'spine3-neck': {'start_radius': 0.07, 'end_radius': 0.06, 'type': 'torso', 'min_overlap': 0.2},
        
        # Head/Neck - Keep similar
        'neck-head': {'start_radius': 0.06, 'end_radius': 0.08, 'type': 'head', 'min_overlap': 0.25},
        
        # Legs - Much smaller spheres, higher density
        'pelvis-left_hip': {'start_radius': 0.07, 'end_radius': 0.07, 'type': 'leg_upper', 'min_overlap': 0.25},
        'left_hip-left_knee': {'start_radius': 0.07, 'end_radius': 0.05, 'type': 'leg_upper', 'min_overlap': 0.25},
        'left_knee-left_ankle': {'start_radius': 0.05, 'end_radius': 0.035, 'type': 'leg_lower', 'min_overlap': 0.25},
        'left_ankle-left_foot': {'start_radius': 0.035, 'end_radius': 0.04, 'type': 'foot', 'min_overlap': 0.3},
        
        'pelvis-right_hip': {'start_radius': 0.07, 'end_radius': 0.07, 'type': 'leg_upper', 'min_overlap': 0.25},
        'right_hip-right_knee': {'start_radius': 0.07, 'end_radius': 0.05, 'type': 'leg_upper', 'min_overlap': 0.25},
        'right_knee-right_ankle': {'start_radius': 0.05, 'end_radius': 0.035, 'type': 'leg_lower', 'min_overlap': 0.25},
        'right_ankle-right_foot': {'start_radius': 0.035, 'end_radius': 0.04, 'type': 'foot', 'min_overlap': 0.3},
        
        # Arms - Slightly smaller, keep similar overlap
        'spine3-left_collar': {'start_radius': 0.08, 'end_radius': 0.07, 'type': 'shoulder', 'min_overlap': 0.25},
        'left_collar-left_shoulder': {'start_radius': 0.07, 'end_radius': 0.06, 'type': 'shoulder', 'min_overlap': 0.25},
        'left_shoulder-left_elbow': {'start_radius': 0.06, 'end_radius': 0.045, 'type': 'arm_upper', 'min_overlap': 0.3},
        'left_elbow-left_wrist': {'start_radius': 0.045, 'end_radius': 0.035, 'type': 'arm_lower', 'min_overlap': 0.3},
        'left_wrist-left_hand': {'start_radius': 0.035, 'end_radius': 0.04, 'type': 'hand', 'min_overlap': 0.3},
        
        'spine3-right_collar': {'start_radius': 0.08, 'end_radius': 0.07, 'type': 'shoulder', 'min_overlap': 0.25},
        'right_collar-right_shoulder': {'start_radius': 0.07, 'end_radius': 0.06, 'type': 'shoulder', 'min_overlap': 0.25},
        'right_shoulder-right_elbow': {'start_radius': 0.06, 'end_radius': 0.045, 'type': 'arm_upper', 'min_overlap': 0.3},
        'right_elbow-right_wrist': {'start_radius': 0.045, 'end_radius': 0.035, 'type': 'arm_lower', 'min_overlap': 0.3},
        'right_wrist-right_hand': {'start_radius': 0.035, 'end_radius': 0.04, 'type': 'hand', 'min_overlap': 0.3},
    }
    
    # Default capsule radii for layers 2 and 3 (starting points - will be adjusted for containment)
    DEFAULT_CAPSULE_RADII = {
        # Layer 2 (detailed) - starting radii
        'torso': 0.11,
        'head': 0.07,
        'leg_upper': 0.085,
        'leg_lower': 0.055,
        'foot': 0.04,
        'shoulder': 0.075,
        'arm_upper': 0.06,
        'arm_lower': 0.045,
        'hand': 0.035,
        
        # Layer 3 (simplified) - starting radii (will grow to contain Layer 2)
        'pelvis-head': 0.15,           # Contains entire torso
        'left_foot-left_knee': 0.08,   # Contains lower leg
        'right_foot-right_knee': 0.08,
        'left_knee-pelvis': 0.12,      # Contains upper leg
        'right_knee-pelvis': 0.12,
        'left_shoulder-left_elbow': 0.08,   # Contains upper arm
        'right_shoulder-right_elbow': 0.08,
        'left_elbow-left_hand': 0.06,       # Contains forearm+hand
        'right_elbow-right_hand': 0.06,
    }
    
    @classmethod
    def get_joint_index(cls, joint_name):
        """Get index of joint by name"""
        try:
            return cls.JOINT_NAMES.index(joint_name)
        except ValueError:
            raise ValueError(f"Unknown joint name: {joint_name}")
    
    @classmethod
    def get_bone_definition(cls, bone_name):
        """Get bone definition by name"""
        return cls.BONE_DEFINITIONS.get(bone_name)
    
    @classmethod
    def get_layer3_children(cls, layer3_name):
        """Get Layer 2 capsules that should be contained by this Layer 3 capsule"""
        return cls.LAYER3_TO_LAYER2_CONTAINMENT.get(layer3_name, [])
    
    @classmethod
    def get_layer2_children(cls, layer2_name):
        """Get Layer 1 sphere name prefix that should be contained by this Layer 2 capsule"""
        # Layer 2 contains all spheres with matching bone name
        return layer2_name  # e.g., 'pelvis-spine1' contains all 'pelvis-spine1_X' spheres
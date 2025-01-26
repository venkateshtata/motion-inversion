import numpy as np
import random
from scipy.spatial.transform import Rotation as R

def introduce_inconsistencies(motion_data, frames_to_modify=[20, 40, 60]):
    """
    Introduce extreme spatial and temporal inconsistencies by randomly repositioning joints
    at specified frames
    """
    # Deep copy of motion data to avoid modifying the original
    modified_data = {key: np.copy(value) if isinstance(value, np.ndarray) else value.copy()
                     for key, value in motion_data.items()}

    # Randomly place `pos_root` at extreme positions for selected frames
    if 'pos_root' in modified_data and isinstance(modified_data['pos_root'], np.ndarray):
        for frame in frames_to_modify:
            if frame < modified_data['pos_root'].shape[0]:  # Ensure frame index is within bounds
                modified_data['pos_root'][frame] = np.random.uniform(-500, 500, size=modified_data['pos_root'][frame].shape)  

    # Assign extreme random rotations to `rot_root` for selected frames
    if 'rot_root' in modified_data and isinstance(modified_data['rot_root'], np.ndarray):
        for frame in frames_to_modify:
            if frame < modified_data['rot_root'].shape[0]:  # Ensure frame index is within bounds
                modified_data['rot_root'][frame] = R.random().as_quat()  # Completely random rotation

    # Randomly reposition joint offsets (`offsets_no_root`) only at selected frames
    if 'offsets_no_root' in modified_data and isinstance(modified_data['offsets_no_root'], np.ndarray):
        # Ensure we apply changes correctly if `offsets_no_root` is time-dependent
        if len(modified_data['offsets_no_root'].shape) == 3:  # If it has a time axis
            for frame in frames_to_modify:
                if frame < modified_data['offsets_no_root'].shape[0]:  # Ensure frame index is valid
                    modified_data['offsets_no_root'][frame] = np.random.uniform(-10, 10, modified_data['offsets_no_root'][frame].shape)
    
    return modified_data  # Return modified dictionary

# Load existing test data
test_data = np.load('./data/test_edge_rot_data.npy', allow_pickle=True)

# Modify only the first dictionary while preserving structure
test_data[0] = introduce_inconsistencies(test_data[0])

# Save the modified data back to the original file
np.save('./assignment_files/0_0_gt_modified.npy', test_data)

print("Modified test data saved successfully with extreme inconsistencies at frames 20, 40, and 60!")

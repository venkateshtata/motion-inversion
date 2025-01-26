import numpy as np

# Function to load `.bvh` file into text format
def load_bvh(bvh_path):
    with open(bvh_path, "r") as f:
        lines = f.readlines()
    return lines

# Function to modify motion data in `.bvh`
def modify_motion(bvh_lines, keyframes=[20, 40, 60], noise_factor=10.0):
    print(f"Modifying keyframes: {keyframes}")

    motion_start_idx = None
    for i, line in enumerate(bvh_lines):
        if line.startswith("Frame Time"):
            motion_start_idx = i
            break

    if motion_start_idx is None:
        raise ValueError("Could not find motion data in BVH file.")

    # Modify keyframe motion data
    for frame in keyframes:
        motion_idx = motion_start_idx + frame
        if motion_idx < len(bvh_lines):
            motion_values = np.array(bvh_lines[motion_idx].split(), dtype=float)
            motion_values += np.random.uniform(-noise_factor, noise_factor, motion_values.shape)  # Add noise
            bvh_lines[motion_idx] = " ".join(map(str, motion_values)) + "\n"  # Convert back to string

    return bvh_lines

# Function to save modified `.bvh`
def save_bvh(modified_bvh_lines, save_path):
    with open(save_path, "w") as f:
        f.writelines(modified_bvh_lines)
    print(f"Saved modified motion to: {save_path}")

# Main execution
if __name__ == "__main__":
    input_bvh = "./assignment_files/0_0_gt.bvh"
    output_bvh = "./assignment_files/0_0_gt_modified.bvh"

    bvh_lines = load_bvh(input_bvh)  # Load BVH as text lines
    modified_bvh_lines = modify_motion(bvh_lines)  # Modify keyframes
    save_bvh(modified_bvh_lines, output_bvh)  # Save modified motion

    print("BVH modification complete!")

import numpy as np
from scipy.spatial.transform import Rotation as R

class BVHParser:
    def __init__(self, bvh_file):
        self.file = bvh_file
        self.joint_names = []
        self.joint_parents = []
        self.joint_offsets = []
        self.motion_data = []
        self.parse()

    def parse(self):
        with open(self.file, 'r') as f:
            lines = f.readlines()

        hierarchy = True
        motion = False
        stack = []
        current_parent = -1

        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue

            if tokens[0] == "HIERARCHY":
                continue
            elif tokens[0] == "MOTION":
                motion = True
                hierarchy = False
                continue

            if hierarchy:
                if tokens[0] in ["ROOT", "JOINT"]:
                    self.joint_names.append(tokens[1])
                    self.joint_parents.append(current_parent)
                    stack.append(len(self.joint_names) - 1)
                    current_parent = len(self.joint_names) - 1
                elif tokens[0] == "OFFSET":
                    offset = list(map(float, tokens[1:]))
                    self.joint_offsets.append(offset)
                elif tokens[0] == "End":
                    self.joint_names.append(f"End_{self.joint_names[current_parent]}")
                    self.joint_parents.append(current_parent)
                elif tokens[0] == "}":
                    if stack:
                        current_parent = stack.pop()
            elif motion:
                if tokens[0] == "Frames:":
                    num_frames = int(tokens[1])
                elif tokens[0] == "Frame" and tokens[1] == "Time:":
                    continue
                else:
                    self.motion_data.append(list(map(float, tokens)))

        self.motion_data = np.array(self.motion_data)
        self.joint_offsets = np.array(self.joint_offsets)

    def to_npy(self, output_file):
        motion_sequence = []
        num_joints = len(self.joint_names)

        for frame in self.motion_data:
            frame_dict = {}
            
            root_pos = frame[:3]
            root_rot = R.from_euler('zyx', frame[3:6], degrees=True).as_quat()

            frame_dict['pos_root'] = np.array([root_pos])
            frame_dict['rot_root'] = np.array([root_rot])
            frame_dict['offset_root'] = self.joint_offsets[0]
            frame_dict['offsets_no_root'] = self.joint_offsets[1:]
            
            frame_dict['parents_with_root'] = np.array([-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 0, 17, 18])
            
            frame_dict['names_with_root'] = np.array([
                'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot'
            ], dtype='<U40')
            
            rot_data = frame[6:].reshape(-1, 3)
            rot_quats = np.array([R.from_euler('zyx', rot, degrees=True).as_quat() for rot in rot_data])
            
            frame_dict['rot_edge_no_root'] = rot_quats
            
            motion_sequence.append(frame_dict)
        
        np.save(output_file, motion_sequence)

bvh_file = '/content/drive/MyDrive/colab_notebooks/MoDi/MoDi/assignment_files/modified_motion.bvh'  # Replace with your actual file name
output_npy_file = './assignment_files/modified_motion.npy'
parser = BVHParser(bvh_file)
parser.to_npy(output_npy_file)

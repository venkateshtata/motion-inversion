import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import re

class BVHReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joint_channels = {}
        self.joint_parents = {}
        self.joint_offsets = {}
        self.channel_indexes = {}
        self.motion_data = None
        self.frames = 0
        self.frame_time = 0
        self.read_file()

    def read_file(self):
        with open(self.file_path, 'r') as f:
            content = f.read()
        hierarchy, motion = content.split('MOTION')
        self._parse_hierarchy(hierarchy)

        lines = motion.strip().split('\n')
        self.frames = int(re.search(r'Frames:\s*(\d+)', lines[0]).group(1))
        self.frame_time = float(re.search(r'Frame Time:\s*([0-9.]+)', lines[1]).group(1))

        motion_data = []
        for line in lines[2:]:
            if line.strip():
                motion_data.append([float(x) for x in line.split()])
        self.motion_data = np.array(motion_data)

    def _parse_hierarchy(self, hierarchy):
        current_joint = None
        joint_stack = []
        channel_index = 0

        for line in hierarchy.split('\n'):
            line = line.strip()
            if not line:
                continue

            if 'JOINT' in line or 'End Site' in line or 'ROOT' in line:
                joint_name = line.split()[-1]
                if joint_name == 'Site':
                    joint_name = f"{current_joint}_end"
                current_joint = joint_name

                if joint_stack:
                    self.joint_parents[current_joint] = joint_stack[-1]
                else:
                    self.joint_parents[current_joint] = None

                joint_stack.append(current_joint)

            elif 'CHANNELS' in line:
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                self.joint_channels[current_joint] = channels
                self.channel_indexes[current_joint] = channel_index
                channel_index += num_channels

            elif 'OFFSET' in line:
                offset = [float(x) for x in line.split()[-3:]]
                self.joint_offsets[current_joint] = np.array(offset)

            elif '}' in line:
                joint_stack.pop()
                if joint_stack:
                    current_joint = joint_stack[-1]

    def get_rotation_matrix(self, angles, order):
        matrices = []
        for angle, axis in zip(angles, order):
            c = np.cos(np.radians(angle))
            s = np.sin(np.radians(angle))
            if axis == 'X':
                matrices.append(np.array([[1, 0, 0],
                                        [0, c, -s],
                                        [0, s, c]]))
            elif axis == 'Y':
                matrices.append(np.array([[c, 0, s],
                                        [0, 1, 0],
                                        [-s, 0, c]]))
            elif axis == 'Z':
                matrices.append(np.array([[c, -s, 0],
                                        [s, c, 0],
                                        [0, 0, 1]]))
        rotation = matrices[0]
        for matrix in matrices[1:]:
            rotation = rotation @ matrix
        return rotation

    def get_joint_positions(self, frame_idx):
        positions = {}
        global_rotations = {}

        def process_joint(joint_name, parent_rotation=np.eye(3), parent_position=np.zeros(3)):
            if joint_name is None:
                return

            channels = self.joint_channels.get(joint_name, [])
            if not channels:
                positions[joint_name] = parent_position + (parent_rotation @ self.joint_offsets[joint_name])
                return

            channel_start = self.channel_indexes[joint_name]
            channel_values = self.motion_data[frame_idx, channel_start:channel_start + len(channels)]

            #remapping here to get orientation correct
            init_transform = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ])

            pos = np.zeros(3)
            rot_angles = []
            rot_order = ''

            for i, channel in enumerate(channels):
                if 'position' in channel.lower():
                    idx = 'XYZ'.index(channel[0])
                    pos[idx] = channel_values[i]
                elif 'rotation' in channel.lower():
                    rot_angles.append(channel_values[i])
                    rot_order += channel[0]

            rotation = self.get_rotation_matrix(rot_angles, rot_order)

            if joint_name == 'Hips':
                rotation = init_transform @ rotation
                pos = init_transform @ pos
                parent_rotation = init_transform @ parent_rotation
                offset = init_transform @ self.joint_offsets[joint_name]
            else:
                offset = self.joint_offsets[joint_name]

            global_rotation = parent_rotation @ rotation
            global_rotations[joint_name] = global_rotation

            if joint_name == 'Hips':  # Root joint
                position = pos + (parent_rotation @ offset)
            else:
                position = parent_position + (parent_rotation @ offset)

            positions[joint_name] = position

            for child, parent in self.joint_parents.items():
                if parent == joint_name:
                    process_joint(child, global_rotation, position)

        root = next(joint for joint, parent in self.joint_parents.items() if parent is None)
        process_joint(root)
        return positions

def create_animation(bvh_reader, output_file):
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()

        # Set white background
        ax.set_facecolor('white')

        # Remove all grid lines, panes, and axes
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax._axis3don = False

        positions = bvh_reader.get_joint_positions(frame)

        for joint, pos in positions.items():
            if not joint.endswith('end'):
                ax.scatter(*pos, c='red', marker='o', s=50)

                parent = bvh_reader.joint_parents[joint]
                if parent and parent in positions:
                    parent_pos = positions[parent]
                    ax.plot([pos[0], parent_pos[0]],
                           [pos[1], parent_pos[1]],
                           [pos[2], parent_pos[2]], 'b-', linewidth=2)

        all_positions = np.array(list(positions.values()))
        min_bounds = np.min(all_positions, axis=0)
        max_bounds = np.max(all_positions, axis=0)
        center = (min_bounds + max_bounds) / 2

        ranges = max_bounds - min_bounds
        max_range = np.max(ranges)
        padding = max_range * 0.2

        ax.set_xlim(center[0] - max_range/2 - padding, center[0] + max_range/2 + padding)
        ax.set_ylim(center[1] - max_range/2 - padding, center[1] + max_range/2 + padding)
        ax.set_zlim(min_bounds[2] - padding/2, max_bounds[2] + padding)

        ax.view_init(elev=15, azim=45)  # decent viewing angle

        ax.set_box_aspect([1, 1, 1])

    ani = FuncAnimation(fig, update, frames=bvh_reader.frames, interval=bvh_reader.frame_time*1000)
    writer = PillowWriter(fps=int(1/bvh_reader.frame_time))
    ani.save(output_file, writer=writer)
    plt.close()

def create_animation2(bvh_reader, output_file):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_facecolor('white')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        ax._axis3don = False
        positions = bvh_reader.get_joint_positions(frame)

        for joint, pos in positions.items():
            if not joint.endswith('end'):
                ax.scatter(*pos, c='red', marker='o', s=50)

                parent = bvh_reader.joint_parents[joint]
                if parent and parent in positions:
                    parent_pos = positions[parent]
                    ax.plot([pos[0], parent_pos[0]],
                           [pos[1], parent_pos[1]],
                           [pos[2], parent_pos[2]], 'b-', linewidth=2)

        all_positions = np.array(list(positions.values()))
        center = np.mean(all_positions, axis=0)
        max_range = np.max(np.max(all_positions, axis=0) - np.min(all_positions, axis=0))
        ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
        ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
        ax.set_zlim(center[2], center[2] + max_range)

        ax.view_init(elev=10, azim=45)


    ani = FuncAnimation(fig, update, frames=bvh_reader.frames, interval=bvh_reader.frame_time*1000)
    writer = PillowWriter(fps=int(1/bvh_reader.frame_time))
    ani.save(output_file, writer=writer)
    plt.close()

def process_bvh_to_gif(input_file, output_file):
    reader = BVHReader(input_file)
    create_animation(reader, output_file)


# process_bvh_to_gif('generated_motions_mlp_1/0_motion_epoch_4900.bvh', 'generated_motions_mlp_1/0_motion_epoch_4900.gif')


import os
directory = 'generated_motions_mlp_modified/'  # Change this to your target directory

for filename in os.listdir(directory):
    if filename.endswith(".bvh"):
        file_path = os.path.join(directory, filename)
        print("processing:", file_path)
        gif_name = file_path[:-4] + '.gif'
        process_bvh_to_gif(file_path, gif_name)





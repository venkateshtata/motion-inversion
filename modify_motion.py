import numpy as np

motion_arr = np.load('./data/test_edge_rot_data.npy', allow_pickle=True)

print('shape: ', motion_arr.shape)
print('arr[0]: ', motion_arr[0])
print('-------------------------------------')


motion_arr[0]['pos_root'][19] = [10, 1, 10]
motion_arr[0]['pos_root'][39] = [10, 1, 10]
motion_arr[0]['pos_root'][59] = [10, 1, 10]

np.save('./data/test_edge_rot_data_modified.npy', motion_arr, allow_pickle=True)
print("Updated array saved successfully!")

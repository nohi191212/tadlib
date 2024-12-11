import os

target_dir = "/home1/share/filtered/data_part2"

folder_list = sorted(os.listdir(target_dir))
num_video_per_folder = {} 
for folder in folder_list:
    folder_idx = int(folder.split("_")[1])
    num_video_per_folder[folder_idx] = len(os.listdir(os.path.join(target_dir, folder)))


for i in range(1, 23):
    print(f'2_{i}: {num_video_per_folder[i]}')    
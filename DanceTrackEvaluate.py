from demo import *
import os
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置device

path = '/mnt/disk_1/peilang/datasets/dancetrack/test'

video_names = [
    v for v in os.listdir(path) if v.startswith("dancetrack")
]

for video_name in video_names:
    print(f'----------tracking video {video_name}-----------------------')
    video_dir = f'{path}/{video_name}/img1'
    output_path = f'{path}/tracker/{video_name}.txt'
    video_save_dir = f'{path}/{video_name}/result'
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
    if os.path.exists(output_path):
        continue
    # if video_name=='dancetrack0011' or video_name=='dancetrack0038':
    #     continue
    sam2mot(video_dir, output_path, video_save_dir)
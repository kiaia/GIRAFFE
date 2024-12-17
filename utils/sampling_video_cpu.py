import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

input_dir = ''
output_dir = ''
target_fps = 2

os.makedirs(output_dir, exist_ok=True)

def process_video(video_file):
    input_path = os.path.join(input_dir, video_file)
    output_path = os.path.join(output_dir, video_file)
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / target_fps)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            out.write(frame)
        
        frame_count += 1
    
    cap.release()
    out.release()

if __name__ == '__main__':
    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    num_cores = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_video, video_file) for video_file in video_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="processing videos"):
            future.result()

    print("All tasks are done")

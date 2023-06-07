import torch
import torch.multiprocessing as mp
from torch import nn
import os
from tqdm.auto import tqdm
import cv2
from cv2 import dnn_superres
import moviepy
import moviepy.editor
import numpy as np
import sys
sys.path.append(os.path.abspath('./input/BSRGAN'))
from models.network_rrdbnet import RRDBNet as net

file_name = 'low_res_2' # 'Megan Is Missing'
ext = 'mp4'
scale = 4
input_video = f'./input/videos/{file_name}.{ext}'
output_root_path = f'./output/videos/{file_name}_multiproc'

os.makedirs(output_root_path, exist_ok=True)

input_video_parts_root = f'./input/videos/{file_name}_parts'
os.makedirs(input_video_parts_root, exist_ok=True)

threads = 2
runtime_env = {"working_dir": "./", "conda": "env_pytorch"}
num_cpus = threads
num_gpus = 1/threads

model_name = 'BSRGAN'
modelScale = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = f'./input/models/{model_name}.pth'
time_limit = 1*60 # seconds

torch.cuda.empty_cache()

def get_model():
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=modelScale)  # define network
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()
    model = model.to('cuda')
    return model

def get_output_file_path(file_name, model_name, file_part, parent_path, ext):
    file_prefix = f'{file_name}_{model_name}_{file_part}'
    output_path = f'{parent_path}/{file_prefix}.{ext}'
    return output_path

def split_video():
    vcap = cv2.VideoCapture(input_video)

    total_frames = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_per_part = total_frames//threads
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    frame_rate = vcap.get(cv2.CAP_PROP_FPS)
    success, frame = vcap.read()
    frame_count = 0
    vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    input_width = frame.shape[1]
    input_height = frame.shape[0]
    out_resolution = (input_width, input_height)

    for part in range(threads):
        part_path = f'{input_video_parts_root}/{file_name}_part{part}.{ext}'
        print(f'writing to {part_path}')
        vwriter = cv2.VideoWriter(part_path, codec,
                            frame_rate, out_resolution)
        start = int(frames_per_part * part)
        end = int(start + frames_per_part if part < threads-1 else total_frames)
        for i in range(start, end):
            success, frame = vcap.read()
            vwriter.write(frame)
        
        vwriter.release()
    
    vcap.release()

    file_parts = os.listdir(input_video_parts_root)
    num_frames = 0
    for file_part_ in file_parts:
        vs = cv2.VideoCapture(f'{input_video_parts_root}/{file_part_}')
        num_frames += vs.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f'total frames in input={total_frames}, total frames in parts={num_frames}')

def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)

def upscale(model, input_path, output_path):
    print(f'{input_path} --> {output_path}')
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_part_path = output_path  # get_output_file_path(file_name, model_name, f'part{part}', output_root_path, ext)
    
    input_part_path = input_path  # f'{input_video_parts_root}/{file_name}_part{part}.{ext}'
    vs = cv2.VideoCapture(input_part_path)
    total_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    print(vs, input_part_path, total_frames)
    # vs.set(cv2.CAP_PROP_POS_FRAMES, number_of_frames_completed)
    success, frame = vs.read()
    input_width = frame.shape[1] # 640
    input_height = frame.shape[0] # 352
    frame_count = 0 # number_of_frames_completed
    frame_rate = vs.get(cv2.CAP_PROP_FPS)
    out_resolution = (modelScale*input_width, modelScale*input_height)

    out = cv2.VideoWriter(output_part_path, codec,
                        frame_rate, out_resolution)

    print(f'input res: {frame.shape}, total frames={total_frames},\
        frame rate={frame_rate}, scale={modelScale},\
        outpath={output_part_path}')

    # with prof(period=0.001):
    with tqdm(total=total_frames) as pbar:
        # pbar.update(number_of_frames_completed)

        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = uint2tensor4(frame).to(device)
            upsampled_frame = model(frame)
            upsampled_frame = tensor2uint(upsampled_frame)
            upsampled_frame = cv2.cvtColor(upsampled_frame, cv2.COLOR_RGB2BGR)
            
            out.write(upsampled_frame)
            frame_count += 1
            pbar.update(1)
            pbar.set_description(f'frame no: {frame_count}')
            success, frame = vs.read()
            # if frame_count % 50:
            #     # check if run time > time_limit
            #     if time.time() - start_time > time_limit:
            #         break
                
        pbar.close()
    # prof.print_stats()

    vs.release()
    out.release()
    print(f'finished part={output_path}, {frame_count} frames ({frame_count/total_frames*100}%)')
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    num_processes = threads
    split_video()
    model = get_model()
    model.share_memory()
    # print(model)

    processes = []
    for rank in range(num_processes):
        output_path = get_output_file_path(file_name, model_name, f'part{rank}', output_root_path, ext)
        input_path = f'{input_video_parts_root}/{file_name}_part{rank}.{ext}'
        p = mp.Process(target=upscale, args=(model, input_path, output_path))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
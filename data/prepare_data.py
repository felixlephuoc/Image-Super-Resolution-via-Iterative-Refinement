import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from glob import glob
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, videos_path_subset):
    for video_path in videos_path_subset:
        video_id = os.path.basename(video_path)
        for file in Path(video_path).rglob('*.png'):
            if not file.is_file():
                continue
            i, imgs = wctx.resize_fn(file)
            lr_img, hr_img, sr_img = imgs
            
            # create the output directory
            os.makedirs('{}/lr_{}/{}'.format(wctx.out_path, sizes[0], video_id), exist_ok=True)
            os.makedirs('{}/hr_{}/{}'.format(wctx.out_path, sizes[1], video_id), exist_ok=True)
            os.makedirs('{}/sr_{}_{}/{}'.format(wctx.out_path, sizes[0], sizes[1], video_id), exist_ok=True)
            
            if not wctx.lmdb_save:
                lr_img.save(
                    '{}/lr_{}/{}/{}.png'.format(wctx.out_path, wctx.sizes[0],video_id, i.zfill(5)))
                hr_img.save(
                    '{}/hr_{}/{}/{}.png'.format(wctx.out_path, wctx.sizes[1],video_id, i.zfill(5)))
                sr_img.save(
                    '{}/sr_{}_{}/{}/{}.png'.format(wctx.out_path, wctx.sizes[0], wctx.sizes[1],video_id, i.zfill(5)))
            else:
                with wctx.env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
            curr_total = wctx.inc_get()
            if wctx.lmdb_save:
                with wctx.env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(frame_path, out_path, n_worker, n_frame, sizes=(16, 128), resample=Image.BICUBIC, lmdb_save=False):
    
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    
    videos_path = [os.path.join(frame_path, video_id) for video_id in os.listdir(frame_path) if os.path.isdir(os.path.join(frame_path, video_id))]

    if not lmdb_save: 
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        videos_path_subsets = np.array_split(videos_path, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, videos_path_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_videos_path = len(videos_path)
        total_num_frames = int(total_videos_path * n_frame)
        while not all_threads_inactive(worker_threads):
            print("\r {}/{} frames processed".format(wctx.value(), total_num_frames), end=" ")
            time.sleep(0.1)

    else:  
        total = 0
        for video_path in videos_path:
            video_id = os.path.basename(video_path)
            for file in Path(video_path).rglob('*.jpg'):
                if not file.is_file():
                    continue
                i, imgs = resize_fn(file)
                lr_img, hr_img, sr_img = imgs
                
                # create the output directory
                os.makedirs('{}/lr_{}/{}'.format(out_path, sizes[0], video_id), exist_ok=True)
                os.makedirs('{}/hr_{}/{}'.format(out_path, sizes[1], video_id), exist_ok=True)
                os.makedirs('{}/sr_{}_{}/{}'.format(out_path, sizes[0], sizes[1], video_id), exist_ok=True)
                    
                if not lmdb_save:
                    lr_img.save(
                        '{}/lr_{}/{}/{}.png'.format(out_path, sizes[0],video_id, i.zfill(5)))
                    hr_img.save(
                        '{}/hr_{}/{}/{}.png'.format(out_path, sizes[1], video_id, i.zfill(5)))
                    sr_img.save(
                        '{}/sr_{}_{}/{}/{}.png'.format(out_path, sizes[0], sizes[1], video_id, i.zfill(5)))
                else:
                    with env.begin(write=True) as txn:
                        txn.put('lr_{}_{}'.format(
                            sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                        txn.put('hr_{}_{}'.format(
                            sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                        txn.put('sr_{}_{}_{}'.format(
                            sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
                total += 1
                if lmdb_save:
                    with env.begin(write=True) as txn:
                        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_path', type=str,
                        default='sample_frames',help="Path to the video_id directories containing extracted frames")
    parser.add_argument('--n_frames', type=int, default=3, help="Number of frames extracted from each video")
    parser.add_argument('--interval', type=float, default=1.0, help="The temporal stride, or temporal distance between extracted frames")
    parser.add_argument('--split', type=str, choices=['train', 'val'], help="create train or validation dataset", default='train')
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/celebvhq')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_frames_{}_seconds_{}_{}_{}'.format(args.out, args.n_frames, args.interval, sizes[0], sizes[1], args.split)
    #print(args.out)
    prepare(args.frames_path, args.out, args.n_worker, args.n_frames,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)

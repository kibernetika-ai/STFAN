import logging
import time
import argparse
from datetime import datetime as dt

import cv2
import numpy as np
import torch

from models import DeblurNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output')
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


def normalize(img: np.ndarray):
    # rank = len(img.shape)
    # height_dim = 1 if rank == 4 else 0
    # nearest_multiple_16 = img.shape[height_dim] // 16 * 16
    # if nearest_multiple_16 != img.shape[height_dim]:
    #     # crop by height
    #     crop_need = img.shape[height_dim] - nearest_multiple_16
    #     if rank == 4:
    #         img = img[:, crop_need // 2:-crop_need // 2, :, :]
    #     else:
    #         img = img[crop_need // 2:-crop_need // 2, :, :]

    img = img.astype(np.float32) / 255.0
    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img).float()


def denormalize(tensor):
    numpy = tensor.detach().cpu().numpy()
    img = (numpy * 255.0).clip(0, 255).astype(np.uint8)
    return img.transpose([1, 2, 0])


def main():
    args = parse_args()

    print(f'Loading model from {args.model}...')
    model = DeblurNet.DeblurNet()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    print('[INFO] %s Recovering from %s ...' % (dt.now(), args.model))
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['deblurnet_state_dict'])
    # deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
    init_epoch = checkpoint['epoch_idx'] + 1
    best_img_psnr = checkpoint['Best_Img_PSNR']
    best_epoch = checkpoint['Best_Epoch']
    print(
        f'[INFO] {dt.now()} Recover complete. Current epoch #{init_epoch}, '
        f'Best_Img_PSNR = {best_img_psnr} at epoch #{best_epoch}.'
    )

    model.eval()
    print(f'Done.')

    vc = cv2.VideoCapture(args.video)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.output:
        fps = vc.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_format = video.get(cv2.CAP_PROP_FORMAT)
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps,
            frameSize=(width, height)
        )

    log_frames = 100
    frame_num = 0
    frame_processed = 0
    time_sum = 0
    last_img_blur = None
    output_last_img = None
    output_last_fea = None
    with torch.no_grad():
        while True:
            ret, frame = vc.read()
            if not ret:
                break

            frame_num += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_blur = normalize(frame)
            if use_cuda:
                img_blur = img_blur.cuda(non_blocking=True)

            if last_img_blur is None:
                last_img_blur = img_blur
                output_last_img = img_blur
            # if len(imgs_in) <= n_frames:
            #     imgs_in.append(img)
            #     if len(imgs_in) < n_frames:
            #         continue
            #     if len(imgs_in) > n_frames:
            #         imgs_in = [imgs_in[-1]]
            #         continue

            t = time.time()
            output_img, output_fea = model(img_blur, last_img_blur, output_last_img, output_last_fea)
            time_sum += time.time() - t

            output_frame = denormalize(output_img[0])

            # *** Update output_last_img/feature ***
            last_img_blur = img_blur
            output_last_img = output_img.clamp(0.0, 1.0)
            output_last_fea = output_fea

            frame_processed += 1
            if frame_processed % log_frames == 0:
                print(f'Processed {frame_processed} frames.')

            cv_frame = output_frame[:, :, ::-1]
            if args.output:
                video_writer.write(cv_frame)
            if args.show:
                cv2.imshow('Video', cv_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

    print(f'Total time: {time_sum:0.3f}s')
    print(f'Total frames: {frame_processed}')
    print(f'Average inference time: {time_sum / frame_processed * 1000:0.3f}ms')
    vc.release()
    if args.output:
        video_writer.release()


if __name__ == '__main__':
    main()

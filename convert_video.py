import time
import argparse
from datetime import datetime as dt
import logging

import cv2
import numpy as np
import torch

from models import DeblurNet


LOG = logging.getLogger(__name__)


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
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
    args = parse_args()

    LOG.info(f'Loading model from {args.model}...')
    model = DeblurNet.DeblurNet()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
    if use_cuda:
        model.cuda()
    checkpoint = torch.load(args.model, map_location=torch.device('cuda' if use_cuda else 'cpu'))
    model.load_state_dict(checkpoint['deblurnet_state_dict'])
    # deblurnet_solver.load_state_dict(checkpoint['deblurnet_solver_state_dict'])
    init_epoch = checkpoint['epoch_idx'] + 1
    best_img_psnr = checkpoint['Best_Img_PSNR']
    best_epoch = checkpoint['Best_Epoch']
    LOG.info(
        f'Recover complete. Current epoch #{init_epoch}, '
        f'Best_Img_PSNR = {best_img_psnr} at epoch #{best_epoch}.'
    )

    model.eval()
    LOG.info(f'Done.')

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
    sequence_len = 20

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
            # if use_cuda:
            #     img_blur = img_blur.cuda(non_blocking=True)

            if last_img_blur is None:
                last_img_blur = img_blur
                output_last_img = img_blur

            t = time.time()
            output_img, output_fea = model(img_blur, last_img_blur, output_last_img, output_last_fea)
            torch.cuda.synchronize()
            time_sum += time.time() - t

            output_frame = denormalize(output_img[0])

            # *** Update output_last_img/feature ***
            last_img_blur = img_blur
            output_last_img = output_img.clamp(0.0, 1.0)
            output_last_fea = output_fea

            frame_processed += 1
            if frame_processed % log_frames == 0:
                LOG.info(f'Processed {frame_processed} frames.')
            if frame_processed % sequence_len == 0:
                last_img_blur = None
                output_last_img = None
                output_last_fea = None

            cv_frame = output_frame[:, :, ::-1]
            if args.output:
                video_writer.write(cv_frame)
            if args.show:
                cv2.imshow('Video', cv_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break

    LOG.info(f'Total time: {time_sum:0.3f}s')
    LOG.info(f'Total frames: {frame_processed}')
    LOG.info(f'Average inference time: {time_sum / frame_processed * 1000:0.3f}ms')
    vc.release()
    if args.output:
        video_writer.release()


if __name__ == '__main__':
    main()

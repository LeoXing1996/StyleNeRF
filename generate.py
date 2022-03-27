# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Generate images using pretrained network pickle."""

from functools import partial
import glob
import math
import os
import re
from typing import List, Optional

import click
import imageio
import numpy as np
import PIL.Image
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

import dnnlib
import legacy
from renderer import Renderer

# ----------------------------------------------------------------------------


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a
    range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def float_range(s: str):
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [float(x) for x in vals]


def get_factor_close_to_sqr(num, factor):
    sqr = math.ceil(math.sqrt(num))
    for n in range(sqr, 0, -1):
        if num % n == 0 and n % factor == 0:
            return n


def proc_img(img, nrows=None, padding=2, col_scale=1):
    if nrows is None:
        bz = img.size(0)
        ncols = get_factor_close_to_sqr(bz * col_scale, factor=col_scale)
        # make `n_cols` larger than `nrows`
        nrows = bz // ncols

    # rescale and clip
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

    # make grid
    img = make_grid(img, nrows, padding=padding).permute(1, 2, 0)
    return img.cpu().numpy()


def proc_and_save(img_list, img_dir, proc_fn):

    imgs = [proc_fn(img) for img in img_list]

    os.makedirs(img_dir, exist_ok=True)
    for step, img in enumerate(tqdm(imgs)):
        PIL.Image.fromarray(img, 'RGB').save(f'{img_dir}/{step:03d}.png')
    return imgs


# ----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'


@click.command()
@click.pass_context
@click.option('--network',
              'network_pkl',
              help='Network pickle filename',
              required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--bz', type=int, default=2, help='The batch size.')
@click.option('--nrows',
              type=int,
              default=None,
              help='Number of rows when vis grid')
@click.option('--trunc',
              'truncation_psi',
              type=float,
              help='Truncation psi',
              default=1,
              show_default=True)
@click.option('--class',
              'class_idx',
              type=int,
              help='Class label (unconditional if not specified)')
@click.option('--noise-mode',
              help='Noise mode',
              type=click.Choice(['const', 'random', 'none']),
              default='const',
              show_default=True)
@click.option('--projected-w',
              help='Projection result file',
              type=str,
              metavar='FILE')
@click.option('--outdir',
              help='Where to save the output images',
              type=str,
              default='render_out',
              metavar='DIR')
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option',
              default=None,
              type=str,
              help="e.g. up_256, camera, depth")
@click.option('--n_steps',
              default=8,
              type=int,
              help="number of steps for each seed")
@click.option('--no-video', default=False)
@click.option('--relative_range_u_scale',
              default=1.0,
              type=float,
              help="relative scale on top of the original range u")
@click.option('--res-scale',
              type=float_range,
              default=None,
              help=('the scale of resolution to evaluate. sample from '
                    '[res_scale[0] * curr_res, res_scale[1] * curr_res]'))
@click.option('--num-res',
              type=int,
              default=1,
              help='The number of resolution used in evaluation.')
@click.option('--save-resample',
              is_flag=True,
              help='The number of resolution used in evaluation.')
def generate_images(ctx: click.Context,
                    network_pkl: str,
                    seeds: Optional[List[int]],
                    truncation_psi: float,
                    noise_mode: str,
                    outdir: str,
                    bz: Optional[int],
                    nrows: Optional[int],
                    class_idx: Optional[int],
                    projected_w: Optional[str],
                    render_program=None,
                    render_option=None,
                    n_steps=8,
                    no_video=False,
                    relative_range_u_scale=1.0,
                    res_scale=None,
                    num_res=1,
                    save_resample=False):

    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device)  # type: ignore
        D = network['D'].to(device)
    # from fairseq import pdb;pdb.set_trace()
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with '
                     '--class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an '
                  'unconditional network')

    # avoid persistent classes...
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    from training.networks import Generator
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        # D2 = Discriminator(*D.init_args, **D.init_kwargs).to(device)
        # misc.copy_params_and_buffers(D, D2, require_all=False)

    if (res_scale is None) or (num_res <= 1):
        res_range = [1]
    else:
        assert len(res_scale) == 2
        res_range = np.linspace(*res_scale, num=num_res).tolist()
        if not (1 in res_range):
            res_range = [1, *res_range]
    G2 = Renderer(G2,
                  D,
                  program=render_program,
                  res_range=res_range,
                  save_resample=save_resample)

    network_pkl_name = network_pkl.split('/')[-1].split('.')[0]
    # Generate images.
    all_imgs = []
    all_imgs_resample = []

    if projected_w is not None:
        ws = np.load(projected_w)
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        img = G2(
            styles=ws,
            truncation_psi=truncation_psi,
            noise_mode=noise_mode,
            render_option=render_option)
        assert isinstance(img, List)
        imgs = [proc_img(i, nrows) for i in img]
        all_imgs += [imgs]

    else:
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' %
                  (seed, seed_idx, len(seeds)))
            G2.set_random_seed(seed)
            z = torch.from_numpy(
                np.random.RandomState(seed).randn(bz, G.z_dim)).to(device)
            relative_range_u = [
                0.5 - 0.5 * relative_range_u_scale,
                0.5 + 0.5 * relative_range_u_scale
            ]
            outputs = G2(z=z,
                         c=label,
                         batch_size=bz,
                         truncation_psi=truncation_psi,
                         noise_mode=noise_mode,
                         render_option=render_option,
                         n_steps=n_steps,
                         relative_range_u=relative_range_u,
                         return_cameras=True)
            if isinstance(outputs, tuple):
                if len(outputs) == 3:
                    img, img_resample, cameras = outputs
                else:
                    img, cameras = outputs
                    img_resample = None
            else:
                img = outputs
                img_resample = None

            # get folder
            proj_dir_name = (f'{network_pkl_name}-'
                             f'trunc_{truncation_psi:.2f}'
                             f'_seed_{seed:0>6d}-bz_{bz:0>2d}'
                             f'-n_steps_{n_steps:0>3d}')
            if len(res_range) > 1:
                proj_dir_name = proj_dir_name + f'-res_scale_{res_range}'

            curr_out_dir = os.path.join(outdir, render_program, proj_dir_name)
            os.makedirs(curr_out_dir, exist_ok=True)

            # save render options
            if (render_option is not None) and ("gen_ibrnet_metadata"
                                                in render_option):
                intrinsics = []
                poses = []
                _, H, W, _ = imgs[0].shape
                for i, camera in enumerate(cameras):
                    intri, pose, _, _ = camera
                    focal = (H - 1) * 0.5 / intri[0, 0, 0].item()
                    intri = np.diag([focal, focal, 1.0,
                                     1.0]).astype(np.float32)
                    intri[0, 2], intri[1, 2] = (W - 1) * 0.5, (H - 1) * 0.5

                    pose = pose.squeeze().detach().cpu().numpy() @ np.diag(
                        [1, -1, -1, 1]).astype(np.float32)
                    intrinsics.append(intri)
                    poses.append(pose)

                intrinsics = np.stack(intrinsics, axis=0)
                poses = np.stack(poses, axis=0)

                np.savez(os.path.join(curr_out_dir, 'cameras.npz'),
                         intrinsics=intrinsics,
                         poses=poses)
                with open(os.path.join(curr_out_dir, 'meta.conf'), 'w') as f:
                    f.write('depth_range = '
                            f'{G2.generator.synthesis.depth_range}\n'
                            f'test_hold_out = 2\n'
                            f'height = {H}\nwidth = {W}')

            # save img list
            proc_fn = partial(proc_img, nrows=nrows, col_scale=len(res_range))
            img_dir = os.path.join(curr_out_dir, 'images_raw')
            imgs = proc_and_save(img, img_dir, proc_fn)
            if not no_video:
                all_imgs += [imgs]

            # handle img resample
            if img_resample:
                img_resample_dir = os.path.join(curr_out_dir,
                                                'images_resample')
                imgs_resample = proc_and_save(img_resample, img_resample_dir,
                                              proc_fn)
                if not no_video:
                    all_imgs_resample += [imgs_resample]

    if len(all_imgs) > 0 and (not no_video):
        for idx, seed in enumerate(seeds):
            all_frames = all_imgs[idx]
            vid_name = f'{curr_out_dir}/{network_pkl_name}_{seed}.mp4'
            imageio.mimwrite(vid_name, all_frames, fps=30, quality=8)

    if len(all_imgs_resample) > 0 and (not no_video):
        for idx, seed in enumerate(seeds):
            all_frames = all_imgs_resample[idx]
            vid_name = f'{curr_out_dir}/{network_pkl_name}_{seed}_resample.mp4'
            imageio.mimwrite(vid_name, all_frames, fps=30, quality=8)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------

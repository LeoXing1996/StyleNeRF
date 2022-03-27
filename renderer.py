# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrap the generator to render a sequence of images"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import trimesh


class Renderer(object):

    def __init__(self,
                 generator,
                 discriminator=None,
                 program=None,
                 res_range=[1]):
        self.generator = generator
        self.discriminator = discriminator
        self.sample_tmp = 0.65
        self.program = program
        self.seed = 0

        if (program is not None) and (len(program.split(':')) == 2):
            from training.dataset import ImageFolderDataset
            self.image_data = ImageFolderDataset(program.split(':')[1])
            self.program = program.split(':')[0]
        else:
            self.image_data = None

        self.res_range = res_range
        if self.res_range is not None:
            self.res_range.sort()

    def set_random_seed(self, seed):
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        self.generator.eval()  # eval mode...

        if len(self.res_range) > 1:
            # resolution more than 1
            return self.render_super_resolution(*args, **kwargs)

        if self.program is None:
            if hasattr(self.generator, 'get_final_output'):
                return self.generator.get_final_output(*args, **kwargs)
            return self.generator(*args, **kwargs)

        if self.image_data is not None:
            batch_size = 1
            indices = (np.random.rand(batch_size) *
                       len(self.image_data)).tolist()
            rimages = np.stack(
                [self.image_data._load_raw_image(int(i)) for i in indices], 0)
            rimages = torch.from_numpy(rimages).float().to(
                kwargs['z'].device) / 127.5 - 1
            kwargs['img'] = rimages

        outputs = getattr(self, f"render_{self.program}")(*args, **kwargs)

        if self.image_data is not None:
            imgs = outputs if not isinstance(outputs, tuple) else outputs[0]
            size = imgs[0].size(-1)
            rimg = F.interpolate(rimages, (size, size),
                                 mode='bicubic',
                                 align_corners=False)
            imgs = [torch.cat([img, rimg], 0) for img in imgs]
            outputs = imgs if not isinstance(outputs, tuple) else (imgs,
                                                                   outputs[1])
        return outputs

    def get_additional_params(self, ws, t=0):
        gen = self.generator.synthesis
        batch_size = ws.size(0)

        kwargs = {}
        if not hasattr(gen, 'get_latent_codes'):
            return kwargs

        s_val, t_val, r_val = [[0, 0, 0]], [[0.5, 0.5, 0.5]], [0.]
        # kwargs["transformations"] = gen.get_transformations(batch_size=batch_size, mode=[s_val, t_val, r_val], device=ws.device)  # noqa
        # kwargs["bg_rotation"] = gen.get_bg_rotation(batch_size, device=ws.device)  # noqa
        # kwargs["light_dir"] = gen.get_light_dir(batch_size, device=ws.device)  # noqa
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size,
                                                      tmp=self.sample_tmp,
                                                      device=ws.device)
        kwargs["camera_matrices"] = self.get_camera_traj(t,
                                                         ws.size(0),
                                                         device=ws.device)
        return kwargs

    def get_camera_traj(self,
                        t,
                        batch_size=1,
                        traj_type='pigan',
                        device='cpu'):
        gen = self.generator.synthesis
        if traj_type == 'pigan':
            range_u, range_v = gen.C.range_u, gen.C.range_v
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi / 2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
            cam = gen.get_camera(batch_size=batch_size,
                                 mode=[u, v, 0.5],
                                 device=device)
        else:
            raise NotImplementedError
        return cam

    def _fill_on_canvas(self, img_list, canvas_size):
        """The helper function to paste a image on a large canvas
        Args:
            img_list (list): A list of image generated with same latent code
                and different camera parameters.
            canvas_size (int): The size of the canvas. We only support square
                canvas.

        Returns:
            torch.Tensor: Image tensor pasted on a large canvas.
        """
        bz, res = img_list[0].size(0), img_list[0].size(-1)
        canvas_shape = [bz, 3, canvas_size, canvas_size]

        s_idx = (canvas_size - res) // 2
        img_list_new = []

        for img in img_list:
            canvas = torch.zeros(*canvas_shape)
            canvas[..., s_idx:s_idx + res, s_idx:s_idx + res] = img.cpu()
            img_list_new.append(canvas)

        return img_list_new

    def render_super_resolution(self, *args, **kwargs):
        """Render with image-plane super resolution"""
        gen = self.generator.synthesis

        curr_res = gen.img_resolution
        vol_res = gen.resolution_vol
        canavs_size = int(max(self.res_range) * curr_res)

        out_dict = dict()
        for res_scale in self.res_range:
            # avoid forward same resolution duplicate times
            if res_scale in out_dict:
                continue

            gen.resolution_vol = int(res_scale * vol_res)
            out = getattr(self, f"render_{self.program}")(*args, **kwargs)

            if isinstance(out, tuple):
                img, cam = out
                out_dict['camera'] = cam
            else:
                img = out
            img = self._fill_on_canvas(img, canavs_size)
            out_dict[res_scale] = img

            torch.cuda.empty_cache()

        # cat put one res at the same row
        img_list = []
        for step in range(len(img)):
            # concencate at column dimension
            img_comb = torch.cat(
                [out_dict[res][step] for res in self.res_range], dim=-1)
            img_list.append(img_comb)

        if 'camera' in out_dict:
            return img_list, out_dict['camera']
        return img_list

    def render_rotation_camera(self, *args, **kwargs):
        batch_size, n_steps = kwargs['batch_size'], kwargs["n_steps"]
        gen = self.generator.synthesis

        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        # ws = ws.repeat(batch_size, 1, 1)

        # kwargs["not_render_background"] = True
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size,
                                                          tmp=self.sample_tmp,
                                                          device=ws.device)
            kwargs.pop('img', None)

        out = []
        cameras = []
        relatve_range_u = kwargs['relative_range_u']
        u_samples = np.linspace(relatve_range_u[0], relatve_range_u[1],
                                n_steps)
        for step in tqdm.tqdm(range(n_steps)):
            # Set Camera
            u = u_samples[step]
            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size,
                                                       mode=[u, 0.5, 0.5],
                                                       device=ws.device)
            cameras.append(
                gen.get_camera(batch_size=batch_size,
                               mode=[u, 0.5, 0.5],
                               device=ws.device))
            with torch.no_grad():
                out_i = gen(ws, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)

        if 'return_cameras' in kwargs and kwargs["return_cameras"]:
            return out, cameras
        else:
            return out

    def render_rotation_camera3(self, styles=None, *args, **kwargs):
        """rendero with 3D rotation."""
        gen = self.generator.synthesis
        # n_steps = 36  # 120
        n_steps = kwargs['n_steps']

        if styles is None:
            batch_size = kwargs['batch_size']
            if 'img' not in kwargs:
                ws = self.generator.mapping(*args, **kwargs)
            else:
                ws = self.generator.encoder(kwargs['img'])['ws']
            # ws = ws.repeat(batch_size, 1, 1)
        else:
            ws = styles
            batch_size = ws.size(0)
            assert batch_size == kwargs['batch_size']

        # kwargs["not_render_background"] = True
        # Get Random codes and bg rotation
        self.sample_tmp = 0.72
        if hasattr(gen, 'get_latent_codes'):
            kwargs["latent_codes"] = gen.get_latent_codes(batch_size,
                                                          tmp=self.sample_tmp,
                                                          device=ws.device)
            kwargs.pop('img', None)

        # TODO: what does the following code do?
        # if getattr(gen, "use_noise", False):
        #     from dnnlib.geometry import extract_geometry
        #     kwargs['meshes'] = {}
        #     low_res, high_res = gen.resolution_vol, gen.img_resolution
        #     res = low_res * 2
        #     while res <= high_res:
        #         kwargs['meshes'][res] = [trimesh.Trimesh(*extract_geometry(gen, ws, resolution=res, threshold=30.))]  # noqa
        #         kwargs['meshes'][res] += [
        #             torch.randn(len(kwargs['meshes'][res][0].vertices),
        #                 2, device=ws.device)[kwargs['meshes'][res][0].faces]]
        #         res = res * 2
        # if getattr(gen, "use_noise", False):
        #     kwargs['voxel_noise'] = gen.get_voxel_field(styles=ws, n_vols=2048, return_noise=True, sphere_noise=True)  # noqa
        # if getattr(gen, "use_voxel_noise", False):
        #     kwargs['voxel_noise'] = gen.get_voxel_field(styles=ws, n_vols=128, return_noise=True)  # noqa

        kwargs['noise_mode'] = 'const'

        out = []
        tspace = np.linspace(0, 1, n_steps)
        range_u, range_v = gen.C.range_u, gen.C.range_v

        for step in tqdm.tqdm(range(n_steps)):
            t = tspace[step]
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi / 2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])

            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size,
                                                       mode=[u, v, t],
                                                       device=ws.device)

            with torch.no_grad():
                out_i = gen(ws, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']
            out.append(out_i)
        return out

    def render_rotation_both(self, *args, **kwargs):
        gen = self.generator.synthesis
        # batch_size, n_steps = 1, 36
        batch_size, n_steps = kwargs['batch_size'], kwargs['n_stpes']
        if 'img' not in kwargs:
            ws = self.generator.mapping(*args, **kwargs)
        else:
            ws, _ = self.generator.encoder(kwargs['img'])
        ws = ws.repeat(batch_size, 1, 1)

        # kwargs["not_render_background"] = True
        # Get Random codes and bg rotation
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size,
                                                      tmp=self.sample_tmp,
                                                      device=ws.device)
        kwargs.pop('img', None)

        out = []
        tspace = np.linspace(0, 1, n_steps)
        range_u, range_v = gen.C.range_u, gen.C.range_v

        for step in tqdm.tqdm(range(n_steps)):
            t = tspace[step]
            pitch = 0.2 * np.cos(t * 2 * np.pi) + np.pi / 2
            yaw = 0.4 * np.sin(t * 2 * np.pi)
            u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])

            kwargs["camera_matrices"] = gen.get_camera(batch_size=batch_size,
                                                       mode=[u, v, 0.5],
                                                       device=ws.device)

            with torch.no_grad():
                out_i = gen(ws, **kwargs)
                if isinstance(out_i, dict):
                    out_i = out_i['img']

                kwargs_n = copy.deepcopy(kwargs)
                kwargs_n.update(
                    {'render_option': 'early,no_background,up64,depth,normal'})
                out_n = gen(ws, **kwargs_n)
                out_n = F.interpolate(out_n,
                                      size=(out_i.size(-1), out_i.size(-1)),
                                      mode='bicubic',
                                      align_corners=True)
                out_i = torch.cat([out_i, out_n], 0)
            out.append(out_i)
        return out

    def render_rotation_grid(self,
                             styles=None,
                             return_cameras=False,
                             *args,
                             **kwargs):
        gen = self.generator.synthesis
        if styles is None:
            # batch_size = 1
            batch_size = kwargs['batch_size']
            ws = self.generator.mapping(*args, **kwargs)
            ws = ws.repeat(batch_size, 1, 1)
        else:
            ws = styles
            batch_size = ws.size(0)
            assert batch_size == kwargs['batch_size']

        kwargs["latent_codes"] = gen.get_latent_codes(batch_size,
                                                      tmp=self.sample_tmp,
                                                      device=ws.device)
        kwargs.pop('img', None)

        if getattr(gen, "use_voxel_noise", False):
            kwargs['voxel_noise'] = gen.get_voxel_field(styles=ws,
                                                        n_vols=128,
                                                        return_noise=True)

        out = []
        cameras = []
        range_u, range_v = gen.C.range_u, gen.C.range_v

        a_steps, b_steps = 6, 3
        aspace = np.linspace(-0.4, 0.4, a_steps)
        bspace = np.linspace(-0.2, 0.2, b_steps) * -1
        for b in tqdm.tqdm(range(b_steps)):
            for a in range(a_steps):
                t_a = aspace[a]
                t_b = bspace[b]
                camera_mat = gen.camera_matrix.repeat(batch_size, 1,
                                                      1).to(ws.device)
                loc_x = np.cos(t_b) * np.cos(t_a)
                loc_y = np.cos(t_b) * np.sin(t_a)
                loc_z = np.sin(t_b)
                loc = torch.tensor([[loc_x, loc_y, loc_z]],
                                   dtype=torch.float32).to(ws.device)
                from dnnlib.camera import look_at
                R = look_at(loc)
                RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
                RT[:, :3, :3] = R
                RT[:, :3, -1] = loc

                world_mat = RT.to(ws.device)
                # kwargs["camera_matrices"] = gen.get_camera(
                #     batch_size=batch_size, mode=[u, v, 0.5], device=ws.device)  # noqa
                kwargs["camera_matrices"] = (camera_mat, world_mat, "random",
                                             None)

                with torch.no_grad():
                    out_i = gen(ws, **kwargs)
                    if isinstance(out_i, dict):
                        out_i = out_i['img']

                    # kwargs_n = copy.deepcopy(kwargs)
                    # kwargs_n.update({'render_option': 'early,no_background,up64,depth,normal'})  # noqa
                    # out_n = gen(ws, **kwargs_n)
                    # out_n = F.interpolate(out_n,
                    #                       size=(out_i.size(-1), out_i.size(-1)),  # noqa
                    #                       mode='bicubic', align_corners=True)  # noqa
                    # out_i = torch.cat([out_i, out_n], 0)
                out.append(out_i)

        if return_cameras:
            return out, cameras
        else:
            return out

    def render_rotation_camera_grid(self, *args, **kwargs):
        # batch_size, n_steps = 1, 60
        batch_size, n_steps = kwargs['batch_size'], kwargs['n_steps']
        gen = self.generator.synthesis
        bbox_generator = self.generator.synthesis.boundingbox_generator

        ws = self.generator.mapping(*args, **kwargs)
        ws = ws.repeat(batch_size, 1, 1)

        # Get Random codes and bg rotation
        kwargs["latent_codes"] = gen.get_latent_codes(batch_size,
                                                      tmp=self.sample_tmp,
                                                      device=ws.device)
        del kwargs['render_option']

        out = []
        for v in [0.15, 0.5, 1.05]:
            for step in tqdm.tqdm(range(n_steps)):
                # Set Camera
                u = step * 1.0 / (n_steps - 1) - 1.0
                kwargs["camera_matrices"] = gen.get_camera(
                    batch_size=batch_size, mode=[u, v, 0.5], device=ws.device)
                with torch.no_grad():
                    out_i = gen(ws, render_option=None, **kwargs)
                    if isinstance(out_i, dict):
                        out_i = out_i['img']
                    # option_n = 'early,no_background,up64,depth,direct_depth'
                    # option_n = 'early,up128,no_background,depth,normal'
                    # out_n = gen(ws, render_option=option_n, **kwargs)
                    # out_n = F.interpolate(out_n,
                    #     size=(out_i.size(-1), out_i.size(-1)),
                    #     mode='bicubic', align_corners=True)
                    # out_i = torch.cat([out_i, out_n], 0)

                out.append(out_i)

        # out += out[::-1]
        return out

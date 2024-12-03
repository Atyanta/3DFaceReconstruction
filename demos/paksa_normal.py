import os
import torch
import cv2
import imageio
import numpy as np
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.render import batch_euler2axis, deg2rad


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)
    
    # DECA
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    visdict_list_list = []
    for i in range(len(testdata)):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)  # tensor
        
        # Menjaga pose dan ekspresi wajah normal (netral)
        # Set pose menjadi netral (yaw=0, pitch=0, roll=0)
        euler_pose = torch.zeros((1, 3)).to(device)
        global_pose = batch_euler2axis(deg2rad(euler_pose[:, :3].cuda()))  # Pose normal
        codedict['pose'][:, :3] = global_pose
        codedict['cam'][:, :] = 0.
        codedict['cam'][:, 0] = 8

        # Setelah memastikan pose wajah normal, kita ambil gambar dengan pose ini
        _, visdict_view = deca.decode(codedict)
        visdict = {x: visdict[x] for x in ['inputs', 'shape_detail_images']}
        visdict['pose'] = visdict_view['shape_detail_images']

        # Menambahkan ke list
        visdict_list = [visdict]
        
        # Transfer ekspresi wajah
        for (i, k) in enumerate(range(len(expdata))):
            exp_images = expdata[i]['image'].to(device)[None,...]
            exp_codedict = deca.encode(exp_images)

            # Transfer ekspresi dari gambar ekspresi ke gambar input
            codedict['pose'][:, 3:] = exp_codedict['pose'][:, 3:]  # Transfer pose wajah ekspresi
            codedict['exp'] = exp_codedict['exp']  # Transfer ekspresi wajah

            # Hasil transfer ekspresi
            _, exp_visdict = deca.decode(codedict)
            visdict_list[i]['exp'] = exp_visdict['shape_detail_images']
        
        visdict_list_list.append(visdict_list)
    
    # Menulis gambar GIF hasil
    writer = imageio.get_writer(os.path.join(savefolder, 'teaser.gif'), mode='I')
    yaw_list = list(range(0, 31, 5)) + list(range(30, -31, -5))  # Yaw angles from -30 to 30
    for i in range(len(yaw_list)):
        grid_image_list = []
        for j in range(len(testdata)):
            grid_image = deca.visualize(visdict_list_list[j][i])
            grid_image_list.append(grid_image)
        grid_image_all = np.concatenate(grid_image_list, 0)
        grid_image_all = rescale(grid_image_all, 0.6, multichannel=True)  # Resize for showing in github
        writer.append_data(grid_image_all[:, :, [2, 1, 0]])

    print(f'-- please check the teaser figure in {savefolder}')


def rescale(image, scale, multichannel=True):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    # Path untuk gambar input dan ekspresi
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str, help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/7.jpg', type=str, help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str, help='output directory to save results')
    parser.add_argument('--device', default='cuda', type=str, help='set device, cpu for using cpu')

    # Rendering options
    parser.add_argument('--rasterizer_type', default='standard', type=str, help='rasterizer type: pytorch3d or standard')

    # Proses gambar test
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'], help='whether to crop input image')
    parser.add_argument('--detector', default='fan', type=str, help='detector for cropping face')

    # Pilihan untuk menyimpan hasil
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'], help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'], help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'], help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'], help='whether to save outputs as .obj')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'], help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'], help='whether to save visualization output as separate images')

    # Memulai program
    main(parser.parse_args())

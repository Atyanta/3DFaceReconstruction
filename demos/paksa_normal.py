import os
import torch
import cv2
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
import numpy as np
from time import time
import argparse


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # Load test image (hanya menggunakan input gambar tanpa ekspresi lain)
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    
    # DECA
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    visdict_list_list = []
    for i in range(len(testdata)):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]

        # Encode gambar input tanpa ekspresi lain
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)  # Tensor output

        # Set pose and ekspresi normal (zero values untuk pose dan ekspresi)
        euler_pose = torch.zeros((1, 3)).to(device)  # Normal pose (no head movement)
        global_pose = batch_euler2axis(deg2rad(euler_pose[:, :3].cuda()))
        codedict['pose'][:,:3] = global_pose  # Reset pose to normal (neutral)
        
        # Ekspresi normal (tanpa ekspresi)
        codedict['exp'] = torch.zeros_like(codedict['exp'])  # Ekspresi normal (neutral)

        # Dekode kembali untuk mendapatkan visualisasi dengan pose normal dan ekspresi normal
        _, visdict_normal = deca.decode(codedict)

        visdict = {x: visdict_normal[x] for x in ['inputs', 'shape_detail_images']}
        visdict_list_list.append(visdict)

        # Save hasil dengan pose dan ekspresi normal
        cv2.imwrite(os.path.join(savefolder, f"{name}_normal.jpg"), deca.visualize(visdict_normal))

    print(f'-- Please check the results in {savefolder}')


def batch_euler2axis(euler):
    """ Convert euler angles (yaw, pitch, roll) to rotation matrix (axis-angle) """
    # Implementasi konversi euler ke axis (rotasi matrix)
    pass


def deg2rad(euler):
    """ Convert degrees to radians """
    return euler * np.pi / 180


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results (obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details')
    main(parser.parse_args())

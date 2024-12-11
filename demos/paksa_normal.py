import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
import imageio
from skimage.transform import rescale
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # Load test images (tidak perlu folder ekspresi)
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    
    # Inisialisasi DECA
    deca_cfg.model.use_tex = args.useTex  # Menentukan apakah menggunakan FLAME texture
    deca_cfg.model.extract_tex = args.extractTex  # Menentukan apakah akan mengekstrak tekstur dari gambar
    deca_cfg.rasterizer_type = args.rasterizer_type  # Jenis rasterizer
    deca = DECA(config=deca_cfg, device=device)

    visdict_list_list = []

    for i in range(len(testdata)):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)  # tensor
            
        visdict_list = []

        # Set pose normal (yaw, pitch, roll = 0)
        euler_pose = torch.zeros((1, 3))
        euler_pose[:, 0] = 0  # Pitch
        euler_pose[:, 1] = 0  # Yaw
        euler_pose[:, 2] = 0  # Roll
        global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda())) 
        codedict['pose'][:,:3] = global_pose
        
        # Atur kamera untuk pemrosesan
        codedict['cam'][:,:] = 0.
        codedict['cam'][:,0] = 8
        
        # Decode dan simpan hasil
        _, visdict_view = deca.decode(codedict) 
        visdict = {x: visdict[x] for x in ['inputs', 'shape_detail_images']}
        visdict['pose'] = visdict_view['shape_detail_images']
        visdict_list.append(visdict)
        visdict_list_list.append(visdict_list)

        # Simpan objek .obj
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)

    print(f'-- Please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    # Argument input dan output
    parser.add_argument('-i', '--inputpath', default='TestSamples/teaser', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/teaser/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')

    # Argument extractTex untuk mengekstrak tekstur atau albedo map
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to extract texture from input image as the uv texture map, '
                         'set false if you want albedo map from FLAME model')

    # Argument useTex untuk menggunakan model FLAME untuk tekstur
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, '
                             'set it to True only if you downloaded texture model')    

    # Argument rasterizer_type untuk jenis rasterizer
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')

    # Argument iscrop untuk memilih apakah gambar akan dipotong (crop)
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')

    # Argument detector untuk memilih detektor wajah
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details')

    # Argument saveObj untuk menyimpan file .obj hasil rekonstruksi
    parser.add_argument('--saveObj', action='store_true', 
                        help='whether to save the output as .obj files')

    main(parser.parse_args())

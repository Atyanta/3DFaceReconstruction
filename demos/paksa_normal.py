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

    # Load test images
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)
    
    # Initialize DECA model
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    visdict_list_list = []
    
    # Iterate through each image in the test dataset
    for i in range(len(testdata)):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        
        # Run DECA encode and decode without gradients (no optimization)
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)  # Get tensor output and visual dictionary

        # Create list for storing visualizations for this particular image
        visdict_list = []
        
        # Set pose wajah normal (yaw=0, pitch=0, roll=0)
        euler_pose = torch.zeros((1, 3))  # Euler pose with yaw, pitch, and roll all set to 0
        global_pose = batch_euler2axis(deg2rad(euler_pose[:, :3].cuda()))  # Convert to rotation matrix
        codedict['pose'][:,:3] = global_pose  # Apply the normal face pose
        
        # Set camera parameters to default
        codedict['cam'][:,:] = 0.
        codedict['cam'][:,0] = 8
        
        # Decode again to get the shape with normal pose (no yaw or pitch)
        _, visdict_view = deca.decode(codedict)   
        
        # Extract relevant parts of the visualization (e.g., inputs, shape detail images)
        visdict = {x: visdict[x] for x in ['inputs', 'shape_detail_images']}
        visdict['pose'] = visdict_view['shape_detail_images']
        
        # Append this result to the visdict list for the current image
        visdict_list.append(visdict)
        
        # Save the 3D object in .obj format
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        
        # Append the visual dict list for this image to the main list
        visdict_list_list.append(visdict_list)
    
    print(f'-- Please check the teaser figure in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/teaser', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/teaser/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details')
    parser.add_argument('--saveObj', action='store_true', 
                        help='whether to save the output as .obj files')
    
    main(parser.parse_args())

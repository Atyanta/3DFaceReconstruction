import os, sys
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import savemat

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

# Convert degrees to radians
def deg2rad(degrees):
    return degrees * (np.pi / 180.0)

# Fungsi untuk konversi Euler ke axis (untuk rotasi)
def batch_euler2axis(euler_angles):
    batch_size = euler_angles.shape[0]
    cos_a = torch.cos(euler_angles)
    sin_a = torch.sin(euler_angles)
    rot_matrix = torch.zeros(batch_size, 3, 3).to(euler_angles.device)

    rot_matrix[:, 0, 0] = cos_a[:, 1] * cos_a[:, 2]
    rot_matrix[:, 0, 1] = -cos_a[:, 1] * sin_a[:, 2]
    rot_matrix[:, 0, 2] = sin_a[:, 1]

    rot_matrix[:, 1, 0] = sin_a[:, 0] * sin_a[:, 1] * cos_a[:, 2] + cos_a[:, 0] * sin_a[:, 2]
    rot_matrix[:, 1, 1] = -sin_a[:, 0] * sin_a[:, 1] * sin_a[:, 2] + cos_a[:, 0] * cos_a[:, 2]
    rot_matrix[:, 1, 2] = -cos_a[:, 1] * sin_a[:, 0]

    rot_matrix[:, 2, 0] = -cos_a[:, 0] * sin_a[:, 1] * cos_a[:, 2] + sin_a[:, 0] * sin_a[:, 2]
    rot_matrix[:, 2, 1] = cos_a[:, 0] * sin_a[:, 1] * sin_a[:, 2] + sin_a[:, 0] * cos_a[:, 2]
    rot_matrix[:, 2, 2] = cos_a[:, 0] * cos_a[:, 1]

    return rot_matrix

def main(args):
    # Buat folder output jika belum ada
    os.makedirs(args.savefolder, exist_ok=True)

    # Load test images
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # Inisialisasi DECA
    deca_cfg.model.use_tex = args.useTex  # Menyesuaikan penggunaan tekstur FLAME
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex  # Menyertakan ekstraksi tekstur
    deca = DECA(config=deca_cfg, device=args.device)

    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(args.device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)  # tensor

            # Pose Normal (Netral)
            euler_pose = torch.zeros((1, 3))  # Menetapkan pose netral
            global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda())) 

            # Fix: Assign just the pose vector, not the full rotation matrix
            codedict['pose'][:,:3] = euler_pose[:,:3]  # Assign Euler angles directly to pose (1x3)

            codedict['cam'][:] = 0.
            codedict['cam'][:,0] = 8
            _, visdict_view = deca.decode(codedict)   
            visdict = {x: visdict[x] for x in ['inputs', 'shape_detail_images']}         
            visdict['pose'] = visdict_view['shape_detail_images']

            # Ekspresi Wajah Normal (tidak ekstrem)
            euler_pose = torch.zeros((1, 3))  # Ekspresi netral
            jaw_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda())) 
            codedict['pose'][:,3:] = jaw_pose
            _, visdict_view_exp = deca.decode(codedict)     
            visdict['exp'] = visdict_view_exp['shape_detail_images']

            # Visualisasi hasil
            os.makedirs(os.path.join(args.savefolder, name), exist_ok=True)
            cv2.imwrite(os.path.join(args.savefolder, name + '_pose_exp_vis.jpg'), deca.visualize(visdict))

            if args.saveImages:  # Save all visualization images
                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                    if vis_name not in visdict.keys():
                        continue
                    image = util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(os.path.join(args.savefolder, name, name + '_' + vis_name + '.jpg'), image)

    print(f'-- please check the results in {args.savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    # Input and output paths
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='Path to the test data, can be image folder, image path, image list, or video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='Path to the output directory, where results will be stored.')
    
    # Device configuration
    parser.add_argument('--device', default='cuda', type=str, help='Set device, use cpu for CPU.')
    
    # Image processing parameters
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to crop input image. Set false only when the test images are well-cropped.')
    parser.add_argument('--sample_step', default=10, type=int, help='Sample images from video data for every step.')
    parser.add_argument('--detector', default='fan', type=str,
                        help='Detector for cropping face, check decalib/detectors.py for details.')
    
    # Rendering options
    parser.add_argument('--rasterizer_type', default='standard', type=str, help='Rasterizer type: pytorch3d or standard')
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to render results in original image size (only for rasterizer_type=standard).')
    
    # Texture and output save options
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to use FLAME texture model to generate UV texture map.')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to extract texture from input image as the UV texture map.')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save visualization of output.')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save 2D and 3D keypoints.')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save depth image.')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save outputs as .obj.')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save outputs as .mat.')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to save visualization output as separate images.')
    
    # Parse the arguments and run the main function
    args = parser.parse_args()
    main(args)

# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

def main(args):
    # Konfigurasi direktori hasil
    savefolder = args.savefolder
    os.makedirs(savefolder, exist_ok=True)
    device = args.device

    # Memuat data gambar untuk pengujian
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # Konfigurasi DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)

    # Proses rekonstruksi untuk setiap gambar
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]

        with torch.no_grad():
            # Rekonstruksi wajah dengan DECA
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)

            # Memaksa ekspresi dan pose normal
            codedict['pose'][:, :3] = 0  # Atur pose global (rotasi kepala) ke nol
            codedict['pose'][:, 3:] = 0  # Atur jaw pose (bukaan mulut) ke nol
            codedict['exp'] = 0          # Atur ekspresi ke nol (neutral expression)

            # Dekode ulang dengan pose dan ekspresi normal
            opdict, visdict = deca.decode(codedict)

            # Menampilkan koordinat landmark 2D dan 3D
            landmarks2d = opdict['landmarks2d'][0].cpu().numpy()
            landmarks3d = opdict['landmarks3d'][0].cpu().numpy()

            if args.saveDepth or args.saveObj or args.saveImages:
                os.makedirs(os.path.join(savefolder, name), exist_ok=True)

            # Menyimpan hasil Depth Image
            if args.saveDepth:
                depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
                visdict['depth_images'] = depth_image
                cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))

            # Menyimpan file Detail OBJ
            if args.saveObj:
                detail_obj_path = os.path.join(savefolder, name, name + '_detail.obj')
                deca.save_obj(detail_obj_path, opdict)

            # Menyimpan file Mat
            if args.saveMat:
                opdict = util.dict_tensor2npy(opdict)
                savemat(os.path.join(savefolder, name, name + '.mat'), opdict)

            # Menyimpan Visualisasi 3D dan Depth
            if args.saveImages:
                # Menyimpan gambar-gambar terpisah untuk visualisasi
                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                    if vis_name in visdict:
                        image = util.tensor2image(visdict[vis_name][0])
                        cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'), image)

                # Gabungkan gambar Depth dan 3D menjadi satu gambar _vis.jpg
                depth_image = util.tensor2image(visdict['depth_images'][0])
                rendered_image = util.tensor2image(visdict['rendered_images'][0])

                # Menggabungkan gambar depth dan 3D untuk visualisasi
                combined_image = np.hstack((depth_image, rendered_image))
                cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), combined_image)

    print(f'-- Hasil disimpan di {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paksa Normal: Rekonstruksi Wajah Normal dengan Ekspresi')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str, help='Path ke data gambar input')
    parser.add_argument('-s', '--savefolder', default='Result_Normal', type=str, help='Path ke folder hasil rekonstruksi')
    parser.add_argument('--device', default='cuda', type=str, help='Perangkat yang digunakan (cpu/cuda)')
    
    # Pengaturan gambar
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'], help='Apakah gambar input sudah dipotong dengan baik?')
    parser.add_argument('--sample_step', default=10, type=int, help='Ambil gambar dari video setiap langkah ini')
    parser.add_argument('--detector', default='fan', type=str, help='Detektor untuk pemotongan wajah')

    # Pengaturan rendering
    parser.add_argument('--rasterizer_type', default='standard', type=str, help='Tipe rasterizer: pytorch3d atau standard')
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'], help='Render gambar dalam ukuran asli')
    
    # Pengaturan hasil simpan
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'], help='Gunakan model tekstur FLAME untuk membuat peta tekstur')
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'], help='Ekstraksi tekstur dari gambar input')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'], help='Simpan gambar depth')
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'], help='Simpan output sebagai file .obj')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'], help='Simpan file .mat')
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'], help='Simpan gambar visualisasi')

    # Jalankan program
    main(parser.parse_args())

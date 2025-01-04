import os, sys
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

# Konfigurasi awal file dan perangkat
csv_path = '/kaggle/input/dataset-normjbu/voice_features_sample.csv'
output_path = 'voice_features_sample.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inisialisasi model DECA
deca_cfg.model.use_tex = False  # Ubah jika Anda menggunakan tekstur
deca_model = DECA(config=deca_cfg, device=device)

def process_image(image_path):
    """Memproses gambar menggunakan encoder DECA."""
    try:
        testdata = datasets.TestData(image_path, iscrop=True, crop_size=224)
        images = testdata[0]['image'].to(device).unsqueeze(0)

        # Encode image with DECA
        codedict = deca_model.encode(images, use_detail=True)

        # Ambil nilai yang diminta
        detail = codedict['detail'].detach().cpu().numpy().tolist()
        shape = codedict['shape'].detach().cpu().numpy().tolist()
        tex = codedict['tex'].detach().cpu().numpy().tolist()
        exp = codedict['exp'].detach().cpu().numpy().tolist()

        return detail, shape, tex, exp
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None, None

def main():
    # Baca dataset asli
    df = pd.read_csv(csv_path)

    # Pastikan dataset memiliki kolom yang diperlukan
    assert len(df.columns) >= 3, "Dataset harus memiliki minimal 3 kolom (ID, nama, dan path gambar)."

    # Tambahkan kolom baru jika belum ada
    if 'codedict_detail' not in df.columns:
        df['codedict_detail'] = ''
        df['codedict_shape'] = ''
        df['codedict_tex'] = ''
        df['codedict_exp'] = ''

    # Cache untuk menghindari pemrosesan ulang gambar yang sama
    processed_cache = {}

    for index, row in df.iterrows():
        image_id = row[0]  # Kolom pertama sebagai ID
        image_path = row[2]  # Kolom ketiga adalah path gambar

        # Periksa apakah ID sudah diproses sebelumnya
        if image_id in processed_cache:
            detail, shape, tex, exp = processed_cache[image_id]
        else:
            # Proses gambar
            detail, shape, tex, exp = process_image(image_path)
            if detail is not None:
                processed_cache[image_id] = (detail, shape, tex, exp)

        # Simpan hasilnya ke dalam dataframe
        df.at[index, 'codedict_detail'] = str(detail)
        df.at[index, 'codedict_shape'] = str(shape)
        df.at[index, 'codedict_tex'] = str(tex)
        df.at[index, 'codedict_exp'] = str(exp)

    # Simpan hasil ke file CSV baru
    df.to_csv(output_path, index=False)
    print(f"Hasil disimpan di {output_path}")

if __name__ == "__main__":
    main()

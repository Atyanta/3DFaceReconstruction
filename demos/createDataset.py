import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg

# Fungsi untuk memuat dan memproses gambar
def load_image(image_path, device):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Sesuaikan ukuran input DECA
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')  # Pastikan gambar dalam mode RGB
    return transform(image).to(device)

def main(args):
    # Tentukan device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    deca_cfg.rasterizer_type = args.rasterizer_type  # Tetapkan jenis rasterizer
    deca = DECA(config=deca_cfg, device=device)

    # Validasi keberadaan file CSV input
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV file tidak ditemukan: {args.csv_path}")

    # Baca CSV input
    data = pd.read_csv(args.csv_path)

    # Pastikan kolom yang diperlukan ada
    required_columns = ['voice_id', 'features', 'image_path']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Kolom wajib '{col}' tidak ada dalam CSV input.")

    # Tambahkan kolom baru untuk hasil DECA
    new_columns = ['detail', 'shape', 'tex', 'exp']
    for col in new_columns:
        if col not in data.columns:
            data[col] = None

    # Cache untuk menghindari pemrosesan ulang
    cache = {}

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        voice_id = row['voice_id']
        image_path = row['image_path']

        # Abaikan baris dengan path gambar yang tidak valid
        if pd.isna(image_path) or not os.path.exists(image_path):
            print(f"Path gambar tidak valid pada index {index}: {image_path}")
            continue

        # Gunakan hasil cache jika sudah ada
        if voice_id in cache:
            result = cache[voice_id]
        else:
            try:
                # Load dan proses gambar dengan DECA
                image = load_image(image_path, device)[None, ...]  # Tambahkan dimensi batch
                with torch.no_grad():
                    codedict = deca.encode(image)
                    result = {
                        'detail': codedict['detail'].cpu().numpy().tolist(),
                        'shape': codedict['shape'].cpu().numpy().tolist(),
                        'tex': codedict['tex'].cpu().numpy().tolist(),
                        'exp': codedict['exp'].cpu().numpy().tolist(),
                    }
                    # Simpan hasil ke cache
                    cache[voice_id] = result
            except Exception as e:
                print(f"Error memproses gambar pada index {index}: {e}")
                continue

        # Simpan hasil ke DataFrame
        data.at[index, 'detail'] = str(result['detail'])
        data.at[index, 'shape'] = str(result['shape'])
        data.at[index, 'tex'] = str(result['tex'])
        data.at[index, 'exp'] = str(result['exp'])

    # Simpan DataFrame yang diperbarui ke CSV output
    data.to_csv(args.output_csv_path, index=False)
    print(f"CSV yang diperbarui disimpan ke: {args.output_csv_path}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Proses gambar dan tambahkan hasil DECA ke file CSV.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path ke file CSV input.")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path ke file CSV output.")
    parser.add_argument("--device", type=str, default="cuda", help="Device untuk komputasi ('cuda' atau 'cpu').")
    parser.add_argument("--rasterizer_type", type=str, default="standard", help="Jenis rasterizer ('standard' atau 'pytorch3d').")

    args = parser.parse_args()
    main(args)

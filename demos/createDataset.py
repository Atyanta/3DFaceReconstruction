import os
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg


def main(args):
    # Load CSV
    csv_path = args.csv_path
    output_csv_path = args.output_csv_path
    device = args.device

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Read the input CSV
    data = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_columns = ['voice_id', 'features', 'image_path']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add new columns for DECA outputs
    new_columns = ['detail', 'shape', 'tex', 'exp']
    for col in new_columns:
        if col not in data.columns:
            data[col] = None

    # Initialize DECA model
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device=device)

    # Cache to avoid redundant processing
    cache = {}

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        voice_id = row['voice_id']
        image_path = row['image_path']

        if pd.isna(image_path) or not os.path.exists(image_path):
            print(f"Skipping invalid image path at index {index}: {image_path}")
            continue

        # Check if this voice_id is already processed
        if voice_id in cache:
            result = cache[voice_id]
        else:
            # Load image and process with DECA
            try:
                image = util.load_image(image_path).to(device)[None, ...]
                with torch.no_grad():
                    codedict = deca.encode(image)
                    result = {
                        'detail': codedict['detail'].cpu().numpy().tolist(),
                        'shape': codedict['shape'].cpu().numpy().tolist(),
                        'tex': codedict['tex'].cpu().numpy().tolist(),
                        'exp': codedict['exp'].cpu().numpy().tolist(),
                    }
                    # Cache the result for this voice_id
                    cache[voice_id] = result
            except Exception as e:
                print(f"Error processing image at index {index}: {e}")
                continue

        # Save results to the DataFrame
        data.at[index, 'detail'] = str(result['detail'])
        data.at[index, 'shape'] = str(result['shape'])
        data.at[index, 'tex'] = str(result['tex'])
        data.at[index, 'exp'] = str(result['exp'])

    # Save the updated DataFrame to a new CSV
    data.to_csv(output_csv_path, index=False)
    print(f"Updated CSV saved to {output_csv_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process images from a CSV and extract DECA features')
    parser.add_argument('--csv_path', required=True, type=str, help='Path to the input CSV file')
    parser.add_argument('--output_csv_path', required=True, type=str, help='Path to save the updated CSV file')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for processing (cpu or cuda)')
    
    args = parser.parse_args()
    main(args)

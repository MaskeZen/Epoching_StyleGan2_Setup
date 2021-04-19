
from pathlib import Path

class Config:
    model_path = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    fps = 20
    results_size = 400
    ## PATHS
    orig_img_path = Path('imgs')
    aligned_imgs_path = Path('aligned_imgs')
    output_imgs = Path('output_imgs')
    output_gifs_path = Path('output_gifs')
    
    # Make folders if doesn't exist.
    if not output_gifs_path.exists():
        output_gifs_path.mkdir()
    if not aligned_imgs_path.exists():
        aligned_imgs_path.mkdir()
    if not output_imgs.exists():
        output_imgs.mkdir()

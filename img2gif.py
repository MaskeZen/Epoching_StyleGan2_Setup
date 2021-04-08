from PIL import Image
from pathlib import Path

tot_aligned_imgs = 6
fps = 20
results_size = 400

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def make_project_progress_gifs():
    
    all_result_folders = list(Path('results/').iterdir())
    all_result_folders.sort()

    last_result_folder = all_result_folders[-1]
    
    for img_num in range(tot_aligned_imgs):
        all_step_pngs = [x for x in last_result_folder.iterdir() if x.name.endswith('png') and 'image{0:04d}'.format(img_num) in x.name]
        all_step_pngs.sort()

        target_image = Image.open(all_step_pngs[-1]).resize((results_size, results_size))

        all_concat_imgs = []
        for step_img_path in all_step_pngs[:-1]:
            step_img = Image.open(step_img_path).resize((results_size, results_size))
            all_concat_imgs.append(get_concat_h(target_image, step_img))

        all_concat_imgs[0].save('output_gifs/image{0:04d}_project_progress.gif'.format(img_num), save_all=True, append_images=all_concat_imgs[1:], duration=1000/fps, loop=0)

make_project_progress_gifs()
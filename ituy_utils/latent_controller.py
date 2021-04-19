import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from options.config import Config

class LatentController:

    def get_control_latent_vectors(path):
        files = [x for x in Path(path).iterdir() if str(x).endswith('.npy')]
        latent_vectors = {f.name[:-4]:np.load(f) for f in files}
        return latent_vectors

    def get_final_latents():
        all_results = list(Path('results/').iterdir())
        all_results.sort()
        
        last_result = all_results[-1]

        latent_files = [x for x in last_result.iterdir() if 'final_latent_code' in x.name]
        latent_files.sort()
        
        all_final_latents = []
        
        for file in latent_files:
            print(' --> final_latent: ' + file.name)
            with open(file, mode='rb') as latent_pickle:
                all_final_latents.append(pickle.load(latent_pickle))
        
        return all_final_latents

    def make_latent_control_animation(model_loader, latent_code, feature, start_amount, end_amount, step_size, person):
    
        all_imgs = []
        
        amounts = np.linspace(start_amount, end_amount, abs(end_amount-start_amount)/step_size)

        latent_controls = LatentController.get_control_latent_vectors('stylegan2directions/')
        
        for amount_to_move in tqdm(amounts):
            modified_latent_code = np.array(latent_code)
            modified_latent_code += latent_controls[feature]*amount_to_move
            images = model_loader.generate_image_from_projected_latents(modified_latent_code)
            latent_img = Image.fromarray(images[0]).resize((Config.results_size, Config.results_size))
            all_imgs.append(latent_img)

        save_name = Config.output_gifs_path/'{0}_{1}.gif'.format(person, feature)
        all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/Config.fps, loop=0)

    def linear_interpolate(code1, code2, alpha):
        return code1 * alpha + code2 * (1 - alpha)

    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def make_latent_interp_animation(self, modelLoader, code1, code2, img1, img2, num_interps):
        step_size = 1.0/num_interps

        all_imgs = []

        amounts = np.arange(0, 1, step_size)

        for alpha in tqdm(amounts):
            interpolated_latent_code = self.linear_interpolate(code1, code2, alpha)
            images = modelLoader.generate_image_from_z(interpolated_latent_code)
            interp_latent_image = Image.fromarray(images[0]).resize((400, 400))
            frame = self.get_concat_h(img1, interp_latent_image)
            frame = self.get_concat_h(frame, img2)
            all_imgs.append(frame)

        save_name = Config.output_gifs_path/'latent_space_traversal.gif'
        all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/Config.fps, loop=0)
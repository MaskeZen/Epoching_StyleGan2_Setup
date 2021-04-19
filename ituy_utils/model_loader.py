from ituy_utils.latent_controller import LatentController
import sys
import numpy as np

sys.path.append('stylegan2/')
from stylegan2 import pretrained_networks
from stylegan2 import dnnlib
from stylegan2.dnnlib import tflib
from options.config import Config
from align_face import align_face

class ModelLoader:

    # model_path = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    _G = None
    _D = None
    Gs = None
    noise_vars = None
    Gs_kwargs = None

    def __init__(self) -> None:
        self._G, self._D, self.Gs = pretrained_networks.load_networks(Config.model_path)
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
        self.Gs_kwargs = dnnlib.EasyDict()
        self.Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_kwargs.randomize_noise = False

    def generate_image_from_projected_latents(self, latent_vector):
        images = self.Gs.components.synthesis.run(latent_vector, **self.Gs_kwargs)
        return images
    
    # Generate images given a random seed (Integer)
    def generate_image_random(self, rand_seed):
        rnd = np.random.RandomState(rand_seed)
        z = rnd.randn(1, *self.Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars})
        images = self.Gs.run(z, None, **self.Gs_kwargs)
        return images, z

    # Generate images given a latent code ( vector of size [1, 512] )
    def generate_image_from_z(self, z):
        images = self.Gs.run(z, None, **self.Gs_kwargs)
        return images

    def align_images():
        all_imgs = list(Config.orig_img_path.iterdir())
        for img in all_imgs:
            align_face(str(img)).save(Config.aligned_imgs_path/('aligned_'+img.name))
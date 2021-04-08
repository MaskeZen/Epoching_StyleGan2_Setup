import sys

sys.path.append('stylegan2/')
from stylegan2 import pretrained_networks
from stylegan2 import dnnlib
from stylegan2.dnnlib import tflib

from align_face import align_face
from pathlib import Path
from PIL import Image
from options.cmd_options import CmdOptions

import pickle
import numpy as np

## import ipywidgets as widgets Barra de progreso
from tqdm import tqdm

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

opts = CmdOptions().parse()

run_align_images = opts.align
run_latent_direction = opts.direction
run_feature = opts.feature
run_latent_index = opts.latent_index
get_model = False

# load model?
if run_latent_direction:
    get_model = True

# Code to load the StyleGAN2 Model
def load_model():
    _G, _D, Gs = pretrained_networks.load_networks(model_path)

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    return Gs, noise_vars, Gs_kwargs

# Generate images given a random seed (Integer)
def generate_image_random(rand_seed):
    rnd = np.random.RandomState(rand_seed)
    z = rnd.randn(1, *Gs.input_shape[1:])
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
    images = Gs.run(z, None, **Gs_kwargs)
    return images, z

# Generate images given a latent code ( vector of size [1, 512] )
def generate_image_from_z(z):
    images = Gs.run(z, None, **Gs_kwargs)
    return images

def linear_interpolate(code1, code2, alpha):
    return code1 * alpha + code2 * (1 - alpha)

#collapse-hide
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def make_latent_interp_animation(code1, code2, img1, img2, num_interps):

    step_size = 1.0/num_interps

    all_imgs = []

    amounts = np.arange(0, 1, step_size)

    for alpha in tqdm(amounts):
        interpolated_latent_code = linear_interpolate(code1, code2, alpha)
        images = generate_image_from_z(interpolated_latent_code)
        interp_latent_image = Image.fromarray(images[0]).resize((400, 400))
        frame = get_concat_h(img1, interp_latent_image)
        frame = get_concat_h(frame, img2)
        all_imgs.append(frame)

    save_name = output_gifs_path/'latent_space_traversal.gif'
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)

def get_random_images():
    # Ask the generator to make an output, given a random seed number: 42
    images, latent_code1 = generate_image_random(42)
    image1 = Image.fromarray(images[0]).resize((results_size, results_size))
    #print('latent_code1.shape'.join(latent_code1.shape))
    image1.save('output_imgs/image1.jpg')

    images, latent_code2 = generate_image_random(1234)
    image2 = Image.fromarray(images[0]).resize((results_size, results_size))
    #print('latent_code2.shape'.join(latent_code2.shape))
    image2.save('output_imgs/image2.jpg')


    interpolated_latent_code = linear_interpolate(latent_code1, latent_code2, 0.5)
    #interpolated_latent_code.shape

    images = generate_image_from_z(interpolated_latent_code)
    Image.fromarray(images[0]).resize((results_size, results_size))

    make_latent_interp_animation(latent_code1, latent_code2, image1, image2, num_interps=200)

def align_images():
    all_imgs = list(orig_img_path.iterdir())
    for img in all_imgs:
        align_face(str(img)).save(aligned_imgs_path/('aligned_'+img.name))

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

def generate_image_from_projected_latents(latent_vector):
    images = Gs.components.synthesis.run(latent_vector, **Gs_kwargs)
    return images

def make_latent_control_animation(latent_code, feature, start_amount, end_amount, step_size, person):
    
    all_imgs = []
    
    amounts = np.linspace(start_amount, end_amount, abs(end_amount-start_amount)/step_size)
    
    for amount_to_move in tqdm(amounts):
        modified_latent_code = np.array(latent_code)
        modified_latent_code += latent_controls[feature]*amount_to_move
        images = generate_image_from_projected_latents(modified_latent_code)
        latent_img = Image.fromarray(images[0]).resize((results_size, results_size))
        # img_reference = Image.fromarray(
        #         (generate_image_from_projected_latents(latent_code)[0])
        #     ).resize((results_size, results_size))
        # all_imgs.append(get_concat_h(img_reference, latent_img))
        all_imgs.append(latent_img)

    save_name = output_gifs_path/'{0}_{1}.gif'.format(person, feature)
    all_imgs[0].save(save_name, save_all=True, append_images=all_imgs[1:], duration=1000/fps, loop=0)

if get_model:
    # Loading the StyleGAN2 Model!
    Gs, noise_vars, Gs_kwargs = load_model()

if run_align_images:
    print(' =========== run_align_images =========== ')
    align_images()

if run_latent_direction:
    print(' =========== run_latent_direction =========== ')
    print(' =========== feature '+run_feature+' =========== ')
    latent_controls = get_control_latent_vectors('stylegan2directions/')
    latent_codes = get_final_latents()
    latent_index = run_latent_index or 0

    img_reference = Image.fromarray(
            (generate_image_from_projected_latents(latent_codes[latent_index])[0])
        ).resize((results_size, results_size))
    file_name = 'img_reference_person-'+str(latent_index)+'.jpg'
    img_reference.save(str(output_gifs_path) + '/img_reference_person-'+str(latent_index)+'.jpg')
    # features: 'age', 'eye_distance', 'eye_eyebrow_distance', 'eye_ratio', 'eyes_open', 'gender', 'lip_ratio', 
    # 'mouth_open', 'mouth_ratio', 'nose_mouth_distance', 'nose_ratio', 'nose_tip', 'pitch', 'roll', 'smile', 'yaw'
    make_latent_control_animation(
        latent_code=latent_codes[latent_index],
        feature=run_feature, 
        start_amount=-10, 
        end_amount=10, 
        step_size=0.5, 
        person='person-'+str(latent_index)
        )

    # len(latent_controls), latent_controls.keys(), latent_controls['age'].shape

# Create stylegan2 Dataset
#python -W ignore stylegan2/dataset_tool.py create_from_images datasets_stylegan2/custom_imgs aligned_imgs/

## ALIGN
# One-Time Download of Facial Landmark Detection Model Weights
# if not Path('shape_predictor_68_face_landmarks.dat').exists():
#     exec('curl --remote-name http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
#     exec('bzip2 -dv shape_predictor_68_face_landmarks.dat.bz2')


# exec("python -W ignore stylegan2/epoching_custom_run_projector.py project-real-images --network=$model_path \
#   --dataset=custom_imgs --data-dir=datasets_stylegan2 --num-images=9 --num-snapshots 500")
# python -W ignore stylegan2/epoching_custom_run_projector.py project-real-images --network='gdrive:networks/stylegan2-ffhq-config-f.pkl' --dataset=custom_imgs --data-dir=datasets_stylegan2 --num-images=9 --num-snapshots 500





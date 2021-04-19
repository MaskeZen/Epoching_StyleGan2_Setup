
from PIL import Image
from options.cmd_options import CmdOptions
from ituy_utils.latent_controller import LatentController
from ituy_utils.model_loader import ModelLoader

from options.config import Config

opts = CmdOptions().parse()

run_align_images = opts.align
run_latent_direction = opts.direction
run_feature = opts.feature
run_latent_index = opts.latent_index
get_model = False

modelLoader = ModelLoader()

if run_align_images:
    print(' =========== run_align_images =========== ')
    modelLoader.align_images()

if run_latent_direction:
    print(' =========== run_latent_direction =========== ')
    print(' =========== feature '+run_feature+' =========== ')
    # latent_controls = get_control_latent_vectors('stylegan2directions/')
    latent_codes = LatentController.get_final_latents()
    latent_index = run_latent_index or 0

    img_reference = Image.fromarray(
            (modelLoader.generate_image_from_projected_latents(latent_codes[latent_index])[0])
        ).resize((Config.results_size, Config.results_size))
    file_name = 'img_reference_person-'+str(latent_index)+'.jpg'
    img_reference.save(str(Config.output_gifs_path) + '/img_reference_person-'+str(latent_index)+'.jpg')

    print(' ======= latent_codes[latent_index] ======= ')
    print(latent_codes[latent_index])
    print(' ======= type(latent_codes[latent_index]) ======= ')
    print(type(latent_codes[latent_index]))
    print(latent_codes[latent_index].shape)

    # features: 'age', 'eye_distance', 'eye_eyebrow_distance', 'eye_ratio', 'eyes_open', 'gender', 'lip_ratio', 
    # 'mouth_open', 'mouth_ratio', 'nose_mouth_distance', 'nose_ratio', 'nose_tip', 'pitch', 'roll', 'smile', 'yaw'
    LatentController.make_latent_control_animation(
        model_loader=modelLoader,
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
# python -W ignore stylegan2/epoching_custom_run_projector.py project-real-images --network='gdrive:networks/stylegan2-ffhq-config-f.pkl' --dataset=custom_imgs --data-dir=datasets_stylegan2 --num-images=1 --num-snapshots 10





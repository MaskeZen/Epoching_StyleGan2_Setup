
import os
from PIL import Image
from options.cmd_options import CmdOptions
from ituy_utils.latent_controller import LatentController
from ituy_utils.model_loader import ModelLoader

from options.config import Config

opts = CmdOptions().parse()

# SELECT TASK
opt_task = opts.task
run_make_gif = opt_task == 'gif'
run_latent_direction = opt_task == 'direction'
#------------------
run_align_images = opts.align
run_feature = opts.feature
run_latent_index = opts.latent_index
get_model = False

opt_amount = opts.amount or 1
opt_latent_file = opts.latent_file

modelLoader = ModelLoader()

if run_align_images:
    print(' =========== run_align_images =========== ')
    modelLoader.align_images()

if run_make_gif:
    run_feature = run_feature or 'age'
    print(' =========== run_make_gif feature '+run_feature+' =========== ')
    # latent_controls = get_control_latent_vectors('stylegan2directions/')
    latent_codes = LatentController.get_final_latents()
    latent_index = run_latent_index or 0

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

if run_latent_direction:
    run_feature = run_feature or 'age'
    print(' =========== run_latent_direction, feature: '+run_feature+' =========== ')

    ## latent_file
    latent_code = LatentController.get_latent(opt_latent_file)
    f_name, f_ext = os.path.splitext(os.path.basename(opt_latent_file))
    output_filename = 'output_' + run_feature + '_' + str(opt_amount) + '_' + f_name
    print(' ======= latent_code.shape ======= ')
    print(latent_code.shape)

    # features: 'age', 'eye_distance', 'eye_eyebrow_distance', 'eye_ratio', 'eyes_open', 'gender', 'lip_ratio', 
    # 'mouth_open', 'mouth_ratio', 'nose_mouth_distance', 'nose_ratio', 'nose_tip', 'pitch', 'roll', 'smile', 'yaw'
    LatentController.make_latent_control_image(
        model_loader=modelLoader,
        amount=opt_amount,
        feature=run_feature,
        latent_code=latent_code,
        output_name=output_filename
    )

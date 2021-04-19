import os
import logging

from flask import Flask
from flask.globals import request
from flask.helpers import send_file
from werkzeug.utils import secure_filename
from ituy_utils.model_loader import ModelLoader

app = Flask(__name__)

UPLOAD_FOLDER = './inputs'
if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

## ------------------------------------------------------------------------------------------------

CODE_DIR = os.getcwd()
logging.basicConfig(filename="pixel2style.log")

# Initialize the aplication options
task_type = 'ffhq_frontalize'
image_align = True
input_path = ''
#----------------------------------

# Gs, noise_vars, Gs_kwargs = ModelBootstrap.load_model()
modelLoader = ModelLoader()
## ------------------------------------------------------------------------------------------------

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/latent', methods=['GET', 'POST'])
def frontalize():
    if request.method == 'GET':
        return 'mostrar ayuda'
    else:
        latent_space = request.files['latent_space']
        if latent_space != None:
            filename = secure_filename(latent_space.filename)
            latent_space.save(os.path.join(UPLOAD_FOLDER, filename))
            logging.info('latent_space registrado... ' + filename)

            # input_image = BootstrapPixel2Style.get_input_image(tmp_file_path, image_align)
            # if input_image == None:
            #     return 'Error al frontalizar la imagen'
            # res_image = BootstrapPixel2Style.process_image(input_image, net_ffhq_frontalize, EXPERIMENT_DATA_ARGS[task_type])

            # if not os.path.exists('output'):
            #     os.mkdir('output')
            #     print("se creo el directorio output.")
            
            # base_path, image_file_name = os.path.split(tmp_file_path)
            # output_file = "output/flask_"+task_type+"_"+image_file_name+".jpg"
            # res_image.save(output_file)
            # res_image.tobytes()
            # return send_file(filename_or_fp=output_file,attachment_filename=image_file_name+"_"+task_type+".jpg",as_attachment=True)
            return 'latent space registrado con éxito.'
        else:
            return 'no se envio ninguna imagen'

from flask import Flask,render_template, request, flash, request, redirect, url_for, send_from_directory
from flask_paginate import Pagination, get_page_args
from werkzeug.utils import secure_filename
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from helper_toolbox.FilePickling import pkl_load
from helper_toolbox.Inference import inference
from helper_toolbox.Ranking import ranking
import os
import numpy as np
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
app.template_folder = ''
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#load tflite model
tflite_model = pkl_load("helper_toolbox/tflite_model.pkl")
tflite_interpreter_obj = tflite.Interpreter(model_content=tflite_model)
tflite_interpreter_obj.allocate_tensors()

# PCA
pca_obj = pkl_load("helper_toolbox/pca_obj.pkl")

# MinMaxScaler
scaler_obj = pkl_load("helper_toolbox/scaler_obj.pkl")

# images features array
images_features_scaled_pca = np.load("helper_toolbox/images_features_scaled_pca.npy")

# approx search
DIM = 500
NUM_NEAREST_NEIGHBORS = 100
approx_search_obj = AnnoyIndex(DIM, 'euclidean')
approx_search_obj.load('helper_toolbox/approx_tree.ann')

#
image_vector = inference("static/images/input.jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
global image_ids
image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)
image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)

INPUT_FILE_NAME = "input.jpg"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def get_image_ids(offset=0, per_page=10):
    return image_ids[offset: offset + per_page]

# This is for cleaning website cache
@app.after_request
def add_header(response):
    # response.cache_control.no_store = Tru
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

#
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_ids
    
    # If users post an image onto the server
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # if user submit a file and that file is valid
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # construct the input path
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILE_NAME)
            #  save that file to the folder
            file.save(input_file_path)
        
        # Transform that image into vectors user 3 pre-initialized moduel
        image_vector = inference("static/images/input.jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
        
        # Get the NUM_NEAREST_NEIGHBORS neighbors of that vector to a pretrained indexing module
        # This will return a top nearest image_ids that closest to the vector
        image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)

        # Perform ranking by comparing the query vector to nearest neighbor
        image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)
        
        # This is for rendering the table in result page
        page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
        total = len(image_ids)
        pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, per_page=per_page, total=total,
                                css_framework='bootstrap4')
        return redirect(url_for('result'),)
        
    else:
        return render_template('templates/home.html')
    
# Sample page for IR
# Given an idx that represent the idx-th sample
@app.route('/sample/<idx>/')
def sample(idx):
    # If idx != 0 then perform retrieval on the idx-th sample
    if idx!='0':
        global image_ids
        # Transform that image into vectors user 3 pre-initialized moduel
        image_vector = inference("static/images/"+idx+".jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
        
        # Get the NUM_NEAREST_NEIGHBORS neighbors of that vector to a pretrained indexing module
        # This will return a top nearest image_ids that closest to the vector
        image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)

        # Perform ranking by comparing the query vector to nearest neighbor
        image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)
        
        # This is for rendering the table in result page
        page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
        total = len(image_ids)
        pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, per_page=per_page, total=total,
                                css_framework='bootstrap4')
        return redirect(url_for('result'))
    else:
        # If idx = 0
        # Render the sample template
        return render_template('templates/sample.html')

# This is for rendering the table in result page
@app.route('/result')
def result():
    page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
    total = len(image_ids)
    pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    return render_template('templates/page.html',
                        image_ids=pagination_image_ids,
                        page=page,
                        per_page=per_page,
                        pagination=pagination,
                        )

if __name__ == '__main__':
    app.run(debug=True)

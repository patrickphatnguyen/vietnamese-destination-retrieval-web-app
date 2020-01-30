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

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = Tru
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global image_ids
   
    if request.method == 'POST':
        print('hello')
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
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILE_NAME)
            file.save(input_file_path)
            # if not (scanner(input_file_path)):
            #     return render_template("failscan.html")
            # return render_template("succeedscan.html")
        image_vector = inference("static/images/input.jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
        image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)
        print("before ",image_ids[0:10])

        image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)
        print("after ",image_ids[0:10])
        page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
        total = len(image_ids)
        pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, per_page=per_page, total=total,
                                css_framework='bootstrap4')
        return redirect(url_for('result'),)
        
    else:
        return render_template('templates/home.html')
@app.route('/sample/<idx>/')
def sample(idx):
    if idx!='0':
        global image_ids
        image_vector = inference("static/images/"+idx+".jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
        image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)

        image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)
        page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
        total = len(image_ids)
        pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, per_page=per_page, total=total,
                                css_framework='bootstrap4')
        return redirect(url_for('result'))
    else:
        return render_template('templates/sample.html')

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
    
@app.route('/cropper',methods=['GET', 'POST'])
def cropper():
    if request.method == 'POST':
        print("hey")
        # check if the post request has the file part
        if 'file' not in request.files:
            print("no file")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("no select")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            print("ok")
            filename = secure_filename(file.filename)
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILE_NAME)
            file.save(input_file_path)
            # if not (scanner(input_file_path)):
            #     return render_template("failscan.html")
            # return render_template("succeedscan.html")
        image_vector = inference("static/images/input.jpg",tflite_interpreter_obj,scaler_obj,pca_obj)
        image_ids = approx_search_obj.get_nns_by_vector(image_vector,NUM_NEAREST_NEIGHBORS)
        print("before ",image_ids[0:10])

        image_ids = ranking(image_vector,images_features_scaled_pca,image_ids)
        print("after ",image_ids[0:10])
        page, per_page, offset = get_page_args(page_parameter='page',
                                            per_page_parameter='per_page')
        total = len(image_ids)
        pagination_image_ids = get_image_ids(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, per_page=per_page, total=total,
                                css_framework='bootstrap4')
        return redirect(url_for('result'),)
    return render_template("templates/form.html")
if __name__ == '__main__':
    app.run(debug=True)
import numpy as np
from PIL import Image
def inference(image_path,tflite_interpreter_obj,scaler_obj,pca_obj):

    chosen_image =  np.expand_dims(np.array(Image.open(image_path).convert("RGB").resize((224,224),Image.ANTIALIAS)),0)
    chosen_image = chosen_image.astype(np.float32) # the interpreter wants so!
    
    interpreter = tflite_interpreter_obj
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], chosen_image)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    results = interpreter.get_tensor(output_details[0]['index'])
    # scaling
    results = scaler_obj.transform(results)
    # perform pca
    results = pca_obj.transform(results)


    return results[0]
from flask import render_template, jsonify, Flask, redirect, url_for, request, make_response
import os
import io
import numpy as np
from PIL import Image
import keras.utils as image
from keras.models import model_from_json

app = Flask(__name__)

SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion',
    7: 'Squamous Cell Carcinoma',
    8: 'Seborrheic Keratosis',
    9: 'Lentigo Maligna',
    10: 'Tinea Corporis (Ringworm)',
    11: 'Psoriasis',
    12: 'Eczema',
    13: 'Lupus Rash',
    14: 'Rosacea',
    15: 'Kaposi Sarcoma',
    16: 'Impetigo',
    17: 'Cutaneous T-cell Lymphoma',
    18: 'Neurofibromatosis',
    19: 'Alopecia Areata'
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signin')
def signin():
    return render_template('signin.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

def findMedicine(pred):
    medicines = {
        0: "Fluorouracil, Imiquimod, Diclofenac gel,   SYMPTOMS : Rough scaly patches, Red or brown lesions",
        1: "Imiquimod, Fluorouracil, Vismodegib--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  Pearly or waxy bump, Bleeding sore that doesn’t heal",
        2: "Salicylic acid, Cryotherapy (liquid nitrogen),   SYMPTOMS :  Thick, crusty growth, Dark wart-like texture",
        3: " No treatment needed, excision if symptomatic,   SYMPTOMS :  Firm reddish nodule, Dimpled surface",
        4: "Surgery, Immunotherapy (Pembrolizumab), Targeted therapy (Dabrafenib)--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  New mole or unusual growth, Changes in color or border",
        5: "No treatment needed, excision if suspicious,   SYMPTOMS :  Brown or black moles, Smooth, symmetrical shape",
        6: "Laser Therapy, Sclerotherapy,   SYMPTOMS :  Red or purple marks, Enlarged blood vessels",
        7: "Fluorouracil, Imiquimod, Surgery--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  Scaly, red patches, Open sores",
        8: "Cryotherapy, Curettage,   SYMPTOMS :  Waxy, stuck-on appearance, Brown or black growths",
        9: "Surgery, Imiquimod--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  Flat brown spots, Irregular borders",
        10: "Terbinafine, Clotrimazole,   SYMPTOMS :  Circular rash, Itchy, red skin",
        11: "Topical Corticosteroids,   SYMPTOMS :  Red patches, Scaly plaques",
        12: "Moisturizers & Hydrocortisone,   SYMPTOMS :  Itchy, dry skin, Inflammation",
        13: "Immunosuppressants,   SYMPTOMS :  Butterfly rash, Photosensitivity",
        14: "Metronidazole,   SYMPTOMS :  Flushing, Visible blood vessels",
        15: "Antiretroviral Therapy--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  Purple or red skin lesions, Swelling",
        16: "Antibiotics,   SYMPTOMS :  Honey-colored crusts, Blisters",
        17: "Chemotherapy--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  Persistent rash, Swollen lymph nodes",
        18: "Surgical Excision--[[URGENT CONSULTATION REQUIRED]],   SYMPTOMS :  Soft, skin-colored nodules, Growth over time",
        19: "Corticosteroids,   SYMPTOMS :  Bald patches, Hair loss"
    }
    return medicines.get(pred, "Unknown, No data available")
       


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    json_response = {}
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            return make_response(jsonify({
                'error': 'No file part in the request',
                'code': 'FILE',
                'message': 'file is not valid'
            }), 400)

        imagePil = Image.open(io.BytesIO(file.read()))
        # Save the image to a BytesIO object
        imageBytesIO = io.BytesIO()
        imagePil.save(imageBytesIO, format='JPEG')
        imageBytesIO.seek(0)
        print("detected ")
        path = imageBytesIO
        j_file = open('model.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('model.h5')
        img = image.load_img(path, target_size=(224, 224))
        img = np.array(img)
        img = img.reshape((1, 224, 224, 3))
        img = img/255
        prediction = model.predict(img)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        accuracy = round(accuracy*100, 2)
        medicine =findMedicine(pred)

        json_response = {
            "detected": False if pred == 2 else True,
            "disease": disease,
            "accuracy": accuracy,
            "medicine" : medicine,
            "img_path": file.filename,

        }

        return make_response(jsonify(json_response), 200)

    else:
        return render_template('detect.html')


if __name__ == "__main__":
    app.run(debug=True, port=3000)

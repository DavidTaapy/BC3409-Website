from flask import Flask, render_template, request, flash, Markup, get_flashed_messages, url_for
from werkzeug.utils import secure_filename
import joblib
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

# Initialise App
app = Flask(__name__, static_folder = 'static', static_url_path = '/static')
app.config['SECRET_KEY'] = "BC3409"

# Import Models
rn_imported = load_model("ResNet.h5")
dt_imported = joblib.load("DecisionTree")
rf_imported = joblib.load("RandomForest")

# Function to get predictions
def get_pred_from_img(img_tensor):
  rf_dt_input = img_tensor.reshape(img_tensor.shape[0], img_tensor.shape[1] * img_tensor.shape[2] * img_tensor.shape[3])
  pred_rn = np.argmax(rn_imported.predict(img_tensor / 255.0))
  pred_rf = rf_imported.predict(rf_dt_input)
  pred_dt = dt_imported.predict(rf_dt_input)

  List = [pred_rn, pred_rf[0], pred_dt[0]]
  if len(set(List)) < 3:
    return max(set(List), key = List.count)
  else:
    if List[0] == 2:
      return 2
    return 0

@app.route('/', methods = ["GET", "POST"])
def index():
    flash("We have a Telegram chatbot @MelanomaCheck_bot to get a second opinion on the go!", "alert-primary")
    result = ""
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename) 
        if filename.split(".")[-1] not in ('jpg', 'png', 'jpeg', 'tiff', 'bmp', 'gif'):
            flash("Please upload an image file.", "alert-danger")
            return render_template("index.html", result = result)

        try:
            img = Image.open(request.files['file'])
            img = img.resize((224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis = 0)
        except Exception as e:
            print(e)
            flash("We are unable to properly read your image. Please try again.","alert-danger")
        else:
            result = get_pred_from_img(img.reshape(1, 224, 224, 3))
            if result == 0:
                flash(Markup('The image appears to be <b>melanoma</b>.<br><a href="/info">Learn more about melanoma</a>'),
                     'alert-warning')
            elif result == 1:
                flash(Markup('The image appears to be <b>nevus</b>.<br><a href="/info">Learn more about nevus</a>'),
                     'alert-success')
            elif result == 2:
                flash(Markup('The image appears to be <b>seborrheic keratosis</b>.<br><a href="/info">Learn more about seborrheic keratosis</a>'),
                     'alert-success')
        return render_template("index.html", result = result)
    else:
        return render_template("index.html", result = result)

@app.route('/info')
def info():
    return render_template("info.html")

if __name__ == '__main__':
    app.run()





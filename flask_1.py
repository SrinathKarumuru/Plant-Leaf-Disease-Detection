import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import tensorflow as tf
import mysql.connector
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import tensorflow as tf
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Flatten, Dropout, Conv2D
from keras.layers import BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from IPython.display import SVG, Image
from flask import Flask, render_template, redirect, request, session
app = Flask(__name__)
UPLOAD_FOLDER = r"C:\Users\prave\OneDrive\Documents\plant_disease_website\uploads"
model = tf.keras.models.load_model(
    r"C:\Users\prave\OneDrive\Documents\plant_disease_website\my_disease.h5")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = 'super secret key'


# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template('./index.html')


# @app.route('/index', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')


# @app.route('/main_page', methods=['GET', 'POST'])
# def main_page():
#     if not session.get("user_name"):
#         return render_template("login.html")
#     else:
#         name = session['user_name']
#     return render_template('main_page.html')


# @app.route('/about', methods=['GET', 'POST'])
# def about():
#     return render_template('about.html')


# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     return render_template('contact.html')


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     return render_template('login.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
    file1 = request.files['file1']
    if(not file1):
        render_template('detection.html', message='Please Upload Image')
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        test_image = image.load_img(path)
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import imshow
        plt.imshow(test_image)
        test_img = image.load_img(path, target_size=(48, 48))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        result = model.predict(test_img)
        a = result.argmax()
        s = {'Peach Bacterial_spot': 0, 'Peach healthy': 1, 'Pepper,_bell Bacterial_spot': 2, 'Pepper,_bell _healthy': 3, 'Potato Early_blight': 4, 'Potato Late_blight': 5, 'Potato healthy': 6, 'Tomato Bacterial_spot': 7, 'Tomato Early_blight': 8,
             'Tomato Late_blight': 9, 'Tomato Leaf_Mold': 10, 'Tomato Septoria_leaf_spot': 11, 'Tomato Spider_mites Two-spotted_spider_mite': 12, 'Tomato Target_Spot': 13, 'Tomato Tomato_Yellow_Leaf_Curl_Virus': 14, 'Tomato Tomato_mosaic_virus': 15, 'Tomato healthy': 16}
        name = []
        for i in s:
            name.append(i)
        for i in range(len(s)):
            if (i == a):
                p = name[i]
                render_template('main_page.html', prediction_text='The Predicted disease is {}'.format(
                    p), name=session['user_name'])


@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    global name, name1, user, results
    if request.method == "POST":
        email = request.form['username']
        password1 = request.form['password']
        session["name"] = email
        mydb = mysql.connector.connect(
            host="localhost", user="root", password="", database="plant_disease_detector")
        cursor = mydb.cursor()
        sql = "select * from user_details where email='%s' and password='%s'" % (
            email, password1)
        if(sql):
            x = cursor.execute(sql)
            results = cursor.fetchall()
            print(results)
            if(len(results) > 0):
                session['user_name'] = results[0][0]
                render_template('main_page.html', name=session['user_name'])

            else:
                render_template(
                    'login.html', message="Please Enter correct Credentials")


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session["name"] = None
    session["user_name"] = None
    render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    if request.method == "POST":
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        crop_name = request.form['crop']
        mobile = request.form['mobile']
        email = request.form['email']
        password = request.form['password']
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="plant_disease_detector")
        mycursor = mydb.cursor()

        sql = "INSERT INTO `user_details`(`first_name`, `last_name`, `crop_name`,`mobile`, `email`, `password`) VALUES(%s,%s,%s,%s,%s,%s) "
        val = (first_name, last_name, crop_name, mobile, email, password)
        mycursor.execute(sql, val)
        mydb.commit()
        render_template('signup.html', result="Successfully Registered")
        if name == " main ":
            app.run(debug=True)

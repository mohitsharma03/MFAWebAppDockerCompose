from flask import Flask, render_template,  request, Response, redirect, jsonify, make_response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import mysql.connector
from pipeline import Pipeline 
from faceDetect import FaceDetect

app = Flask(__name__)
CORS(app)
pipeLine = Pipeline()
fd = FaceDetect(0.50,"cpu")
UserName = None
UserImage  = None

mysql_conn = mysql.connector.connect(
    #host='localhost',
    host='mysql',
    user='root',
    #user='admin',
    password='mysql@123',
    database='mydatabase'
)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # get the form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        image = request.files['image']
        #save the image file to disk
        #image.save('static/images/' + image.filename)
        #image.save('../Data/images/' + name +'.png')
        #filepath = '../Data/images/'+name+'.png'
        filepath = 'Data/images/'+name+'.png'
        image.save(filepath)
        if(process_profile_img(filepath) == False):
            os.remove(filepath)
            return '<div style="text-align:center;"> <p>Issue with Image Please upload an image with face!!</p><a href="/">Login</a></div>'
        
        # create a new user in the database
        cursor = mysql_conn.cursor()
        #query = "INSERT INTO users (name, email, password, image) VALUES (%s, %s, %s, %s)"
       
        query = "INSERT INTO users (username, email, password, image) VALUES (%s, %s, %s, %s)"
        #values = (name, email, password, 'static/images/' + image.filename)
        #values = (name, email, password, '../Data/images/' + name +'.png')
        values = (name, email, password, filepath)
        cursor.execute(query, values)
        mysql_conn.commit()
        cursor.close()
        
        # render a success message to the user
        return '<div style="text-align:center;"> <p>User created successfully!!</p><a href="/">Login</a></div>'
    
    # if the request method is GET, render the signup form
    return render_template('signup.html')


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/test', methods=['POST'])
def test_cam():
    return render_template('webcam.html')
    
@app.route('/sendImages', methods=['POST'])
def image_process():
    prev = data_uri_to_cv2_img(request.form["prev"])
    curr = data_uri_to_cv2_img(request.form["curr"])
    flag =  request.form["flag"]
    user = request.form["user"]
    res = pipeLine.pipeline2(prev,curr, flag, user)
    #res = pipeLine.pipeline2(prev,curr, flag, UserName, UserImage)
    img = base64.b64encode(res[0])
    ret = {
        "img": str(img),
        "code": res[1]
    }
    return make_response(ret, 200)

@app.route('/video_feed', methods=['POST'])
def login_post():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # check if the user exists in the database
        cur = mysql_conn.cursor(buffered=True)
        #cur.execute('SELECT * FROM users WHERE name = %s', [username])
        cur.execute('SELECT * FROM users WHERE username = %s', [username])
        user = cur.fetchone()
        cur.close()
        print(user)
        """
        if username == 'admin' and password == 'password':
        #return render_template('face.html')
        return Response(pipeLine.runPipeline(), mimetype='multipart/x-mixed-replace; boundary=frame')
        """
        #if username==user[1] and password==user[3]:
        if username==user[0] and password==user[2]:
            # if the username and password are correct, log the user in
            #session['user_id'] = user[0]
            #session['username'] = user[1]
            #global UserName
            #global UserImage
            #UserName = user[1]
            #UserImage = user[4]
            #UserName = user[0]
            #UserImage = user[3]
            # redirect the user to the home page
            return render_template('webcam.html', user = user)

        else:
            #return redirect('/error', code=302)
            return redirect('/error')

@app.route('/error')
def errorPg():
   return 'Incorrect username or pass'

#@app.route('/video_feed')
#def video_feed():
#   return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
def process_profile_img(filepath):
    img = cv2.imread(filepath)
    height, width, c = img.shape
    numFaces, conf ,boxes = fd.detect(img)
    if(numFaces != 1):
        return False
    (x,y,w,h) = boxes[0]
    x, y, w, h = int(x), int(y), int(w), int(h)
    ymin = max(y-25,0)
    hmax = min(h+30,height)
    xmin = max(x-15,0)
    wmax = min(w+15,width)
    cv2.imwrite(filepath, cv2.resize(img[ymin:hmax,xmin:wmax],(180,250)))
    return True


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0',debug=True,port=5000)

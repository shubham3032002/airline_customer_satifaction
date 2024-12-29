from flask import Flask, render_template, request
import joblib
from config.path_config import *

app = Flask(__name__)

# Load the model
model_path = MODEL_SAVE_PATH
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':  # Corrected 'Post' to 'POST'
        try:
            # Collecting input data from the form
            data = [
                int(request.form['Online boarding']),
                float(request.form['Delay Ratio']),
                int(request.form['Inflight wifi service']),
                int(request.form['Class']),
                int(request.form['Type of Travel']),
                int(request.form['Inflight entertainment']),
                float(request.form['Flight Distance']),
                int(request.form['Seat comfort']),
                int(request.form['Leg room service']),
                int(request.form['On-board service']),
                int(request.form['Cleanliness']),
                int(request.form['Ease of Online booking']),
            ]
            
            # Make a prediction using the model
            prediction = model.predict([data])
            output = prediction[0]
            
            return render_template('index.html', prediction=output)
        except Exception as e:
            # Handle any errors and display them on the webpage
            return render_template('index.html', error=str(e))
    
    # Render the default page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)

from flask import Flask,request, render_template, jsonify
import os
import pycaret.regression as reg
import pycaret.classification as clf
import pandas as pd
import numpy as np
import json

app = Flask(__name__, template_folder='../templates', static_folder="../static")

app.config["DEBUG"] = True

UPLOAD_FOLDER = 'uploads/batch_files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

hp_model=reg.load_model('models/zhining_housing_prices_pipeline')
car_model=reg.load_model('models/used_car_price_pipeline')
ws_model=clf.load_model('models/pycaret_wheat_seed_pipline')

# Routes
@app.route('/')
def main_home():
    return render_template("home_page_combined.html")

@app.route('/hp_home')
def hp_home():
    return render_template("home_page_hp.html")

@app.route('/car_home')
def car_home():
    return render_template("home_page_uc.html")

@app.route('/wheat_home')
def wheat_home():
    return render_template("home_page_wheat.html")

# App functions for dataset 1 
@app.route('/predict_house_price', methods=['POST'])
def predict_house_price():
    # Get data from the form
    int_features = [x for x in request.form.values()]
    
    # Convert to a numpy array (1D)
    final = np.array(int_features)

    # The 7 columns you're using for prediction
    col = ['Distance', 'Age', 'Landsize', 'Bedroom2', 'BuildingArea', 'Type', 'Region']

    # Create a DataFrame for prediction
    data_unseen = pd.DataFrame([final], columns=col)

    # **Preprocessing**: Handle missing values and ensure that Age is numeric (same as CSV)
    data_unseen['Age'] = pd.to_numeric(data_unseen['Age'], errors='coerce')
    data_unseen['Age'] = data_unseen['Age'].fillna(data_unseen['Age'].median())

    # **Binning Age** (same as online version)
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '>100']
    data_unseen['Age'] = pd.cut(data_unseen['Age'], bins=bins, labels=labels, right=True)

    # **Predict using the loaded model**
    prediction = reg.predict_model(hp_model, data=data_unseen)

    # Reverse log1p for the predicted price
    predicted_log_price = prediction.prediction_label[0]
    predicted_price = np.expm1(predicted_log_price)

    # Round the result (optional, depending on your desired precision)
    predicted_price = round(predicted_price, 2)

    # Display the predicted price
    return render_template('home_page_hp.html', pred=f'Predicted Melbourne Housing Price will be ${predicted_price}')

@app.route('/predict_hp_csv', methods=['POST'])
def predict_hp_csv():
    # Get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        # The 7 columns you're using for prediction
        col = ['Distance', 'Age', 'Landsize', 'Bedroom2', 'BuildingArea', 'Type', 'Region']

        # Use Pandas to parse the CSV file
        data_unseen = pd.read_csv(file_path)

        # Check if the CSV contains the required columns
        data_unseen_cols = list(data_unseen.columns)

        if data_unseen_cols != col:
            return_error = f"Wrong CSV file. The CSV should have {col} as columns"
            return render_template('home.html', error=return_error)

        else:
            # **Preprocessing Step**: Ensure Age is treated as numeric and handle missing values
            data_unseen['Age'] = pd.to_numeric(data_unseen['Age'], errors='coerce')  # Convert 'Age' to numeric
            data_unseen['Age'] = data_unseen['Age'].fillna(data_unseen['Age'].median())  # Handle NaN in 'Age'

            # **Binning Age** (same as online version)
            bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
            labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', '>100']
            data_unseen['Age'] = pd.cut(data_unseen['Age'], bins=bins, labels=labels, right=True)

            # **Normalization Step**: Apply normalization to numeric features if required
            # Here we assume the model was trained with normalization; PyCaret handles this internally when we use `predict_model`

            # **Predict using the loaded model**
            prediction = reg.predict_model(hp_model, data=data_unseen)

            # Reverse log1p for the predicted price
            prediction['Predicted_Price'] = np.expm1(prediction['prediction_label'])

            # Round the prediction values
            prediction['Predicted_Price'] = prediction['Predicted_Price'].round(decimals=1)

            # Return the results as a table
            return render_template('display_table_hp.html', table=prediction.to_html(classes='table table-striped'))

# App functions for dataset 2
CAR_COLS = [
    "Brand_Model","Location","Year","Kilometers_Driven",
    "Fuel_Type","Transmission","Owner_Type",
    "Mileage_kmpl","Engine_cc","Power_bhp","Seats","Brand"
]

CAR_NUMERIC_COLS = {"Year","Kilometers_Driven","Mileage_kmpl","Engine_cc","Power_bhp","Seats"}

@app.route("/predict_car_price", methods=["POST"])
def predict_car_price():
    # Build single-row input
    row = {}
    for col in CAR_COLS:
        v = request.form.get(col, "")
        row[col] = (None if v == "" else float(v)) if col in CAR_NUMERIC_COLS else (v if v != "" else None)

    X = pd.DataFrame([row], columns=CAR_COLS)
    preds = reg.predict_model(car_model, data=X)
    price_lakhs = float(preds.loc[0, "prediction_label"])
    price_inr = round(price_lakhs * 100000)

    return render_template("home_page_uc.html",
                           pred=f"Predicted Price: ₹ {price_inr:,.0f}  ({price_lakhs:.2f} Lakhs)")

# App functions for dataset 3

@app.route('/predict_seed_type',methods=['POST'])
def predict_seed_type():
    try:
        int_features = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('home_page_wheat.html', error="All input features must be numeric.")
    col = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']
    if len(int_features) != len(col):
        return render_template('home_page_wheat.html', error=f"Expected {len(col)} features, got {len(int_features)}.")
    data_unseen = pd.DataFrame([int_features], columns = col)
    prediction=clf.predict_model(ws_model, data=data_unseen, round = 0)
    type_map = {1: "Kama", 2: "Rosa", 3: "Canadian"}
    predicted_type = int(prediction.loc[0, 'prediction_label'])
    type_name = type_map.get(predicted_type, f"Unknown ({predicted_type})")
    return render_template('home_page_wheat.html',pred='Wheat Seed is Type {}'.format(type_name))

@app.route('/predict_st_csv',methods=['POST'])
def predict_st_csv():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
    col = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']

    data_unseen = pd.read_csv(file_path)

    data_unseen_cols = []
    for column in data_unseen.columns:
        data_unseen_cols.append(column)

    if data_unseen_cols != col:
        return_error = f"Invalid csv file. The csv should have {col} as columns"
        return render_template('home_page_wheat.html',error=return_error)
    
    else:
        type_map = {1: "Kama", 2: "Rosa", 3: "Canadian"}
        prediction=clf.predict_model(ws_model, data=data_unseen)
        prediction["Predicted_Seed_Type"] = prediction["prediction_label"].map(lambda x: type_map.get(int(x), f"Unknown ({x})"))
        prediction = prediction.drop(columns=["prediction_label"])
        prediction=prediction.round(2)
        return render_template('display_table_wheat.html',table=prediction.to_html(classes='table table-striped table-bordered table-hover', index=False))

@app.route('/predict_st_api',methods=['POST'])
def predict_st_api():
    uploaded_file = request.files.get('file')
    records = json.load(uploaded_file)
    col = ['Area', 'Perimeter', 'Compactness', 'Length', 'Width', 'AsymmetryCoeff', 'Groove']
    data_unseen = pd.DataFrame(records)
    if list(data_unseen.columns) != col:
        return render_template('home_page_wheat.html', error=f"Invalid JSON structure. Expected columns: {col}")
    type_map = {1: "Kama", 2: "Rosa", 3: "Canadian"}
    prediction = clf.predict_model(ws_model, data=data_unseen)
    prediction["Predicted_Seed_Type"] = prediction["prediction_label"].map(lambda x: type_map.get(int(x), f"Unknown ({x})"))
    prediction = prediction.drop(columns=["prediction_label"])
    prediction = prediction.round(2)
    return render_template('display_table_wheat.html',table=prediction.to_html(classes='table table-striped table-bordered table-hover', index=False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)
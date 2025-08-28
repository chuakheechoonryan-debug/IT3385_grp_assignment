from flask import Flask,request, render_template, jsonify
import os
import pycaret.regression as reg
import pycaret.classification as clf
import pandas as pd
import numpy as np
import json
import hydra
from omegaconf import DictConfig
from os.path import splitext

app = Flask(__name__, template_folder='../templates', static_folder="../static")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    os.makedirs(cfg.app.upload_folder, exist_ok=True)
    app.config['UPLOAD_FOLDER'] =  cfg.app.upload_folder
    app.config["DEBUG"] = cfg.app.debug

    hp_model=reg.load_model(cfg.house.model_path)
    car_model=reg.load_model(cfg.car.model_path)
    ws_model=clf.load_model(cfg.wheat.model_path)

    # Routes
    @app.route('/')
    def main_home():
        return render_template(cfg.app.main_template)

    @app.route('/hp_home')
    def hp_home():
        return render_template(cfg.house.template)

    @app.route('/car_home')
    def car_home():
        return render_template(cfg.car.template)

    @app.route('/wheat_home')
    def wheat_home():
        return render_template(cfg.wheat.template)

    # App functions for dataset 1 
    @app.route('/predict_house_price', methods=['POST'])
    def predict_house_price():
        # Get data from the form
        int_features = [x for x in request.form.values()]
        
        # Convert to a numpy array (1D)
        final = np.array(int_features)

        # Create a DataFrame for prediction
        data_unseen = pd.DataFrame([final], columns=cfg.house.features)

        # **Preprocessing**: Handle missing values and ensure that Age is numeric (same as CSV)
        data_unseen['Age'] = pd.to_numeric(data_unseen['Age'], errors='coerce')
        data_unseen['Age'] = data_unseen['Age'].fillna(data_unseen['Age'].median())

        # **Binning Age** (same as online version)
        data_unseen['Age'] = pd.cut(data_unseen['Age'], bins=cfg.house.bins, labels=cfg.house.labels, right=True)

        # **Predict using the loaded model**
        prediction = reg.predict_model(hp_model, data=data_unseen)

        # Reverse log1p for the predicted price
        predicted_log_price = prediction.prediction_label[0]
        predicted_price = np.expm1(predicted_log_price)

        # Round the result (optional, depending on your desired precision)
        predicted_price = round(predicted_price, 2)

        # Display the predicted price
        return render_template(cfg.house.template, pred=f'Predicted Melbourne Housing Price will be ${predicted_price}')

    @app.route('/predict_hp_csv', methods=['POST'])
    def predict_hp_csv():
        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)

            # Use Pandas to parse the CSV file
            data_unseen = pd.read_csv(file_path)

            # Check if the CSV contains the required columns
            data_unseen_cols = list(data_unseen.columns)

            if data_unseen_cols != cfg.house.features:
                return_error = f"Wrong CSV file. The CSV should have {cfg.house.features} as columns"
                return render_template(cfg.house.template, error=return_error)

            else:
                # **Preprocessing Step**: Ensure Age is treated as numeric and handle missing values
                data_unseen['Age'] = pd.to_numeric(data_unseen['Age'], errors='coerce')  # Convert 'Age' to numeric
                data_unseen['Age'] = data_unseen['Age'].fillna(data_unseen['Age'].median())  # Handle NaN in 'Age'

                # **Binning Age** (same as online version)
                data_unseen['Age'] = pd.cut(data_unseen['Age'], bins=cfg.house.bins, labels=cfg.house.labels, right=True)

                # **Normalization Step**: Apply normalization to numeric features if required
                # Here we assume the model was trained with normalization; PyCaret handles this internally when we use `predict_model`

                # **Predict using the loaded model**
                prediction = reg.predict_model(hp_model, data=data_unseen)

                # Reverse log1p for the predicted price
                prediction['Predicted_Price'] = np.expm1(prediction['prediction_label'])

                # Round the prediction values
                prediction['Predicted_Price'] = prediction['Predicted_Price'].round(decimals=1)

                # Return the results as a table
                return render_template(cfg.house.display_template, table=prediction.to_html(classes='table table-striped'))

    # App functions for dataset 2
    @app.route("/predict_car_price", methods=["POST"])
    def predict_car_price():
        # Build single-row input
        row = {}
        for col in cfg.car.features:
            v = request.form.get(col, "")
            row[col] = (None if v == "" else float(v)) if col in cfg.car.numeric_features else (v if v != "" else None)

        X = pd.DataFrame([row], columns=cfg.car.features)
        preds = reg.predict_model(car_model, data=X)
        price_lakhs = float(preds.loc[0, "prediction_label"])
        price_inr = round(price_lakhs * 100000)

        return render_template(cfg.car.template,
                            pred=f"Predicted Price: ₹ {price_inr:,.0f}  ({price_lakhs:.2f} Lakhs)")

    @app.route("/predict_cp_csv", methods=["POST"])
    def predict_cp_csv():
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            return render_template(cfg.car.template, error="No file uploaded.")

        ext = splitext(uploaded.filename)[1].lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(uploaded)  # read stream directly
            elif ext in {".xlsx", ".xls"}:
                # requires openpyxl in requirements for .xlsx
                df = pd.read_excel(uploaded, engine="openpyxl")
            else:
                return render_template(cfg.car.template, error="Upload .csv, .xlsx or .xls files.")
        except Exception as e:
            return render_template(cfg.car.template, error=f"Failed to read file: {e}")

        # Validate schema (order doesn't matter)
        missing = [c for c in cfg.car.features if c not in df.columns]
        if missing:
            return render_template(cfg.car.template,
                                error=f"Wrong schema. Missing: {missing}. Expected at least: {cfg.car.features}")

        df_in = df[cfg.car.features].copy()
        for c in cfg.car.numeric_features:
            df_in[c] = pd.to_numeric(df_in[c], errors="coerce")

        preds = reg.predict_model(car_model, data=df_in)
        out = df_in.copy()
        out["predicted_price_lakhs"] = preds["prediction_label"].round(2)
        out["predicted_price_inr"]   = (preds["prediction_label"] * 100000).round(0).astype(int)

        return render_template(cfg.car.display_template, table=out.to_html(classes="table table-striped", index=False))

    # App functions for dataset 3

    @app.route('/predict_seed_type',methods=['POST'])
    def predict_seed_type():
        try:
            int_features = [float(x) for x in request.form.values()]
        except ValueError:
            return render_template(cfg.wheat.template, error="All input features must be numeric.")
        if len(int_features) != len(cfg.wheat.features):
            return render_template(cfg.wheat.template, error=f"Expected {len(cfg.wheat.features)} features, got {len(int_features)}.")
        data_unseen = pd.DataFrame([int_features], columns = cfg.wheat.features)
        prediction=clf.predict_model(ws_model, data=data_unseen, round = 0)
        predicted_type = int(prediction.loc[0, 'prediction_label'])
        type_name = cfg.wheat.type_map.get(predicted_type, f"Unknown ({predicted_type})")
        return render_template(cfg.wheat.template,pred='Wheat Seed is Type {}'.format(type_name))

    @app.route('/predict_st_csv',methods=['POST'])
    def predict_st_csv():
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return render_template(cfg.wheat.template, error="No file uploaded.")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        data_unseen = pd.read_csv(file_path)

        data_unseen_cols = []
        for column in data_unseen.columns:
            data_unseen_cols.append(column)

        if data_unseen_cols != cfg.wheat.features:
            return_error = f"Invalid csv file. The csv should have {cfg.wheat.features} as columns"
            return render_template(cfg.wheat.template,error=return_error)
        
        else:
            prediction=clf.predict_model(ws_model, data=data_unseen)
            prediction["Predicted_Seed_Type"] = prediction["prediction_label"].map(lambda x: cfg.wheat.type_map.get(int(x), f"Unknown ({x})"))
            prediction = prediction.drop(columns=["prediction_label"])
            prediction=prediction.round(2)
            return render_template(cfg.wheat.display_template,table=prediction.to_html(classes='table table-striped table-bordered table-hover', index=False))

    @app.route('/predict_st_api',methods=['POST'])
    def predict_st_api():
        uploaded_file = request.files.get('file')
        if not uploaded_file:
            return render_template(cfg.wheat.template, error="No file uploaded.")

        records = json.load(uploaded_file)
        data_unseen = pd.DataFrame(records)
        if list(data_unseen.columns) != cfg.wheat.features:
            return render_template(cfg.wheat.template, error=f"Invalid JSON structure. Expected columns: {cfg.wheat.features}")
        prediction = clf.predict_model(ws_model, data=data_unseen)
        prediction["Predicted_Seed_Type"] = prediction["prediction_label"].map(lambda x: cfg.wheat.type_map.get(int(x), f"Unknown ({x})"))
        prediction = prediction.drop(columns=["prediction_label"])
        prediction = prediction.round(2)
        return render_template(cfg.wheat.display_template,table=prediction.to_html(classes='table table-striped table-bordered table-hover', index=False))

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=cfg.app.debug)

if __name__ == "__main__":
    main()
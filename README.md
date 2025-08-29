# IT3385_grp_assignment

## Team Members
- Chua Khee Choon Ryan (220862D)
- Ong Sze Xuan Janelle (220932Y)
- Tan Zhi Ning (220843P)

## Deployment Guide
### Locally
To deploy the Web Application locally on Docker, do the following:
- Download contents of repository
- Place contents of download zip file into a folder inside a folder inside a folder (ensures docker does not copy anything unwanted, C:\Dev_Assignment\Assignment\assignment_contents e.g.)
- Ensure Docker Desktop and Anaconda are installed on your local device
    - Docker Desktop: https://www.docker.com/get-started/
    - Anaconda (Distribution Installer): https://www.anaconda.com/download/success
- Ensure proper virtual environment is setup in anaconda with python version 3.10 and Pycaret:
    -     conda create --name [virutal env name] python=3.10
    -     pip install pycaret[full]
- Run docker build to create container
    -     docker build -t [container name]:latest .
- Then run created docker container on appropriate host port:container port (5000:5000 etc.)
    -     docker run -d -p [host port]:[container port] [container name]
- It should then appear as seen in docker desktop:

<img width="2517" height="1349" alt="image" src="https://github.com/user-attachments/assets/abc7584b-efd6-4f78-8a32-43e71ded425c" />

- Run the container to deploy it locally and click on the port to visit the locally deployed web application

### Online (Render)
- Copy the link to the github repo (HTTPS)
- Go to: https://render.com/
- Click 'Get Started for Free'
- Create Account/Sign in to render
- Click New → Web Service
- Select Public Git Repository and paste in the the copied HTTPS link

<img width="2407" height="533" alt="image" src="https://github.com/user-attachments/assets/090ed017-a78e-44b0-9d17-902e38cc3939" />

- Settings should be as followed:

<img width="2417" height="885" alt="image" src="https://github.com/user-attachments/assets/d4b64d3b-c668-45e6-8814-44074b4fc511" />

- Choose appropriate instance type
- Click deploy web service to deploy it

## User Guide
Web Application consists of 3 pages:
- Melbourne Property Selling Price Prediction
- Used Car Selling Price Prediction
- Wheat Seed Type Prediction

Each of them consist of online and batch predictions
- Online prediction,
    - write/paste or select appropriate values for each input box and click predict
    - Prediction will be displayed on the screen
- Batch prediction:
    - Select batch option from dropdown or scroll to bottom of page (for Used Car Selling Price Prediction)
    - Allowed File Formats: CSV, JSON (only for Wheat Seed Type Prediction, under Batch (JSON) option)
    - Results will be printed as a table in a separate page that user will be redirected to

## Folder Structure
```
IT3385_grp_assignment/
│   .gitignore
│   Dockerfile
│   README.md
│   requirements.txt
│
├───conf/
│       car.yaml
│       config.yaml
│       house.yaml
│       wheat.yaml
│
├───data/
│   ├───original_assignment_data
│   │       01_Melbourne_Residential.csv
│   │       02_Used_Car_Prices.xlsx
│   │       03_Wheat_Seeds.csv
│   │
│   └───test_data/
│           app_test.csv
│           sample_used_cars_batch.csv
│           synthetic_test_data_first5.json
│           testdata2.csv
│
├───models/
│       pycaret_wheat_seed_pipline.pkl
│       used_car_price_pipeline.pkl
│       zhining_housing_prices_pipeline.pkl
│
├───notebooks/
│       assignment_notebook_ryan.ipynb
│       IT3385_220843P.ipynb
│       IT3385_janelle.ipynb
│
├───src/
│       assignment_combi_app.py
│
├───static/
│       style.css
│
├───templates/
│       display_table_hp.html
│       display_table_uc.html
│       display_table_wheat.html
│       home_page_combined.html
│       home_page_hp.html
│       home_page_uc.html
│       home_page_wheat.html
│
└───uploads/
    └───batch_files/
```

## URLs for GitHub Repository and Render Web Application
Github Repository: https://github.com/chuakheechoonryan-debug/IT3385_grp_assignment.git

Render Web Application: https://it3385-grp-assignment.onrender.com/

# IT3385_grp_assignment

## Team Members
- Chua Khee Choon Ryan (220862D)
- Ong Sze Xuan Janelle (220932Y)
- Tan Zhi Ning (220843P)

## Deployment Guide
To deploy the Web Application locally on Docker, do the following:
- Download contents of repository
- Place contents of download zip file into a folder inside a folder inside a folder (ensures docker does not copy anything unwanted, C:\Dev_Assignment\Assignment\assignment_contents e.g.)
- Ensure Docker Desktop and Anaconda are installed on your local device
- Ensure proper environment is setup:
-     python=3.10 conda (create --name mlops python=3.10)


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

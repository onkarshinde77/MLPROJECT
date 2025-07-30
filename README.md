# End to End machine learning project

# Other Information : 
### In a machine learning (ML) project, a setup.py file is used primarily to package and distribute your project as a Python package. This file contains metadata about your project—such as its name, version, author, dependencies (required libraries like NumPy, Pandas, scikit-learn), and other configuration details—to enable easy installation and reuse

## Note : The __init__.py file is used in Python projects to mark a directory as a Python package. This means that when a directory contains an __init__.py file, Python treats it as a package, allowing you to import modules from that directory using the package syntax.

# Process to creating project
1. **Initialize/create git repo and clone it in code**
2. **Create vertual environment & activate**
```
python -m venv venv
venv\Scripts\activate
```
3. **Create setup.py for setup dependency & configuration**
    - key Features:
      - Package Metadata
      - Installation Instructions
      - Distribution
      - Dependency Management
      - Entry Points and Scripts
      - versions & author    etc....

4. **Create Readme.md file**
```
touch README.md
```

3. **create**
    - .gitignore
    - run.sh (write all command from begening to end)
    - readme.md
    - setup.py 
    - requirements.txt (include (-e .) in last)
### Note : push/pull everytime code on github

4. **Create logger & exception**
    - logger.py : creates a reusable logger configuration that writes logs to both the console and a file, includin 
                  timestamps and log levels.
    - exception.py : user define exception using sys library

5. **Create Notebook directory to Perform EDA(Exploratory Data Analysis) to understand the Dataset**
    Notebook/
        ├── EDA.ipynb
        ├── model_train.ipynb
        └── data/               --> save the clean dataset

6. **Create given folders**
    src/
    ├── components/ (data_ingection.py , data_transformation.py , model_trainer.py , __init__.py)
    ├── pipeline/
    │     ├── predict_pipeline.py
    │     └── train_pipeline.py
    ├── exception.py
    ├── logger.py
    ├── utils.py
    └── __init__.py

7. **Now create data_injection :** 
    - A data injection file in a project (often called "data ingestion" or "data injection") is typically created to handle the process of collecting, loading, and injecting raw or external data into your system or pipeline. The purpose of such a file or module is to centralize and organize how raw data enters your project for further processing, analysis, or model training.
8. **Create data_transformation.py & model_trainer.py**
    - data_transformation.py : seperate the num columns & catagorical columns and tranform it by using OneHotEncoder & StandardScaler and return train & test array (preprocessor.pkl)
    - model_trainer.py : seperate input & target features and hyperparameter tunning. then return best models dictionary contains the models . (model.pkl)
9. **Now create the train_pipeline.py & predict_pipeline.py**
10. **Create backend in Flask/django with frontend using jinja templates** 
11. **Deploy on AWS/render**


## Final ML project Architecture

MLPROJECT/
│
├── app.py ───── Flask web server (serving ML API/UI)
├── templates/ ── HTML frontend (Jinja2 templates)
│
├── artifacts/ ─ Trained model files, datasets, preprocessors
│     ├── model.pkl
│     ├── preprocessor.pkl
│     ├── train.csv
│     └── test.csv
│
├── Notebook/
│     ├── EDA.ipynb
│     ├── model_train.ipynb
│     └── data/
│
├── src/
│    ├── components/  # Modular ML/data pipeline pieces
│    ├── pipeline/
│    │     ├── predict_pipeline.py
│    │     └── ...
│    ├── exception.py
│    ├── logger.py
│    ├── utils.py
│    └── __init__.py
│
├── venv/ ─────── Python virtual environment
├── requirements.txt
├── setup.py
├── .gitignore
│
└── node_modules/  # If using Node for some frontend assets (optional)

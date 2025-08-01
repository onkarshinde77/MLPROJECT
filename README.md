# End to End machine learning project

### 🚀 Excited to share my end-to-end Machine Learning project!
**I built a predictive application that estimates student math scores using features like gender, race/ethnicity, parental education, lunch type, test preparation, and reading & writing scores.**

**Here’s what I worked on:**
**📊 Deep exploratory data analysis (EDA)**
**💡 Advanced feature engineering for improved performance**
**🔧 Hyperparameter tuning and testing multiple algorithms**
**🛠️ Designed robust ML pipelines and ensured clean exception handling**
**⛓️ Backend API developed with Flask for easy integration**
**🐳 Containerized the project with Docker for reproducibility**
**☁️ Seamless deployment on AWS Elastic Beanstalk**
**⬆️ Source code and workflow managed via GitHub**

**Why this matters:**
**This project expanded my hands-on expertise in practical ML, DevOps best practices, and deploying reliable AI services to production.**
**It’s been a fantastic learning journey, from raw data to a fully-operational cloud deployment!**

**🔗 Feel free to check out the project or connect if you want to chat about ML engineering.**


# Other Information : 
### In a machine learning (ML) project, a setup.py file is used primarily to package and distribute your project as a Python package. This file contains metadata about your project—such as its name, version, author, dependencies (required libraries like NumPy, Pandas, scikit-learn), and other configuration details—to enable easy installation and reuse

### Note : The __init__.py file is used in Python projects to mark a directory as a Python package. This means that when a directory contains an __init__.py file, Python treats it as a package, allowing you to import modules from that directory using the package syntax.

# Process to creating project
1. **Initialize/create git repo and clone it in code**
2. **Create vertual environment & activate**
```
python -m venv venv
```
```
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

5. **Create Notebook directory to Perform EDA(Exploratory Data Analysis) to understand the Dataset**<br>
    Notebook/<br>
        ├── EDA.ipynb<br>
        ├── model_train.ipynb<br>
        └── data/               --> save the clean dataset<br>

6. **Create given folders**<br>
    src/<br>
    ├── components/ (data_ingection.py , data_transformation.py , model_trainer.py , __init__.py)<br>
    ├── pipeline/<br>
    │     ├── predict_pipeline.py<br>
    │     └── train_pipeline.py<br>
    ├── exception.py<br>
    ├── logger.py<br>
    ├── utils.py<br>
    └── __init__.py<br>

7. **Now create data_injection :** 
    - A data injection file in a project (often called "data ingestion" or "data injection") is typically created to handle the process of collecting, loading, and injecting raw or external data into your system or pipeline. The purpose of such a file or module is to centralize and organize how raw data enters your project for further processing, analysis, or model training.

8. **Create data_transformation.py & model_trainer.py**
    - data_transformation.py : seperate the num columns & catagorical columns and tranform it by using OneHotEncoder & StandardScaler and return train & test array (preprocessor.pkl)
    - model_trainer.py : seperate input & target features and hyperparameter tunning. then return best models dictionary contains the models . (model.pkl)
9. **Now create the train_pipeline.py & predict_pipeline.py**
10. **Create backend in Flask/django with frontend using jinja templates** 
11. **Deploy on AWS/render**

## Final ML project Architecture

MLPROJECT/<br>
│<br>
├── app.py ───── Flask web server (serving ML API/UI)<br>
├── templates/ ── HTML frontend (Jinja2 templates)<br>
│<br>
├── artifacts/ ─ Trained model files, datasets, preprocessors<br>
│     ├── model.pkl<br>
│     ├── preprocessor.pkl<br>
│     ├── train.csv<br>
│     └── test.csv<br>
│<br>
├── Notebook/<br>
│     ├── EDA.ipynb<br>
│     ├── model_train.ipynb<br>
│     └── data/<br>
│<br>
├── src/<br>
│    ├── components/  # Modular ML/data pipeline pieces<br>
│    ├── pipeline/<br>
│    │     ├── predict_pipeline.py<br>
│    │     └── ...<br>
│    ├── exception.py<br>
│    ├── logger.py<br>
│    ├── utils.py<br>
│    └── __init__.py<br>
│<br>
├── venv/ ─────── Python virtual environment<br>
├── requirements.txt<br>
├── setup.py<br>
├── .gitignore<br>
│<br>
└── node_modules/  # If using Node for some frontend assets (optional)<br>

## 👤 Author

### **Onkar Shinde**

<p align="left">
  <a href="https://www.linkedin.com/in/onkarshinde77" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="https://hub.docker.com/repository/docker/onkarshinde77/mlproject/general" target="_blank">
    <img src="https://img.shields.io/badge/Docker Hub-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Badge"/>
  </a>
  <a href="https://leetcode.com/onkarshinde77" target="_blank">
    <img src="https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=LeetCode&logoColor=black" alt="LeetCode Badge"/>
  </a>
</p>

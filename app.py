from flask import Flask , render_template , request
from src.pipeline.predict_pipeline import CustomData,CustomException
from src.pipeline.predict_pipeline import PredictPipelines

app = Flask(__name__)

@app.route('/')
def index():
    title = 'Welcome'
    name = 'onkar'
    return render_template('index.html',name=name,title=title)

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method =='POST':
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )
        pred_df = data.get_data_in_dataframe()
        pred_obj = PredictPipelines()
        pred = pred_obj.predict(pred_df)
        return render_template('home.html',results=pred[0])
    return render_template('home.html',results=None)

if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
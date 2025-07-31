from flask import Flask , render_template , request
from src.pipeline.predict_pipeline import CustomData,CustomException
from src.pipeline.predict_pipeline import PredictPipelines

app = Flask(__name__)

@app.route('/')
def index():
    title = 'Home'
    description = "Welcome to My ML Project,To saw my project to '/predict' endpoint and for check more information go to '/about' endpoint"
    return render_template('index.html',title=title, description=description)

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
        return render_template('home.html',results=pred[0],display1=None)
    return render_template('home.html',display2=None)

from flask import render_template

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )

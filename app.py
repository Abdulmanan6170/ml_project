from flask import Flask, request, render_template
from src.Pipeline.predict_pipeline import CustomData, PredictPipeline

# Create Flask app
application = Flask(__name__)
app = application


@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Prediction route"""
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # Collect form data
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score', 0)),
            writing_score=float(request.form.get('writing_score', 0))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:\n", pred_df)

        # Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])

    except Exception as e:
        print("Prediction Error:", e)
        return render_template('home.html', error=str(e))


if __name__ == "__main__":
    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=True)

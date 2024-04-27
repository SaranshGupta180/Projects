from flask import Flask, render_template, request,jsonify
import pickle
from utils import *

app= Flask(__name__)

with open('model/spam_model.pkl','rb') as model_file:
    spam_model = pickle.load(model_file)

@app.route('/spam', methods=['POST'])
def spam():
    text=request.form['text']
    text=transform(text)
    prediction = spam_model.predict(text)
    return jsonify({'prediction':'spam' if prediction==1 else 'ham'})

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="0.0.0.0",port='5000')





















'''
    #user_input = np.expand_dims(np.array([request.form['text']]), axis=-1)

    #input
    user_input = request.form['text']
    user_input=" ".join(user_input)

    #preprocessing
    transformed_input=transform(user_input)
    transformed_input=" ".join(transformed_input)

    #vectorsisation
    cv=CountVectorizer()

    #data = StringIO(transformed_input)
    #df = pd.read_csv(data, sep=",")

    #df=pd.DataFrame( list(reader(transformed_input)))

    vectorsied_input=cv.transform(['transformed_input']).toarray()

    #result
    prediction=spam_model.predict(vectorsied_input)
     

    return jsonify(prediction)
    #return render_template("index.html",prediction_text="Given SMS is {}".format(predict))  
    '''   
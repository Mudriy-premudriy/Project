from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import numpy as np
import pickle
from flask import Flask, request

X, y = load_diabetes(return_X_y=True)
X = X[:, 0].reshape(-1, 1) # Берём только один признак
regressor = LinearRegression()
regressor.fit(X,y)

with open('myfile.pkl', 'wb') as output:
   	pickle.dump(regressor, output) #Сохраняем

with open('myfile.pkl', 'rb') as pkl_file:
    regressor_from_file = pickle.load(pkl_file) #Загружаем

app = Flask(__name__)
def model_predict(value):
	print(type(value))
	value_to_predict = np.array([value]).reshape(-1,1)
	return regressor_from_file.predict(value_to_predict)[0]

@app.route('/predict')
def hello_func():
	value1 = request.args.get('value')
	value = float(value1) #Приводим к типу Float
	prediction = model_predict(value)
	return f'the result is {prediction}!'

if __name__ == '__main__':
    		app.run('localhost', 5002)

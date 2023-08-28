
# Stock Market prediction 

In this project, I use the lstm model to predict the future value of a data set .first of all I make a lstm model and save it and use it for future prediction. In this project, I use a data set of a company that is about 11 years data and I train my model on this data set then I use some test cases, in which I use the previous 50 days' data and predict the next two days close value.


## Dependencies

This project has been developed using the following libraries:

- [NumPy](https://numpy.org): A library for numerical computing with Python.
- [Pandas](https://pandas.pydata.org): A library for data manipulation and analysis.
- [scikit-learn](https://scikit-learn.org): A library for machine learning and data preprocessing.
- [TensorFlow](https://www.tensorflow.org): An open-source machine learning framework.
- [Keras](https://keras.io): A high-level deep learning framework.
- [Pickle](https://docs.python.org/3/library/pickle.html): A library for object serialization.

Please ensure that you have these libraries installed or include them in your project environment before running the code.

### Modules and Classes Used

- `numpy`: Imported as `np` to perform numerical computations.
- `pandas`: Imported as `pd` to handle data manipulation and analysis.
- `sklearn.preprocessing.MinMaxScaler`: Used for feature scaling.
- `tensorflow.keras.models.load_model`: Used to load a saved model.
- `keras.models.Sequential`: Used to create a sequential model.
- `keras.layers.Dense`: Used to add dense layers to the model.
- `keras.layers.LSTM`: Used to add LSTM layers to the model.
- `pickle`: Used for object serialization.
- `sklearn.metrics.mean_squared_error`: Used to calculate the mean squared error.

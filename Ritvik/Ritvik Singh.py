import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import warnings
warnings.filterwarnings("ignore")


def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    

def predict_func(data):
    """first i read the data and fill na value with previous two and next two value mean.
    after it i make a lstm model which depend previos 50 value and then i save the model 
    for future prediction.
    this is the code for lstm model =>
    
    df = pd.read_csv('0f411c708e55af442eafb33bfb7ee7585f5b0211a52d9ccc4287a23d8d6abe76_STOCK_INDEX.csv')
    df.set_index('Date', inplace=True)
    data = df.fillna((df.shift(1) + df.shift(-1) + df.shift(2) + df.shift(-2)) / 4)
    data = data.filter(['Close'])
    dataset = data.values

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) *.65))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Create the training data set
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(50, len(train_data)):
        x_train.append(train_data[i - 50:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape

    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Dropout,GRU

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=10, epochs=10)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 50:, :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(50, len(test_data)):
        x_test.append(test_data[i - 50:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = (np.mean(((predictions - y_test) ** 2)))
    print(rmse)

    #save the model
    model.save('lstm_model.h5')
    
    """
    #now we use the model for prediction =>
    # Load the saved model
    model = load_model('lstm_model.h5')

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    data.set_index('Date', inplace=True)
    data = data.fillna((data.shift(1) + data.shift(-1) + data.shift(2) + data.shift(-2)) / 4)
    data_c = data[['Close']]
    new_df = scaler.transform(data_c)

    next_day1 = []
    for i in range(50, len(new_df)):
        next_day1.append(new_df[i - 50:i, 0])
        
    next_day1 = np.array(next_day1)

    # Reshape the data
    next_day1 = np.reshape(new_df, (1, new_df.shape[0], new_df.shape[1]))

    # Get the models predicted price values
    predictions1 = model.predict(next_day1)
    predictions11 = scaler.inverse_transform(predictions1)

    new_df = np.append(new_df, predictions1[0][0])
    new_df = new_df[1:]
    new_df = np.reshape(new_df, (new_df.shape[0], 1))

    next_day2 = []
    for i in range(50, len(new_df)):
        next_day2.append(new_df[i - 50:i, 0])

    next_day2 = np.array(next_day2)

    next_day2 = np.reshape(new_df, (1, new_df.shape[0], new_df.shape[1]))
    predictions2 = model.predict(next_day2)
    predictions22 = scaler.inverse_transform(predictions2)

    return [predictions11[0][0], predictions22[0][0]]

    

if __name__== "__main__":
    evaluate()
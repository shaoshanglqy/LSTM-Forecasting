
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    
    for i in range(len(sequence)):
        
        #find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        
        #checkif we are beyond the dataset
        if out_end_ix > len(sequence):
            break
            
        #gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)



def trainModel(n_steps_in, n_steps_out,model_train_X, model_train_y,model_test_X, model_test_y):
    
    #define model
    n_features = 1

    model = Sequential()
    model.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape = (n_steps_in, n_features)))
    model.add(LSTM(100, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out))
    model.compile(optimizer = 'adam', loss = 'mse')

    # fit model
    history = model.fit(model_train_X, model_train_y, epochs = 50, verbose = 1, validation_data = (model_test_X, model_test_y))
    
    plt.plot(history.history['loss'], label = 'train loss')
    plt.plot(history.history['val_loss'], label = 'test loss')
    plt.legend()
    plt.show()

    return  model

def train_mse(model, model_train_X,model_train_y,n_steps_out):
    
    yhat_t = model.predict(model_train_X)
    train_y_out = [[] for i in range(n_steps_out)]
    yhat_t_out = [[] for i in range(n_steps_out)]
    mse = [[] for i in range(n_steps_out)]

    #get the result seperately from each step out

    for i in range(len(yhat_t)):
        for k in range(n_steps_out):
            yhat_t_out[k].append(yhat_t[i][k])  #get each yhat result from each step out
            train_y_out[k].append(model_train_y[i][k])  #get each model result from each step out


    #mse from each time step
    for i in range(n_steps_out):
        mse[i] = ((np.array(yhat_t_out[i]) - np.array(train_y_out[i]))**2).mean()
        
        
    #plot yhat vs true value on every predicted day

    plt.figure(figsize=(8,4))
    for i in range(n_steps_out):
        plt.subplot(n_steps_out, 1, i +1)

        pd.Series(train_y_out[i]).plot()
        pd.Series(yhat_t_out[i]).plot()

    print('train_mse: ', mse)
    plt.show()

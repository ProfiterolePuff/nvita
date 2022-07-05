
def normalize_data(data, feature_range=(0, 1), test_percentage = 0.1, standard = True):
    readable_test_set_size = int(np.round(test_percentage * data.shape[0])) # no + 1 here unlike split_data
    readable_train_set_size = data.shape[0] - readable_test_set_size
    
    if standard:
        scaler = StandardScaler(feature_range = feature_range)
    else:
        scaler = MinMaxScaler(feature_range = feature_range)

    data_train = scaler.fit_transform(data.iloc[ : readable_train_set_size])
    data_test = scaler.transform(data.iloc[readable_train_set_size : ])
    
    normalized_data = pd.DataFrame(np.concatenate((data_train, data_test), axis=0), columns = list(data.columns))
    
    return normalized_data
                     

def split_data_with_window_ranges(data, target_data, test_percentage = 0.1, training_length = 5):
    test_window_ranges = []
    
    raw_x_data = data.to_numpy()
    raw_y_data = target_data.to_numpy().reshape(target_data.shape[0],1)

    x_data, y_data = [], []
    for index in range(len(raw_x_data) - training_length - 1): 
        x_data.append(raw_x_data[index: index + training_length + 1])
        y_data.append(raw_y_data[index: index + training_length + 1])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    test_set_size = int(np.round(test_percentage * x_data.shape[0])) + 1
    train_set_size = x_data.shape[0] - test_set_size
   
    x_train = x_data[:train_set_size, :-1,:]
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    
    y_train = y_data[:train_set_size, -1, :]
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    
    x_test = x_data[train_set_size:, :-1]
    
    for i in range(x_test.shape[0]):
        one_test_window_ranges = []
        for f in range(x_test.shape[2]):
            one_test_window_ranges.append(np.ptp(x_test[i : i + 1, :, f : f + 1]))
        test_window_ranges.append(torch.FloatTensor(one_test_window_ranges))
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    
    y_test = y_data[train_set_size:, -1, :]
    y_test= torch.from_numpy(y_test).type(torch.Tensor)
    
    single_x_shape = list(x_test.shape)
    single_x_shape[0] = 1
    single_x_shape
    
    single_y_shape = list(y_test.shape)
    single_y_shape[0] = 1
    single_y_shape
    
    return x_train, y_train, x_test, y_test, test_window_ranges, single_x_shape, single_y_shape
    

def training_model(model, criterion, optimiser, num_epochs, x_train, y_train):
    start_time = time.time()
    losses = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        losses[epoch] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    cost_time = time.time() - start_time
    print("Training time of " + model.get_name() + " is: " + str(cost_time) + " Sec.")
    return losses 


def train(model, X):
    model.train()
    # Here is the for loop.
    pass

def evaluate(model, X, y):
    model.eval()
    return 0.

def predict(model, X):
    model.eval()
    return model(X)


from sklearn.svm import SVR

def train_svr(X_train, y_train):
    # You can modify C, epsilon, and kernel as needed
    model = SVR(kernel='rbf', C=100, epsilon=0.01)
    model.fit(X_train, y_train)
    return model

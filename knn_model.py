from sklearn.neighbors import KNeighborsRegressor

def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    model.fit(X_train, y_train)
    return model
from sklearn.tree import DecisionTreeRegressor

def train_rt(X_train, y_train, max_depth=None):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model

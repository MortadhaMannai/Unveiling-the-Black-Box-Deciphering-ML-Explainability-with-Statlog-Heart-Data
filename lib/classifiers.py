def create_rf_estimator(verbose, random_state, n_jobs):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=2,
                                  max_features='auto', max_depth=30, bootstrap=False,
                                  criterion='gini',
                                  random_state=random_state, verbose=verbose, n_jobs=n_jobs)


def create_xgb_estimator(verbose, random_state, n_jobs):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=30, max_depth=20, n_jobs=n_jobs, random_state=random_state, verbose=verbose)


def create_knn_estimator(verbose, random_state, n_jobs):
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)


def create_nb_estimator(verbose, random_state, n_jobs):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB()


def create_qda_estimator(verbose, random_state, n_jobs):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis()


def create_ada_estimator(verbose, random_state, n_jobs):
    from sklearn.ensemble import AdaBoostClassifier
    return AdaBoostClassifier(random_state=random_state)


def create_mlp_estimator(verbose, random_state, n_jobs):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(alpha=1, max_iter=200, random_state=random_state, verbose=verbose)


def create_fcn_estimator(verbose, random_state, n_jobs):
    import keras
    from keras import layers
    from keras import optimizers
    from keras.wrappers.scikit_learn import KerasClassifier

    def create_model(optimizer='rmsprop', kernel_initializer='glorot_uniform', scale_factor=1):
        model = keras.Sequential([
          layers.Dense(32 * scale_factor, input_dim=13, activation='relu', kernel_initializer=kernel_initializer),
          layers.Dense(64 * scale_factor, activation='relu', kernel_initializer=kernel_initializer),
          layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        return model

    return KerasClassifier(build_fn=create_model, epochs=10, batch_size=256, verbose=verbose)


classifier_factories = {
    'Random Forest': create_rf_estimator,
    'XGBoost': create_xgb_estimator,
    'K-Nearest Neighbour': create_knn_estimator,
    'Naive Bayes': create_nb_estimator,
    'Quadratic Discriminant Analysis': create_qda_estimator,
    'Adaptive Boosting': create_ada_estimator,
    'Multilayer Perceptron': create_mlp_estimator,
    'Deep Neural Network': create_fcn_estimator,
}
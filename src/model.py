import pandas as pd, numpy as np, pickle, os
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold

x = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\training-x2.csv").values
y = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\training-y2.csv").values
modelfile = r"C:\Users\Brian\Desktop\Programming\datathon\dataset\model.sav"

def genModel(train_all):
    model = MultiOutputRegressor(
        SGDRegressor(loss="squared_loss", penalty="elasticnet", max_iter=1000), n_jobs=-1)

    if train_all:
        x = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\training-x2.csv").values
        y = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\training-y2.csv").values
        model.fit(x, y)

    pickle.dump(model, open(modelfile, "wb"))


def testModel():
    fold = 15
    kf = KFold(n_splits=fold, shuffle=True)

    if os.path.exists(modelfile):
        model = pickle.load(open(modelfile, 'rb'))
    else:
        genModel(False)
        model = pickle.load(open(modelfile, 'rb'))

    correct = [0] * fold
    incorrect = [0] * fold
    i = 0

    for train_ind, test_ind in kf.split(x):
        x_train, x_test = x[train_ind], x[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        # print(y_test, "\n", y_predict)

        for j in range(len(y_predict)):
            # print("Ys\n", y_test[j], "\n", y_predict[j])

            if np.argmax(y_predict[j]) == np.argmax(y_test[j]):
                correct[i] += 1
            else:
                incorrect[i] += 1
        print("Correct %: {} of {}".format(correct[i] / (correct[i] + incorrect[i]), (correct[i] + incorrect[i])))
        i += 1



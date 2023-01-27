import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pickle
import joblib
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("maas.csv")
df.info()

# Bağımlı değişkeni dışarıda bırakıp geride kalan tüm (bağımsız) değişkenleri bir df'e atayalım
X = df.drop(["SALARY_AVG_TL"], axis=1)

# Bağımlı değişkenimizi ayrı bir df'e atayalım
y = df[["SALARY_AVG_TL"]].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

gbr_model = GradientBoostingRegressor()
gbr_model.fit(X_train, y_train)

np.mean((-1) * cross_val_score(gbr_model,
                               X_test,
                               y_test,
                               cv=5,
                               scoring='neg_root_mean_squared_error'))


np.mean(cross_val_score(gbr_model,
                        X_test,              # Bağımsız değişkenler
                        y_test,              # Bağımlı değişken
                        cv=5,
                        scoring='r2'))


best_hipars = {'learning_rate': 0.05,
               'max_depth': 3,
               'min_samples_split': 0.5,
               'n_estimators': 750,
               'subsample': 0.7}


gbr_final = gbr_model.set_params(**best_hipars).fit(X, y)

gbr_final.get_params()


file = open('g_final', 'wb')
pickle.dump(gbr_final, file)

file.close()
#joblib.dump(gbr_final, "gbr_final.pkl")
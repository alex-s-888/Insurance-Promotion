import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../dataset/InsurancePolicyHistoricalData.csv')

# Clean-up, convert categorical features
df = df.drop(['Unnamed: 0'], axis=1)
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])

# Splitting the data into Train and Test, etc
X = df[['AnnualIncome', 'FamilyMembers', 'Age']]
y = df['InsurancePolicy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Hyperparameters tuning - using GridSearchCV to iterate through parameters and find the best
parameters = {
    "n_estimators": [1, 10, 25, 50],
    "max_depth": [1, 3, 5],
    "learning_rate": [0.01, 0.05, 0.1]
}

gbc = GradientBoostingClassifier()
cv = GridSearchCV(gbc, parameters, cv=5)
cv.fit(X_train, y_train.values.ravel())

# print(cv.score(X_train, y_train))
# print(cv.best_params_)
best_params = cv.best_params_

# Build model using best_params
gbc = GradientBoostingClassifier(
    learning_rate=float(best_params['learning_rate']),
    max_depth=int(best_params['max_depth']),
    n_estimators=int(best_params['n_estimators'])
)
gbc.fit(X_train, y_train)

# Serialize the model
dump(gbc, '../deployment/my_model.joblib')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wczytanie danych
data = pd.read_csv('winequality.csv', sep=';')

# Podział na cechy i etykiety
X = data.drop('quality', axis=1)
y = data['quality']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utworzenie modelu klasyfikatora RandomForest
model = RandomForestClassifier()

# Trenowanie modelu
model.fit(X_train, y_train)

# Ocena modelu na zbiorze testowym
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Przykładowa predykcja na nowych danych
new_data = pd.DataFrame({
    'fixed acidity': [7.2, 6.3, 8.1],
    'volatile acidity': [0.35, 0.31, 0.28],
    'citric acid': [0.26, 0.48, 0.4],
    'residual sugar': [9.6, 1.8, 6.9],
    'chlorides': [0.049, 0.057, 0.05],
    'free sulfur dioxide': [30, 14, 26],
    'total sulfur dioxide': [114, 132, 133],
    'density': [0.995, 0.9935, 0.9951],
    'pH': [3.3, 3.5, 3.2],
    'sulphates': [0.6, 0.55, 0.75],
    'alcohol': [10.4, 11.3, 9.6]
})

predictions = model.predict(new_data)
print('Predictions:')
for prediction in predictions:
    print(prediction)
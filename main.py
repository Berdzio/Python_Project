import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class WineQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wine Quality App")

        # Wczytanie danych
        self.data = pd.read_csv('winequality.csv', sep=';')

        # Podział na cechy i etykiety
        self.X = self.data.drop('quality', axis=1)
        self.y = self.data['quality']

        # Podział danych na zbiór treningowy i testowy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)

        # Utworzenie pustego modelu
        self.model = None

        # Tworzenie elementów interfejsu użytkownika
        self.create_widgets()

    def create_widgets(self):
        # Przycisk trenowania modelu
        self.train_button = tk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        # Przycisk testowania modelu
        self.test_button = tk.Button(self.root, text="Test Model", command=self.test_model)
        self.test_button.pack()

        # Przycisk predykcji na nowych danych
        self.predict_button = tk.Button(self.root, text="Predict New Data", command=self.predict_new_data)
        self.predict_button.pack()

        # Przycisk dodawania nowych danych
        self.add_data_button = tk.Button(self.root, text="Add New Data", command=self.add_new_data)
        self.add_data_button.pack()

        # Przycisk ponownego budowania modelu
        self.rebuild_button = tk.Button(self.root, text="Rebuild Model", command=self.rebuild_model)
        self.rebuild_button.pack()

        # Przycisk wyświetlenia tabeli
        self.display_table = tk.Button(self.root, text="Display Table", command=self.display_table)
        self.display_table.pack()

        #wyniki
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def train_model(self):
        # klasyfikator RandomForest
        self.model = RandomForestClassifier()

        self.model.fit(self.X_train, self.y_train)

        self.result_label.config(text="Model trained successfully!")

    def test_model(self):
        if self.model is not None:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.result_label.config(text=f"Model accuracy: {accuracy}")
        else:
            self.result_label.config(text="Model not trained!")

    def predict_new_data(self):
        if self.model is not None:
            # Pobranie wartości z pól wprowadzania nowych danych
            new_data = {
                'fixed acidity': entries[0].get(),
                'volatile acidity': entries[1].get(),
                'citric acid': entries[2].get(),
                'residual sugar': entries[3].get(),
                'chlorides': entries[4].get(),
                'free sulfur dioxide': entries[5].get(),
                'total sulfur dioxide': entries[6].get(),
                'density': entries[7].get(),
                'pH': entries[8].get(),
                'sulphates': entries[9].get(),
                'alcohol': entries[10].get()
            }

            new_data_df = pd.DataFrame([new_data])

            prediction = self.model.predict(new_data_df)

            self.result_label.config(text=f"Predicted wine quality: {prediction[0]}")
        else:
            self.result_label.config(text="Model not trained!")

    def add_new_data(self):
        # TODO: Implementacja dodawania nowych danych
        self.result_label.config(text="TODO!")

    def rebuild_model(self):
        self.train_model()
        self.result_label.config(text="Model rebuilt successfully!")

    def display_table(self):
        table_window = tk.Toplevel(self.root)
        table_window.title("Tabela danych")

        table = ttk.Treeview(table_window)
        table.pack()

        columns = self.data.columns.tolist()
        table['columns'] = columns
        table.heading('#0', text='Index')
        for col in columns:
            table.heading(col, text=col)

        # Dodawanie danych do tabeli
        for i, row in self.data.iterrows():
            table.insert('', 'end', text=i, values=tuple(row))

root = tk.Tk()
app = WineQualityApp(root)

input_frame = tk.Frame(root)
input_frame.pack()

labels = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar', 'Chlorides',
          'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol']

entries = []
for i, label in enumerate(labels):
    print(i)
    print(label)
    tk.Label(input_frame, text=label).grid(row=i, column=0)
    entry = tk.Entry(input_frame)
    entry.grid(row=i, column=1)
    entries.append(entry)
print(entries)

clear_button = tk.Button(root, text="Clear", command=lambda: [entry.delete(0, tk.END) for entry in entries])
clear_button.pack()

root.mainloop()
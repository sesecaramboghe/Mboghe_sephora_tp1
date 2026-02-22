import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Charger les données
df = pd.read_csv("auto-mpg.csv")  # assure-toi que le fichier existe

# Supprimer les valeurs manquantes
df = df.dropna()

# Variables explicatives
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year']]

# Variable cible
y = df['mpg']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle
with open("model_mpg.pkl", "wb") as f:
    pickle.dump(model, f)

# Sauvegarder le scaler
with open("scaler_mpg.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Modèle et scaler sauvegardés avec succès !")

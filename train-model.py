import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import SimpleImputer

# =====================================================
# 1️⃣ CENSUS MODEL
# =====================================================

census = pd.read_csv("adult.csv")

census = census.dropna()

X_census = census[["age", "education.num", "hours.per.week"]]
y_census = census["income"]

# Convertir income en numérique
y_census = y_census.apply(lambda x: 1 if ">50K" in str(x) else 0)

scaler_census = StandardScaler()
X_census_scaled = scaler_census.fit_transform(X_census)

model_census = LogisticRegression()
model_census.fit(X_census_scaled, y_census)

# Sauvegardes census
pickle.dump(model_census, open("census.pkl", "wb"))
pickle.dump(scaler_census, open("scaler_census.pkl", "wb"))
pickle.dump(list(X_census.columns), open("census_columns.pkl", "wb"))

print("✅ Census model sauvegardé")


# =====================================================
# 2️⃣ MPG MODEL
# =====================================================

mpg = pd.read_csv("auto-mpg.csv")

mpg.replace("?", pd.NA, inplace=True)
mpg["horsepower"] = pd.to_numeric(mpg["horsepower"], errors="coerce")

X_mpg = mpg[["cylinders", "displacement", "horsepower",
             "weight", "acceleration", "model-year" ]]
y_mpg = mpg["mpg"]

imputer = SimpleImputer(strategy="mean")
X_mpg_imputed = imputer.fit_transform(X_mpg)

scaler_mpg = StandardScaler()
X_mpg_scaled = scaler_mpg.fit_transform(X_mpg_imputed)

model_mpg = LinearRegression()
model_mpg.fit(X_mpg_scaled, y_mpg)

# Sauvegardes mpg
pickle.dump(model_mpg, open("auto-mpg.pkl", "wb"))
pickle.dump(scaler_mpg, open("scaler_mpg.pkl", "wb"))
pickle.dump(imputer, open("imputer_mpg.pkl", "wb"))

print("✅ MPG model sauvegardé")
print("🎉 Tous les fichiers sont prêts !")

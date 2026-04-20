import streamlit as st
import pandas as pd

# -------------------------------
# DEBUG: Check sklearn installation
# -------------------------------
try:
    import sklearn
    st.success(f"✅ scikit-learn installed: {sklearn.__version__}")
except Exception as e:
    st.error(f"❌ scikit-learn NOT installed: {e}")
    st.stop()

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(
    "https://raw.githubusercontent.com/jiritkhanpranali-cmyk/Salary_Pred/refs/heads/main/Salary%20Data.csv"
)

st.title("💰 Salary Prediction App")

st.subheader("Raw Data")
st.write(df.head())

# -------------------------------
# Missing value handling
# -------------------------------
for col in ['Age', 'Years of Experience', 'Salary']:
    df[col] = df[col].fillna(df[col].mean())

for col in ['Gender', 'Education Level', 'Job Title']:
    df[col] = df[col].fillna(df[col].mode()[0])

st.subheader("After Missing Value Handling")
st.write(df.head())

# -------------------------------
# Encoding categorical columns
# -------------------------------
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for column in ['Gender', 'Education Level', 'Job Title']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# -------------------------------
# Split features and target
# -------------------------------
X = df.drop("Salary", axis=1)
y = df["Salary"]

# -------------------------------
# Train-test split
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Models
# -------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR(kernel="linear"),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

st.subheader("Model Training Results")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results[name] = {"MAE": mae, "R2": r2}

    st.write(f"### {name}")
    st.write(f"MAE: {mae:.2f}")
    st.write(f"R2 Score: {r2:.2f}")

# -------------------------------
# Best model selection
# -------------------------------
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]

st.success(f"🏆 Best Model: {best_model_name}")

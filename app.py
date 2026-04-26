
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border-left: 4px solid #4fc3f7;
        box-shadow: 0 4px 15px rgba(79,195,247,0.15);
    }
    .metric-card h3 { color: #4fc3f7; font-size: 14px; margin: 0; }
    .metric-card p  { color: #ffffff; font-size: 28px; font-weight: 700; margin: 4px 0 0; }
    .section-title {
        color: #4fc3f7;
        font-size: 22px;
        font-weight: 700;
        border-bottom: 2px solid #4fc3f7;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .step-card {
        background: #1a2535;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid #26c6da;
    }
    .step-card h4 { color: #26c6da; margin: 0 0 4px; }
    .step-card p  { color: #cfd8dc; margin: 0; font-size: 14px; }
    .predict-box {
        background: linear-gradient(135deg, #1b2838, #0f1923);
        border: 1px solid #4fc3f7;
        border-radius: 16px;
        padding: 30px;
    }
    .price-result {
        background: linear-gradient(90deg, #1565c0, #0288d1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 16px;
    }
    .price-result h2 { color: #fff; font-size: 36px; margin: 0; }
    .price-result p  { color: #b3e5fc; margin: 4px 0 0; font-size: 14px; }
    div[data-testid="stSidebar"] { background-color: #0d1b2a; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Ford Car Price\n### Prediction Dashboard")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload `ford.csv`", type=["csv"])
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio("", ["📊 EDA & Insights", "🤖 Model & Metrics", "🔮 Predict Price", "📋 Step-by-Step Guide"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built with Streamlit · Linear Regression · Ford Dataset")

# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_model(df):
    x = df.drop(columns=["price"])
    y = df["price"]
    X_enc = pd.get_dummies(x, columns=["model", "transmission", "fuelType"], drop_first=True).astype(int)
    numerical_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
    scaler = StandardScaler()
    X_enc[numerical_cols] = scaler.fit_transform(X_enc[numerical_cols])
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.33, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    n, p = X_test.shape
    adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, scaler, X_enc.columns.tolist(), r2, adj_r2, mae, rmse, y_test, y_pred, df

if uploaded:
    df = load_data(uploaded)
    model, scaler, feature_cols, r2, adj_r2, mae, rmse, y_test, y_pred, df = train_model(df)
    data_ready = True
else:
    data_ready = False

# ─── Helper ──────────────────────────────────────────────────────────────────
def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0e1117")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161c27")
    ax.tick_params(colors="#cfd8dc")
    ax.xaxis.label.set_color("#4fc3f7")
    ax.yaxis.label.set_color("#4fc3f7")
    ax.title.set_color("#ffffff")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a50")
    return fig, ax

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 – EDA & Insights
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 EDA & Insights":
    st.markdown("<div class='section-title'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)

    if not data_ready:
        st.info("👈 Please upload `ford.csv` from the sidebar to begin.")
        st.markdown("""
        **Expected columns:** `model`, `year`, `price`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`
        
        Download the dataset from Kaggle: [Ford Car Price Dataset](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction)
        """)
    else:
        # Overview cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><h3>Total Records</h3><p>{df.shape[0]:,}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h3>Features</h3><p>{df.shape[1]}</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h3>Avg Price</h3><p>£{df['price'].mean():,.0f}</p></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><h3>Missing Values</h3><p>{df.isnull().sum().sum()}</p></div>", unsafe_allow_html=True)

        st.markdown("---")

        # Dataset preview
        with st.expander("🔍 Preview Dataset", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)

        with st.expander("📈 Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-title'>📉 Visual Insights</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price Distribution**")
            fig, ax = dark_fig()
            sns.histplot(df["price"], bins=50, kde=True, ax=ax, color="#4fc3f7")
            ax.set_xlabel("Price (£)")
            ax.set_title("Car Price Distribution")
            fig_to_st(fig)

        with col2:
            st.markdown("**Price by Fuel Type**")
            fig, ax = dark_fig()
            sns.boxplot(data=df, x="fuelType", y="price", ax=ax, palette="Set2")
            ax.set_title("Price by Fuel Type")
            fig_to_st(fig)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Mileage vs Price**")
            fig, ax = dark_fig()
            ax.scatter(df["mileage"], df["price"], alpha=0.3, color="#4fc3f7", s=10)
            ax.set_xlabel("Mileage")
            ax.set_ylabel("Price (£)")
            ax.set_title("Mileage vs Price")
            fig_to_st(fig)

        with col4:
            st.markdown("**Price by Transmission**")
            fig, ax = dark_fig()
            sns.boxplot(data=df, x="transmission", y="price", ax=ax, palette="coolwarm")
            ax.set_title("Price by Transmission Type")
            fig_to_st(fig)

        st.markdown("**Year vs Price**")
        fig, ax = dark_fig(14, 5)
        sns.boxplot(data=df, x="year", y="price", ax=ax, palette="Blues")
        plt.xticks(rotation=45)
        ax.set_title("Car Year vs Price")
        fig_to_st(fig)

        st.markdown("**Correlation Heatmap**")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#161c27")
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, linecolor="#0e1117",
                    annot_kws={"size": 10, "color": "white"})
        ax.tick_params(colors="#cfd8dc")
        ax.set_title("Feature Correlation Matrix", color="#ffffff", pad=12)
        fig_to_st(fig)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Model & Metrics
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model & Metrics":
    st.markdown("<div class='section-title'>🤖 Model Performance</div>", unsafe_allow_html=True)

    if not data_ready:
        st.info("👈 Please upload `ford.csv` from the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"<div class='metric-card'><h3>R² Score</h3><p>{r2:.4f}</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h3>Adjusted R²</h3><p>{adj_r2:.4f}</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h3>MAE</h3><p>£{mae:,.0f}</p></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-card'><h3>RMSE</h3><p>£{rmse:,.0f}</p></div>", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Actual vs Predicted Prices**")
            fig, ax = dark_fig()
            ax.scatter(y_test, y_pred, alpha=0.4, color="#4fc3f7", s=12)
            mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], color="#ff7043", linewidth=2, label="Perfect fit")
            ax.set_xlabel("Actual Price (£)")
            ax.set_ylabel("Predicted Price (£)")
            ax.set_title("Actual vs Predicted")
            ax.legend(labelcolor="#cfd8dc", facecolor="#161c27")
            fig_to_st(fig)

        with col2:
            st.markdown("**Residual Distribution**")
            residuals = np.array(y_test) - y_pred
            fig, ax = dark_fig()
            sns.histplot(residuals, bins=50, kde=True, ax=ax, color="#ab47bc")
            ax.axvline(0, color="#ff7043", linewidth=2, linestyle="--")
            ax.set_xlabel("Residual (£)")
            ax.set_title("Residuals Distribution")
            fig_to_st(fig)

        st.markdown("**Top Feature Importances (Coefficients)**")
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": model.coef_
        }).reindex(pd.Index(range(len(feature_cols))))
        coef_df["Abs"] = coef_df["Coefficient"].abs()
        top20 = coef_df.nlargest(20, "Abs")

        fig, ax = dark_fig(12, 6)
        colors = ["#4fc3f7" if c > 0 else "#ef5350" for c in top20["Coefficient"]]
        ax.barh(top20["Feature"], top20["Coefficient"], color=colors)
        ax.set_title("Top 20 Feature Coefficients")
        ax.set_xlabel("Coefficient Value")
        ax.invert_yaxis()
        fig_to_st(fig)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Predict Price
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown("<div class='section-title'>🔮 Predict Car Price</div>", unsafe_allow_html=True)

    if not data_ready:
        st.info("👈 Please upload `ford.csv` from the sidebar to enable predictions.")
    else:
        st.markdown("<div class='predict-box'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            sel_model = st.selectbox("🚘 Model", sorted(df["model"].unique()))
            year = st.slider("📅 Year", int(df["year"].min()), int(df["year"].max()), 2017)
            transmission = st.selectbox("⚙️ Transmission", df["transmission"].unique())

        with col2:
            mileage = st.number_input("📏 Mileage (miles)", 0, 200000, 30000, step=1000)
            fuelType = st.selectbox("⛽ Fuel Type", df["fuelType"].unique())
            tax = st.number_input("💷 Road Tax (£)", 0, 600, 145, step=5)

        with col3:
            mpg = st.number_input("🛢️ MPG", 10.0, 200.0, 50.0, step=1.0)
            engineSize = st.selectbox("🔧 Engine Size (L)", sorted(df["engineSize"].unique()))

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔮 Predict Price", use_container_width=True):
            input_df = pd.DataFrame([{
                "model": sel_model,
                "year": year,
                "transmission": transmission,
                "mileage": mileage,
                "fuelType": fuelType,
                "tax": tax,
                "mpg": mpg,
                "engineSize": engineSize,
            }])

            # One-hot encode same as training
            all_data = pd.concat([df.drop(columns=["price"]), input_df], ignore_index=True)
            all_enc = pd.get_dummies(all_data, columns=["model", "transmission", "fuelType"], drop_first=True).astype(int)

            # Align columns with training
            input_enc = all_enc.iloc[[-1]].reindex(columns=feature_cols, fill_value=0)

            # Scale
            numerical_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
            train_num = pd.DataFrame(df[numerical_cols])
            sc = StandardScaler().fit(train_num)
            input_enc[numerical_cols] = sc.transform(input_enc[numerical_cols])

            pred = model.predict(input_enc)[0]

            st.markdown(f"""
            <div class='price-result'>
                <p>Estimated Market Value</p>
                <h2>£{pred:,.0f}</h2>
                <p>{year} Ford {sel_model} · {transmission} · {fuelType} · {engineSize}L · {mileage:,} miles</p>
            </div>
            """, unsafe_allow_html=True)

            # Show similar cars
            st.markdown("---")
            st.markdown("**Similar Cars in Dataset**")
            similar = df[
                (df["model"] == sel_model) &
                (df["year"].between(year - 1, year + 1)) &
                (df["transmission"] == transmission)
            ][["model", "year", "mileage", "fuelType", "engineSize", "price"]].head(10)
            if not similar.empty:
                st.dataframe(similar, use_container_width=True)
            else:
                st.info("No exact matches found for comparison.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Step-by-Step Guide
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Step-by-Step Guide":
    st.markdown("<div class='section-title'>📋 Complete Project Guide</div>", unsafe_allow_html=True)

    steps = [
        ("✅ Step 1 — Data Collection", "DONE",
         "Download the Ford Car Price dataset from Kaggle. It contains model, year, price, transmission, mileage, fuelType, tax, mpg, and engineSize columns."),
        ("✅ Step 2 — Data Loading & Inspection", "DONE",
         "Load with pd.read_csv(). Inspect with df.head(), df.info(), df.describe(), df.isnull().sum()."),
        ("✅ Step 3 — Exploratory Data Analysis (EDA)", "DONE",
         "Visualize price distribution, correlations, boxplots by transmission/fuelType/year, scatter of mileage vs price."),
        ("✅ Step 4 — Encoding Categorical Variables", "DONE",
         "Applied One-Hot Encoding (pd.get_dummies) and Label Encoding on model, transmission, fuelType columns."),
        ("✅ Step 5 — Feature Scaling", "DONE",
         "StandardScaler applied to numerical columns: year, mileage, tax, mpg, engineSize."),
        ("✅ Step 6 — Train-Test Split", "DONE",
         "80/67% train, 33% test split with train_test_split (random_state=42)."),
        ("✅ Step 7 — Model Training", "DONE",
         "Linear Regression fitted on one-hot encoded features. Also tested with label-encoded features."),
        ("✅ Step 8 — Model Evaluation", "DONE",
         "R² Score, Adjusted R², MAE, and RMSE computed on test set."),
        ("🔲 Step 9 — Outlier Detection & Removal", "PENDING",
         "Use IQR or Z-score to remove price/mileage outliers. This can significantly improve R² from ~0.67 to 0.80+."),
        ("🔲 Step 10 — Try Regularization", "PENDING",
         "Test Ridge and Lasso Regression to reduce overfitting and handle multicollinearity in one-hot features."),
        ("🔲 Step 11 — Try Advanced Models", "PENDING",
         "Compare Random Forest, Gradient Boosting (XGBoost/LightGBM), and Decision Tree Regressors for higher accuracy."),
        ("🔲 Step 12 — Hyperparameter Tuning", "PENDING",
         "Use GridSearchCV or RandomizedSearchCV to optimize model parameters for best performance."),
        ("🔲 Step 13 — Cross-Validation", "PENDING",
         "Apply k-fold cross-validation (k=5 or 10) to get a robust, unbiased model performance estimate."),
        ("🔲 Step 14 — Feature Engineering", "PENDING",
         "Create new features: car_age = 2024 - year, price_per_mpg, mileage_per_year. Can improve model signal."),
        ("🔲 Step 15 — Model Saving & Deployment", "PENDING",
         "Save the final model with joblib/pickle. Deploy as an API using FastAPI or Flask, or via Streamlit (this dashboard!)."),
    ]

    for title, status, desc in steps:
        color = "#26c6da" if status == "DONE" else "#ffa726"
        st.markdown(f"""
        <div class='step-card' style='border-left-color:{color};'>
            <h4>{title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    cols = st.columns(5)
    techs = [("🐍 Python", "Core language"), ("🐼 Pandas", "Data manipulation"),
             ("📊 Seaborn", "Visualization"), ("🤖 Scikit-learn", "ML models"),
             ("🌐 Streamlit", "Dashboard UI")]
    for col, (name, desc) in zip(cols, techs):
        with col:
            st.markdown(f"<div class='metric-card'><h3>{name}</h3><p style='font-size:14px;color:#90caf9'>{desc}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 How to Run Locally")
    st.code("""# 1. Install dependencies
pip install streamlit pandas numpy scikit-learn seaborn matplotlib

# 2. Run the dashboard
streamlit run app.py

# 3. Upload ford.csv in the sidebar
""", language="bash")


import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Car Price Lab",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Car Price Lab")
st.caption("Vietnam Car Price Prediction using Linear Regression")


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

CSV_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "car-price.csv"


@st.cache_data
def load_dataset(file_path):
    return pd.read_csv(file_path)


# --------------------------------------------------
# 2. Convert Vietnamese price to million VND
# --------------------------------------------------

def convert_price(price):
    if pd.isna(price):
        return np.nan

    price = str(price).strip()

    ty = re.search(r"([\d.,]+)\s*Tỷ", price, re.IGNORECASE)
    trieu = re.search(r"([\d.,]+)\s*Triệu", price, re.IGNORECASE)

    ty_value = float(ty.group(1).replace(",", ".")) if ty else 0
    trieu_value = float(trieu.group(1).replace(",", ".")) if trieu else 0

    return ty_value * 1000 + trieu_value


try:
    dataset = load_dataset(str(CSV_PATH))
except FileNotFoundError:
    st.error(f"Không tìm thấy dataset: {CSV_PATH}")
    st.info("Kiểm tra lại vị trí file car-price.csv.")
    st.stop()


if "Price" not in dataset.columns:
    st.error("Dataset không có cột Price.")
    st.stop()

dataset["Price_Million"] = dataset["Price"].apply(convert_price)


# --------------------------------------------------
# 3. Sidebar: dataset information
# --------------------------------------------------

with st.sidebar:
    st.header("Dataset")
    st.write(f"**Rows:** {len(dataset):,}")
    st.write(f"**Columns:** {len(dataset.columns)}")
    st.write(f"**File:** car-price.csv")

    st.divider()
    st.caption("Target mặc định: Price_Million (triệu VNĐ)")


# --------------------------------------------------
# 4. Preview dataset
# --------------------------------------------------

with st.expander("Preview dataset", expanded=False):
    st.dataframe(dataset.head(100), width='stretch')


# --------------------------------------------------
# 5. Select X and Y
# --------------------------------------------------

st.subheader("1. Configure Model")

# Exclude columns that are not suitable as simple numeric features.
excluded_columns = {
    "Description",
    "Link",
    "Car_code",
}

available_columns = [
    column
    for column in dataset.columns
    if column not in excluded_columns
    and dataset[column].notna().any()
]

default_x = (
    "Năm sản xuất"
    if "Năm sản xuất" in available_columns
    else available_columns[0]
)

default_y = "Price_Million"

col1, col2 = st.columns(2)

with col1:
    x_column = st.selectbox(
        "Input Feature (X)",
        options=available_columns,
        index=available_columns.index(default_x),
    )

with col2:
    y_column = st.selectbox(
        "Target (Y)",
        options=available_columns,
        index=available_columns.index(default_y),
    )

if x_column == y_column:
    st.warning("X và Y phải là hai cột khác nhau.")
    st.stop()


# --------------------------------------------------
# 6. Prepare numeric data
# --------------------------------------------------

def convert_feature_to_numeric(series):
    """Convert numeric columns and common car attributes to numbers."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype("string").str.strip()

    # Extract numeric values from examples like:
    # "38,000 Km", "5 chỗ", "4 cửa", "Xăng 2.0 L"
    cleaned = text.str.replace(",", "", regex=False)

    extracted = cleaned.str.extract(
        r"(-?\d+(?:\.\d+)?)",
        expand=False,
    )

    return pd.to_numeric(extracted, errors="coerce")


work = pd.DataFrame({
    "X": convert_feature_to_numeric(dataset[x_column]),
    "Y": convert_feature_to_numeric(dataset[y_column]),
})

work = work.replace([np.inf, -np.inf], np.nan).dropna()

if len(work) < 2:
    st.error(
        "Không đủ dữ liệu số để train. "
        "Hãy chọn feature và target có thể chuyển thành số."
    )
    st.stop()

if work["X"].nunique() < 2:
    st.error("Input Feature X cần ít nhất hai giá trị khác nhau.")
    st.stop()


# --------------------------------------------------
# 7. Train Linear Regression
# --------------------------------------------------

st.subheader("2. Train Model")

if st.button("🚀 Train Linear Regression", type="primary"):
    x = work[["X"]].to_numpy()
    y = work["Y"].to_numpy()

    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)

    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    slope = model.coef_[0]
    intercept = model.intercept_
    sign = "+" if intercept >= 0 else "-"

    equation = (
        f"Y = {slope:,.4f} × X "
        f"{sign} {abs(intercept):,.4f}"
    )

    # Store trained model and data for prediction.
    st.session_state["model"] = model
    st.session_state["x_column"] = x_column
    st.session_state["y_column"] = y_column
    st.session_state["equation"] = equation
    st.session_state["rmse"] = rmse
    st.session_state["r2"] = r2
    st.session_state["work"] = work.copy()
    st.session_state["x_min"] = float(work["X"].min())
    st.session_state["x_max"] = float(work["X"].max())

    st.success("Model training completed!")


# --------------------------------------------------
# 8. Show results
# --------------------------------------------------

if (
    "model" in st.session_state
    and st.session_state["x_column"] == x_column
    and st.session_state["y_column"] == y_column
):
    model = st.session_state["model"]
    work = st.session_state["work"]

    st.subheader("3. Model Evaluation")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "RMSE",
        f"{st.session_state['rmse']:,.2f}",
        help="Sai số dự đoán, cùng đơn vị với Target Y.",
    )

    metric2.metric(
        "R² Score",
        f"{st.session_state['r2']:.4f}",
        help="Đánh giá mức độ phù hợp của mô hình trên dữ liệu train.",
    )

    metric3.metric("Training rows", f"{len(work):,}")

    st.code(st.session_state["equation"])

    # ----------------------------------------------
    # 9. Visualization
    # ----------------------------------------------

    st.subheader("4. Visualization")

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.scatter(
        work["X"],
        work["Y"],
        color="green",
        alpha=0.65,
        label="Actual data",
    )

    x_line = np.linspace(
        work["X"].min(),
        work["X"].max(),
        200,
    ).reshape(-1, 1)

    y_line = model.predict(x_line)

    ax.plot(
        x_line.flatten(),
        y_line,
        color="black",
        linewidth=2,
        label="Linear Regression",
    )

    ax.set_title(
        f"{y_column} vs {x_column}",
    )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ----------------------------------------------
    # 10. Prediction
    # ----------------------------------------------

    st.subheader("5. Predict")

    x_min = st.session_state["x_min"]
    x_max = st.session_state["x_max"]

    default_value = min(max(2022.0, x_min), x_max)

    input_x = st.number_input(
        f"Enter {x_column}",
        min_value=x_min,
        max_value=x_max,
        value=default_value,
        step=1.0,
    )

    if st.button("Predict"):
        predicted_y = model.predict([[input_x]])[0]

        st.metric(
            f"Predicted {y_column}",
            f"{predicted_y:,.2f}",
        )

        st.caption(
            "Prediction được giới hạn trong khoảng giá trị X "
            "có trong dataset."
        )

else:
    st.info(
        "Chọn Input Feature và Target, "
        "sau đó nhấn Train Linear Regression."
    )


import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# Configuration
st.set_page_config(
    page_title="Car AI Chatbot",
    page_icon="🚗",
    layout="centered",
)

DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "Dataset"
    / "car-price.csv"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Maximum number of cars to rank using Transformer
MAX_CANDIDATES = 500


# Convert Vietnamese car prices to million VND
def convert_price(price):
    if pd.isna(price):
        return np.nan

    price = str(price).strip()

    ty = re.search(
        r"([\d.,]+)\s*Tỷ", price, re.IGNORECASE
    )
    trieu = re.search(
        r"([\d.,]+)\s*Triệu", price, re.IGNORECASE
    )

    if not ty and not trieu:
        return np.nan

    ty_value = (
        float(ty.group(1).replace(",", "."))
        if ty else 0
    )

    trieu_value = (
        float(trieu.group(1).replace(",", "."))
        if trieu else 0
    )

    return ty_value * 1000 + trieu_value


# Extract budget from Vietnamese question
def extract_budget(question):
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*"
        r"(tỷ|triệu|tr|million)",
        question,
        re.IGNORECASE,
    )

    if not match:
        return None

    amount = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()

    if unit == "tỷ":
        return amount * 1000

    return amount


def extract_year(question):
    match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    return int(match.group(1)) if match else None


def extract_seats(question):
    match = re.search(
        r"\b(\d+)\s*(?:chỗ(?:\s+ngồi)?|ghế)\b",
        question,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def extract_brand(question, dataset):
    name_col = find_column(dataset, ["Name"])

    if not name_col:
        return None

    # Identify brands that appear in the dataset's car names.
    brands = [
        "Mercedes-Benz", "Mercedes Benz", "Mitsubishi",
        "Volkswagen", "Land Rover", "VinFast",
        "Toyota", "Hyundai", "Honda", "Kia", "Mazda",
        "Ford", "BMW", "Audi", "Lexus", "Porsche",
        "Peugeot", "Volvo", "Subaru", "Isuzu",
        "Suzuki", "Nissan", "Chevrolet", "MG", "BYD",
    ]

    question_lower = question.casefold()

    for brand in sorted(brands, key=len, reverse=True):
        if brand.casefold() in question_lower:
            return brand

    return None


def extract_body_type(question):
    body_types = [
        "SUV", "Sedan", "MPV", "Hatchback",
        "Pickup", "Bán tải", "Crossover",
    ]

    question_lower = question.casefold()

    for body_type in body_types:
        if body_type.casefold() in question_lower:
            return body_type

    return None


def extract_price_condition(question):
    """Return (price, mode), where price is in million VND."""
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(tỷ|triệu|tr|million)",
        question,
        re.IGNORECASE,
    )

    if not match:
        return None, None

    amount = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    price = amount * 1000 if unit == "tỷ" else amount

    before = question[:match.start()].casefold()

    if re.search(r"(trên|hơn|ít nhất|từ)\s*$", before):
        mode = "min"
    else:
        # Default: budget means maximum price.
        mode = "max"

    return price, mode

# Find a column using common Vietnamese names
def find_column(df, names):
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for name in names:
        key = name.lower()
        if key in normalized:
            return normalized[key]

    return None


# Read CSV dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    if "Price" not in df.columns:
        raise ValueError("Dataset must contain Price column")

    df["Price_Million"] = df["Price"].apply(
        convert_price
    )

    df = df.dropna(subset=["Price_Million"])
    df = df[df["Price_Million"] > 0].copy()

    return df


# Load pretrained Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


# Build searchable description from real CSV columns
def build_description(row, columns):
    parts = []

    for col in columns:
        value = row[col]

        if pd.notna(value):
            text = str(value).strip()

            if text:
                parts.append(f"{col}: {text}")

    return ". ".join(parts)


# Find suitable cars

def recommend_cars(question, dataset, model):
    price, price_mode = extract_price_condition(question)
    year = extract_year(question)
    seats = extract_seats(question)
    brand = extract_brand(question, dataset)
    body_type = extract_body_type(question)

    candidates = dataset.copy()

    # 1. Filter by price
    if price is not None:
        if price_mode == "min":
            candidates = candidates[
                candidates["Price_Million"] >= price
            ].copy()
        else:
            candidates = candidates[
                candidates["Price_Million"] <= price
            ].copy()

    # 2. Filter by exact manufacturing year
    year_col = find_column(
        candidates, ["Năm sản xuất"]
    )

    if year is not None and year_col:
        year_values = pd.to_numeric(
            candidates[year_col], errors="coerce"
        )
        candidates = candidates[
            year_values == year
        ].copy()

    # 3. Filter by exact number of seats
    seats_col = find_column(
        candidates, ["Số chỗ ngồi", "Số chỗ"]
    )

    if seats is not None and seats_col:
        seat_values = pd.to_numeric(
            candidates[seats_col]
            .astype(str)
            .str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        candidates = candidates[
            seat_values == seats
        ].copy()

    # 4. Filter by car brand
    name_col = find_column(candidates, ["Name"])

    if brand and name_col:
        candidates = candidates[
            candidates[name_col].astype(str).str.contains(
                re.escape(brand),
                case=False,
                na=False,
            )
        ].copy()

    # 5. Filter by body type
    body_col = find_column(candidates, ["Kiểu dáng"])

    if body_type and body_col:
        candidates = candidates[
            candidates[body_col].astype(str).str.contains(
                re.escape(body_type),
                case=False,
                na=False,
            )
        ].copy()

    if candidates.empty:
        return price, price_mode, pd.DataFrame()

    # Use the real columns from the CSV.
    car_columns = [
        "Name",
        "Năm sản xuất",
        "Số chỗ ngồi",
        "Kiểu dáng",
        "Số Km đã đi",
        "Tình trạng",
        "Xuất xứ",
        "Động cơ",
        "Màu ngoại thất",
        "Màu nội thất",
        "Description",
    ]

    available_columns = []

    for name in car_columns:
        col = find_column(candidates, [name])
        if col and col not in available_columns:
            available_columns.append(col)

    if not available_columns:
        return price, price_mode, pd.DataFrame()

    # Prefer newer cars when reducing the candidate list.
    year_col = find_column(candidates, ["Năm sản xuất"])

    if year_col:
        candidates["_Year"] = pd.to_numeric(
            candidates[year_col], errors="coerce"
        )
        candidates = candidates.sort_values(
            "_Year", ascending=False
        )

    candidates = candidates.head(MAX_CANDIDATES).copy()

    candidates["Description"] = candidates.apply(
        lambda row: build_description(
            row, available_columns
        ),
        axis=1,
    )

    candidates = candidates[
        candidates["Description"].str.len() > 0
    ].copy()

    if candidates.empty:
        return price, price_mode, pd.DataFrame()

    # Sentence Transformer embeddings
    question_embedding = model.encode(
        [question], normalize_embeddings=True
    )

    car_embeddings = model.encode(
        candidates["Description"].tolist(),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    # Cosine Similarity
    candidates["Similarity"] = (
        car_embeddings @ question_embedding[0]
    )

    candidates = candidates.sort_values(
        "Similarity", ascending=False
    )

    candidates = candidates.drop_duplicates(
        subset=["Name"] if "Name" in candidates.columns
        else ["Description"]
    )

    return price, price_mode, candidates.head(5)

# Display chatbot response

def show_recommendations(question, dataset, model):
    budget, price_mode, results = recommend_cars(
        question, dataset, model
    )

    if results.empty:
        if budget is not None:
            st.info(
                f"Không tìm thấy xe phù hợp với ngân sách "
                f"{budget:,.0f} triệu VND và các điều kiện đã nhập."
            )
        else:
            st.info(
                "Không tìm thấy xe phù hợp với các điều kiện "
                "bạn yêu cầu. Hãy thử thay đổi hãng xe hoặc năm sản xuất."
            )
        return

    if budget is not None:
        operator = "từ" if price_mode == "min" else "không vượt"
        st.success(
            f"Tìm thấy xe có giá {operator} "
            f"{budget:,.0f} triệu VND."
        )
    else:
        st.success("Đã tìm thấy các xe phù hợp với nhu cầu.")

    st.caption(
        "Các xe được xếp hạng theo độ tương đồng "
        "ngữ nghĩa với câu hỏi. Kết quả không đảm bảo "
        "đáp ứng mọi nhu cầu nếu dataset thiếu thông tin."
    )

    name_col = find_column(
        results,
        ["Name", "Tên xe", "Dòng xe", "Model", "Hãng xe"],
    )

    for index, (_, car) in enumerate(
        results.iterrows(), start=1
    ):
        name = (
            str(car[name_col])
            if name_col and pd.notna(car[name_col])
            else f"Xe #{index}"
        )

        st.markdown(f"### {index}. {name}")

        st.write(
            f"**Giá:** "
            f"{car['Price_Million']:,.0f} triệu VND"
        )

        st.write(
            f"**Similarity:** "
            f"{car['Similarity']:.4f}"
        )

        st.write(
            f"**Thông tin xe:** {car['Description']}"
        )

        st.divider()


# Streamlit interface
st.title("🚗 Car AI Chatbot")

st.caption(
    "Vietnam Car Recommendation using "
    "Sentence Transformer and Cosine Similarity"
)

st.write(
    "Nhập nhu cầu mua xe bằng tiếng Việt. "
    "Chatbot sẽ tìm các xe phù hợp trong dataset."
)

try:
    dataset = load_dataset()
    model = load_model()

    st.caption(
        f"Dataset: {len(dataset):,} xe có giá hợp lệ"
    )

    question = st.chat_input(
        "Ví dụ: Tìm xe Toyota 7 chỗ năm 2020 dưới 600 triệu..."
    )

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm xe phù hợp..."):
                show_recommendations(
                    question, dataset, model
                )

except Exception as error:
    st.error(f"Application error: {error}")

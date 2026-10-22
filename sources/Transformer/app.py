import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sentence_transformers import SentenceTransformer

import hashlib
import json
from PIL import Image

st.set_page_config(page_title="Car Chatbot", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    .block-container {
        max-width: 75%;
        margin: 0 auto;
    }

    [data-testid="stBottom"] {
        width: 75%;
        left: 12.5%;
        right: auto;
    }
</style>
""", unsafe_allow_html=True)

DATASET_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "car-price.csv"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CNN image recognition
CNN_DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"
CNN_TRAIN_DIR = CNN_DATASET_DIR / "Train"
CNN_TEST_DIR = CNN_DATASET_DIR / "Test"

CNN_MODEL_DIR = Path(__file__).resolve().parent / "models"
CNN_MODEL_PATH = CNN_MODEL_DIR / "car_classifier.keras"
CNN_METADATA_PATH = CNN_MODEL_DIR / "metadata.json"

CNN_IMAGE_SIZE = (180, 180)
CNN_BATCH_SIZE = 32
CNN_EPOCHS = 20


def convert_price(value):
    # Convert values such as '1,2 Tỷ 500 Triệu' to triệu VND
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("₫", "")
    ty = re.search(r"([\d.,]+)\s*tỷ", text, re.IGNORECASE)
    trieu = re.search(r"([\d.,]+)\s*triệu", text, re.IGNORECASE)
    if not ty and not trieu:
        # Allow CSV values that are already numeric.
        try:
            return float(text.replace(",", ""))
        except ValueError:
            return np.nan
    ty_value = float(ty.group(1).replace(".", "").replace(",", ".")) if ty else 0
    trieu_value = float(trieu.group(1).replace(".", "").replace(",", ".")) if trieu else 0
    return ty_value * 1000 + trieu_value


def find_column(df, names):
    columns = {str(column).strip().casefold(): column for column in df.columns}
    for name in names:
        if name.casefold() in columns:
            return columns[name.casefold()]
    return None


def price_to_million(amount, unit):
    return float(amount.replace(",", ".")) * (1000 if unit and unit.casefold() == "tỷ" else 1)


def extract_price_filter(question):
    # Return (minimum, maximum, label); prices are in million VND
    text = question.casefold().replace("–", "-").replace("—", "-")
    unit = r"(tỷ|triệu|tr|million)"
    number = r"(\d+(?:[.,]\d+)?)"

    # Example: từ 500 đến 800 triệu / 0.5 đến 0.8 tỷ
    match = re.search(
        rf"(?:từ\s*)?{number}\s*(?:triệu|tr|tỷ)?\s*(?:đến|tới|-)\s*{number}\s*{unit}",
        text,
    )
    if match:
        first, second, final_unit = match.group(1), match.group(2), match.group(3)
        first_unit_match = re.search(rf"{number}\s*(triệu|tr|tỷ)", text[:match.start(2)])
        first_unit = first_unit_match.group(2) if first_unit_match else final_unit
        low = price_to_million(first, first_unit)
        high = price_to_million(second, final_unit)
        return min(low, high), max(low, high), "khoảng giá"

    match = re.search(rf"{number}\s*{unit}", text)
    if not match:
        return None, None, None
    amount, found_unit = match.group(1), match.group(2)
    value = price_to_million(amount, found_unit)

    if "khoảng" in text or "tầm" in text or "xấp xỉ" in text:
        return value * 0.9, value * 1.1, "khoảng ±10%"

    before = text[:match.start()]
    if re.search(r"(trên|hơn|ít nhất|tối thiểu|từ)\s*$", before):
        return value, None, "từ mức giá"
    if re.search(r"(dưới|không quá|tối đa|thấp hơn|chưa đến|ít hơn)\s*$", before):
        return None, value, "không vượt ngân sách"
    # Preserve the original app's behavior: a stated budget is a maximum.
    return None, value, "không vượt ngân sách"


def extract_year(question):
    match = re.search(r"\b(19\d{2}|20[0-3]\d)\b", question)
    return int(match.group(1)) if match else None


def extract_seats(question):
    match = re.search(r"\b(\d+)\s*(?:chỗ(?:\s+ngồi)?|ghế)\b", question, re.IGNORECASE)
    return int(match.group(1)) if match else None


def normalize_brand_text(value):
    # Normalize accents and punctuation for brand-name comparison.
    value = unicodedata.normalize("NFKD", str(value).casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def extract_brand(question, dataset):
    # Detect canonical brands with exact matching and typo-tolerant fuzzy matching.
    known_brands = [
        "Mercedes-Benz", "Mercedes Benz", "Mitsubishi", "Volkswagen", "Land Rover",
        "VinFast", "Toyota", "Hyundai", "Honda", "Kia", "Mazda", "Ford", "BMW",
        "Audi", "Lexus", "Porsche", "Peugeot", "Volvo", "Subaru", "Isuzu", "Suzuki",
        "Nissan", "Chevrolet", "MG", "BYD", "Tata", "Mahindra", "Rolls Royce",
    ]
    normalized_question = normalize_brand_text(question)
    normalized_brands = {brand: normalize_brand_text(brand) for brand in known_brands}

    # Exact brand matching always wins and avoids unnecessary fuzzy guesses.
    for brand in sorted(known_brands, key=len, reverse=True):
        normalized_brand = normalized_brands[brand]
        if re.search(rf"(?<![a-z0-9]){re.escape(normalized_brand)}(?![a-z0-9])", normalized_question):
            return brand

    # Ignore common query words and numbers; they should not be treated as brands.
    stop_words = {
        "tim", "xe", "cho", "toi", "mua", "ban", "gia", "nam", "san", "xuat",
        "duoi", "tren", "tam", "khoang", "ngan", "sach", "trieu", "ty", "dong",
        "oto", "o", "to", "ngoi", "so", "km", "da", "di", "mot", "cac", "co",
    }
    words = [
        word for word in re.findall(r"[a-zA-ZÀ-ỹ]+", question.casefold())
        if normalize_brand_text(word) not in stop_words
        and len(normalize_brand_text(word)) >= 2
    ]

    # Build 1-3 word windows, then compare both spaced and compact forms.
    query_parts = []
    for index in range(len(words)):
        for window_size in (1, 2, 3):
            window = words[index:index + window_size]
            if len(window) == window_size:
                query_parts.append(" ".join(window))

    best_match = None
    best_score = 0.0
    second_best_score = 0.0
    for part in query_parts:
        normalized_part = normalize_brand_text(part)
        compact_part = normalized_part.replace(" ", "")
        if not normalized_part:
            continue
        for brand, normalized_brand in normalized_brands.items():
            compact_brand = normalized_brand.replace(" ", "")
            # Short brands (MG/Kia/BYD) are handled by exact matching only.
            if len(compact_brand) < 4:
                continue
            spaced_score = SequenceMatcher(None, normalized_part, normalized_brand).ratio()
            compact_score = SequenceMatcher(None, compact_part, compact_brand).ratio()
            score = max(spaced_score, compact_score)
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_match = brand
            elif score > second_best_score:
                second_best_score = score

    # Use a slightly lower threshold for long brands; require a margin to avoid
    # confusing ordinary words with a brand name.
    if best_match and best_score >= 0.76 and (best_score - second_best_score) >= 0.04:
        return best_match

    # Preserve support for a full model name that exists in the dataset.
    name_col = find_column(dataset, ["Name", "Tên xe", "Dòng xe", "Model"])
    if name_col:
        for name in dataset[name_col].dropna().astype(str).unique():
            normalized_name = normalize_brand_text(name)
            if len(normalized_name) >= 4 and normalized_name in normalized_question:
                return name
    return None


def extract_body_type(question):
    for body in ["SUV", "Sedan", "MPV", "Hatchback", "Pickup", "Bán tải", "Crossover"]:
        if body.casefold() in question.casefold():
            return body
    return None


@st.cache_data(show_spinner="Đang đọc dữ liệu xe...")
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    price_col = find_column(df, ["Price", "Giá", "Giá bán"])
    if not price_col:
        raise ValueError("Dataset cần có cột Price hoặc cột giá tương đương.")
    df["Price_Million"] = df[price_col].apply(convert_price)
    df = df.dropna(subset=["Price_Million"])
    return df[df["Price_Million"] > 0].copy().reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_text_model(model_name):
    return SentenceTransformer(model_name, device=DEVICE)


def build_description(row, columns):
    parts = []
    for column in columns:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            parts.append(f"{column}: {str(value).strip()}")
    return ". ".join(parts)


def searchable_columns(df):
    preferred = [
        "Name", "Tên xe", "Dòng xe", "Model", "Năm sản xuất", "Số chỗ ngồi", "Số chỗ",
        "Kiểu dáng", "Số Km đã đi", "Tình trạng", "Xuất xứ", "Động cơ",
        "Màu ngoại thất", "Màu nội thất", "Description",
    ]
    columns = []
    for name in preferred:
        column = find_column(df, [name])
        if column and column not in columns:
            columns.append(column)
    return columns or list(df.columns[:10])


@st.cache_data(show_spinner=False)
def cached_car_embedding_batch(descriptions, model_name):
    model = load_text_model(model_name)
    return model.encode(
        list(descriptions),
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

def filter_candidates(question, dataset, *, apply_price=True, apply_year=True, apply_seats=True, apply_body=True):
    candidates = dataset.copy()
    minimum, maximum, _ = extract_price_filter(question)
    year = extract_year(question)
    seats = extract_seats(question)
    brand = extract_brand(question, dataset)
    body_type = extract_body_type(question)

    if apply_price and minimum is not None:
        candidates = candidates[candidates["Price_Million"] >= minimum]
    if apply_price and maximum is not None:
        candidates = candidates[candidates["Price_Million"] <= maximum]

    year_col = find_column(candidates, ["Năm sản xuất"])
    if apply_year and year and year_col:
        candidates = candidates[pd.to_numeric(candidates[year_col], errors="coerce") == year]

    seats_col = find_column(candidates, ["Số chỗ ngồi", "Số chỗ"])
    if apply_seats and seats and seats_col:
        seat_values = pd.to_numeric(candidates[seats_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        candidates = candidates[seat_values == seats]

    name_col = find_column(candidates, ["Name", "Tên xe", "Dòng xe", "Model"])
    if brand and name_col:
        # Brand is a filter only if the dataset contains matching rows.
        matched = candidates[name_col].astype(str).str.contains(re.escape(brand), case=False, na=False)
        candidates = candidates[matched]

    body_col = find_column(candidates, ["Kiểu dáng"])
    if apply_body and body_type and body_col:
        candidates = candidates[candidates[body_col].astype(str).str.contains(re.escape(body_type), case=False, na=False)]

    return candidates.copy(), brand, minimum, maximum


def is_car_related_query(question, dataset):
    # Reject unrelated chat before running semantic search over car listings.
    text = question.casefold()
    car_terms = [
        "xe", "ô tô", "oto", "ô-tô", "car", "automobile", "vehicle", "mua xe",
        "bán xe", "tìm xe", "giá xe", "hãng xe", "dòng xe", "số km", "số chỗ",
        "suv", "sedan", "mpv", "hatchback", "pickup", "crossover", "bán tải",
    ]
    if any(term in text for term in car_terms):
        return True
    if extract_brand(question, dataset):
        return True
    minimum, maximum, _ = extract_price_filter(question)
    return minimum is not None or maximum is not None or extract_year(question) is not None or extract_seats(question) is not None


def recommend_cars(question, dataset):
    if not is_car_related_query(question, dataset):
        empty_results = dataset.iloc[0:0].copy()
        empty_results.attrs["fallback_message"] = (
            "Không tìm thấy xe phù hợp trong dataset. Vui lòng prompt lại với thông tin về hãng xe, "
            "năm sản xuất, giá hoặc đặc điểm xe."
        )
        return empty_results, None, None, None

    candidates, brand, minimum, maximum = filter_candidates(question, dataset)
    fallback_message = None

    if candidates.empty and is_car_related_query(question, dataset):
        # Relax constraints gradually: price first, then year, then other details.
        # Keep the brand whenever possible so fallback results remain relevant.
        candidates, _, _, _ = filter_candidates(
            question, dataset, apply_price=False
        )
        if not candidates.empty:
            fallback_message = "Không có xe đáp ứng đầy đủ điều kiện giá. Các gợi ý dưới đây giữ nguyên hãng/năm và ưu tiên xe có giá gần ngân sách nhất."
        else:
            candidates, _, _, _ = filter_candidates(
                question, dataset, apply_price=False, apply_year=False
            )
            if not candidates.empty:
                fallback_message = "Không có xe đáp ứng đầy đủ điều kiện giá và năm. Các gợi ý dưới đây ưu tiên đúng hãng và giá gần ngân sách nhất."
            else:
                candidates, _, _, _ = filter_candidates(
                    question, dataset, apply_price=False,
                    apply_year=False, apply_seats=False, apply_body=False,
                )
                if not candidates.empty:
                    fallback_message = "Không tìm thấy xe khớp toàn bộ điều kiện. Các gợi ý dưới đây ưu tiên đúng hãng và giá gần ngân sách nhất; anh kiểm tra lại năm, số chỗ hoặc kiểu dáng nhé."

    if candidates.empty:
        candidates.attrs["fallback_message"] = (
            "Không tìm thấy xe phù hợp trong dataset. Vui lòng prompt lại với hãng xe, năm sản xuất hoặc khoảng giá khác."
            if not is_car_related_query(question, dataset)
            else "Dataset hiện không có xe phù hợp với yêu cầu này. Anh thử đổi hãng xe hoặc nới điều kiện giá/năm nhé."
        )
        return candidates, brand, minimum, maximum

    columns = searchable_columns(candidates)
    candidates["_SearchText"] = candidates.apply(lambda row: build_description(row, columns), axis=1)
    candidates = candidates[candidates["_SearchText"].str.len() > 0].copy()
    if candidates.empty:
        return candidates, brand, minimum, maximum

    # Encode candidates in cached batches and report the actual batch progress.
    all_texts = candidates["_SearchText"].astype(str).tolist()
    batch_size = 32
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    progress = st.progress(5, text="Chuẩn bị dữ liệu tìm kiếm... 5%")
    embedding_batches = []

    for batch_index, start in enumerate(range(0, len(all_texts), batch_size), start=1):
        batch = tuple(all_texts[start:start + batch_size])
        embedding_batches.append(cached_car_embedding_batch(batch, MODEL_NAME))
        percent = 5 + int(batch_index / total_batches * 75)
        progress.progress(
            percent,
            text=f"Đang xử lý thông tin... {percent}%",
        )

    car_embeddings = np.vstack(embedding_batches)
    text_model = load_text_model(MODEL_NAME)
    query_embedding = text_model.encode(
        [question],
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    progress.progress(90, text="Đang so sánh và xếp hạng xe... 90%")
    candidates["Similarity"] = car_embeddings @ query_embedding
    if fallback_message and (minimum is not None or maximum is not None):
        # When the requested budget cannot be met, show listings closest to that budget.
        budget = maximum if maximum is not None else minimum
        candidates["_BudgetDistance"] = (candidates["Price_Million"] - budget).abs()
        candidates = candidates.sort_values(["_BudgetDistance", "Similarity"], ascending=[True, False])
    else:
        candidates = candidates.sort_values("Similarity", ascending=False)

    name_col = find_column(candidates, ["Name", "Tên xe", "Dòng xe", "Model"])
    if name_col:
        candidates = candidates.drop_duplicates(subset=[name_col, "Price_Million"])
    candidates = candidates.head(TOP_K).copy()
    candidates["_SearchText"] = candidates["_SearchText"].str.replace(r"\s+", " ", regex=True)
    candidates.attrs["fallback_message"] = fallback_message
    progress.progress(100, text="Hoàn tất tìm kiếm... 100%")
    progress.empty()
    return candidates, brand, minimum, maximum



def display_results(results, dataset, brand=None):
    if results.empty:
        st.info(results.attrs.get(
            "fallback_message",
            "Không tìm thấy xe phù hợp trong dataset. Vui lòng prompt lại với thông tin về xe.",
        ))
        return

    fallback_message = results.attrs.get("fallback_message")
    if fallback_message:
        st.warning(fallback_message)
    st.success(f"Tìm thấy {len(results)} kết quả phù hợp trong dữ liệu hiện có.")
    st.caption("Giá hiển thị là giá tin đăng trong dataset, không phải giá thị trường trực tiếp theo thời gian thực.")

    name_col = find_column(results, ["Name", "Tên xe", "Dòng xe", "Model", "Hãng xe"])
    for index, (_, car) in enumerate(results.iterrows(), start=1):
        name = str(car[name_col]) if name_col and pd.notna(car[name_col]) else f"Xe #{index}"
        with st.container(border=True):
            st.subheader(f"{index}. {name}")
            st.metric("Giá tin đăng", f"{car['Price_Million']:,.0f} triệu VND")
            if "Similarity" in car and pd.notna(car["Similarity"]):
                st.caption(f"Điểm tương đồng ngữ nghĩa: {car['Similarity']:.3f}")
            details = []
            for label in ["Năm sản xuất", "Số chỗ ngồi", "Kiểu dáng", "Số Km đã đi", "Tình trạng", "Xuất xứ", "Động cơ", "Màu ngoại thất"]:
                col = find_column(results, [label])
                if col and pd.notna(car[col]):
                    details.append(f"**{label}:** {car[col]}")
            if details:
                st.markdown("  ·  ".join(details))

    # Two Linear Regression charts: price by manufacturing year and mileage.
    market = dataset.copy()
    market_name_col = find_column(market, ["Name", "Tên xe", "Dòng xe", "Model"])

    if brand and market_name_col:
        brand_rows = market[market[market_name_col].astype(str).str.contains(re.escape(brand), case=False, na=False)]
        if not brand_rows.empty:
            market = brand_rows
            market_label = f"{brand}"
        else:
            market_label = "Toàn bộ dataset"
    else:
        market_label = "Toàn bộ dataset"

    st.markdown("### Phân tích giá bằng Linear Regression")
    st.caption(
        f"Dữ liệu phân tích: {market_label}. Mỗi biểu đồ gồm các tin đăng trong dataset và đường hồi quy tuyến tính; "
        "các xe được gợi ý được đánh dấu riêng khi có đủ thông tin."
    )

    chart_year, chart_mileage = st.columns(2)

    # Chart 1: manufacturing year vs listed price.
    with chart_year:
        st.markdown("**1. Năm sản xuất và giá**")
        market_year_col = find_column(market, ["Năm sản xuất"])
        if market_year_col:
            year_data = market[["Price_Million", market_year_col]].copy()
            year_data["Year"] = pd.to_numeric(year_data[market_year_col], errors="coerce")
            year_data = year_data.dropna(subset=["Year", "Price_Million"])
            year_data = year_data[(year_data["Year"] >= 1900) & (year_data["Year"] <= 2035)]

            if len(year_data) >= 2 and year_data["Year"].nunique() >= 2:
                year_model = LinearRegression().fit(year_data[["Year"]], year_data["Price_Million"])
                year_r2 = r2_score(year_data["Price_Million"], year_model.predict(year_data[["Year"]])) if len(year_data) > 2 else np.nan
                fig_year, ax_year = plt.subplots(figsize=(6, 4.5))
                ax_year.scatter(year_data["Year"], year_data["Price_Million"], color="#7DA7D9", alpha=0.45, s=18, label="Tin đăng")
                line_years = np.linspace(year_data["Year"].min(), year_data["Year"].max(), 100)
                ax_year.plot(line_years, year_model.predict(pd.DataFrame({"Year": line_years})), color="#E63946", linewidth=2.8, label="Linear Regression")

                result_year_col = find_column(results, ["Năm sản xuất"])
                if result_year_col:
                    result_years = results[["Price_Million", result_year_col]].copy()
                    result_years["Year"] = pd.to_numeric(result_years[result_year_col], errors="coerce")
                    result_years = result_years.dropna(subset=["Year", "Price_Million"])
                    if not result_years.empty:
                        ax_year.scatter(result_years["Year"], result_years["Price_Million"], color="#F4A261", marker="x", s=75, linewidths=2.2, label="Xe gợi ý")

                ax_year.set_xlabel("Năm sản xuất")
                ax_year.set_ylabel("Giá (triệu VND)")
                ax_year.set_xlim(year_data["Year"].min(), year_data["Year"].max())
                ax_year.set_ylim(bottom=0)
                ax_year.grid(True, alpha=0.25)
                ax_year.legend(fontsize=8)
                fig_year.tight_layout()
                st.pyplot(fig_year, width='stretch')
                plt.close(fig_year)
                if np.isfinite(year_r2):
                    st.caption(f"Số tin đăng: {len(year_data):,} · R²: {year_r2:.3f}")
            else:
                st.info("Chưa đủ dữ liệu năm sản xuất để vẽ đường hồi quy.")
        else:
            st.info("Dataset không có cột năm sản xuất.")

    # Chart 2: mileage vs listed price.
    with chart_mileage:
        st.markdown("**2. Số km đã đi và giá**")
        mileage_col = find_column(market, ["Số Km đã đi", "Số km đã đi", "Km đã đi", "Mileage", "Kilometers", "Kilometres"])
        if mileage_col:
            mileage_data = market[["Price_Million", mileage_col]].copy()
            mileage_data["Mileage"] = pd.to_numeric(
                mileage_data[mileage_col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                errors="coerce",
            )
            mileage_data = mileage_data.dropna(subset=["Mileage", "Price_Million"])
            mileage_data = mileage_data[mileage_data["Mileage"] >= 0]

            if len(mileage_data) >= 2 and mileage_data["Mileage"].nunique() >= 2:
                mileage_model = LinearRegression().fit(mileage_data[["Mileage"]], mileage_data["Price_Million"])
                mileage_r2 = r2_score(mileage_data["Price_Million"], mileage_model.predict(mileage_data[["Mileage"]])) if len(mileage_data) > 2 else np.nan
                fig_mileage, ax_mileage = plt.subplots(figsize=(6, 4.5))
                ax_mileage.scatter(mileage_data["Mileage"], mileage_data["Price_Million"], color="#7DA7D9", alpha=0.45, s=18, label="Tin đăng")
                line_mileage = np.linspace(0, 100000, 100)
                ax_mileage.plot(line_mileage, mileage_model.predict(pd.DataFrame({"Mileage": line_mileage})), color="#E63946", linewidth=2.8, label="Linear Regression")

                result_mileage_col = find_column(results, ["Số Km đã đi", "Số km đã đi", "Km đã đi", "Mileage", "Kilometers", "Kilometres"])
                if result_mileage_col:
                    result_mileages = results[["Price_Million", result_mileage_col]].copy()
                    result_mileages["Mileage"] = pd.to_numeric(
                        result_mileages[result_mileage_col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                        errors="coerce",
                    )
                    result_mileages = result_mileages.dropna(subset=["Mileage", "Price_Million"])
                    if not result_mileages.empty:
                        ax_mileage.scatter(result_mileages["Mileage"], result_mileages["Price_Million"], color="#F4A261", marker="x", s=75, linewidths=2.2, label="Xe gợi ý")

                ax_mileage.set_xlabel("Số km đã đi")
                ax_mileage.set_ylabel("Giá (triệu VND)")
                ax_mileage.set_xlim(0, 100000)
                ax_mileage.set_ylim(bottom=0)
                ax_mileage.ticklabel_format(style="plain", axis="x")
                ax_mileage.grid(True, alpha=0.25)
                ax_mileage.legend(fontsize=8)
                fig_mileage.tight_layout()
                st.pyplot(fig_mileage, width='stretch')
                plt.close(fig_mileage)
                if np.isfinite(mileage_r2):
                    st.caption(f"Số tin đăng: {len(mileage_data):,} · R²: {mileage_r2:.3f}")
            else:
                st.info("Chưa đủ dữ liệu số km đã đi để vẽ đường hồi quy.")
        else:
            st.info("Dataset không có cột số km đã đi.")


def process_text_question(question, dataset):
    results, brand, minimum, maximum = recommend_cars(question, dataset)
    display_results(results, dataset, brand=brand)
    if minimum is not None or maximum is not None:
        low_text = f"{minimum:,.0f}" if minimum is not None else "0"
        high_text = f"{maximum:,.0f}" if maximum is not None else "0"
        st.caption(f"Khoảng giá: {low_text}–{high_text} triệu VND.")


def get_cnn_dataset_signature():
    # Detect changes in image files to decide whether retraining is needed.
    hasher = hashlib.sha256()

    for split_dir in [CNN_TRAIN_DIR, CNN_TEST_DIR]:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục: {split_dir}")

        image_files = sorted(
            path for path in split_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )

        if not image_files:
            raise ValueError(f"Không tìm thấy ảnh trong {split_dir}")

        for path in image_files:
            stat = path.stat()
            relative_path = str(path.relative_to(CNN_DATASET_DIR))
            hasher.update(
                f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
            )

    return hasher.hexdigest()


@st.cache_resource(show_spinner=False)
def load_or_train_cnn(dataset_signature):
    # TensorFlow is loaded only for image recognition.
    import tensorflow as tf
    from tensorflow.keras import layers, models

    CNN_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse the model when the dataset has not changed.
    if CNN_MODEL_PATH.exists() and CNN_METADATA_PATH.exists():
        try:
            with open(CNN_METADATA_PATH, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            if metadata.get("dataset_signature") == dataset_signature:
                model = tf.keras.models.load_model(CNN_MODEL_PATH)
                return model, metadata["class_names"]
        except (OSError, ValueError, KeyError):
            pass

    st.info("Đang huấn luyện CNN từ dữ liệu ảnh Train/Test...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        CNN_TRAIN_DIR,
        image_size=CNN_IMAGE_SIZE,
        batch_size=CNN_BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        CNN_TEST_DIR,
        image_size=CNN_IMAGE_SIZE,
        batch_size=CNN_BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names

    if test_ds.class_names != class_names:
        raise ValueError(
            "Các lớp ảnh trong Train và Test không khớp nhau. "
            f"Train: {class_names}; Test: {test_ds.class_names}"
        )

    if len(class_names) < 2:
        raise ValueError("CNN cần ít nhất hai lớp xe để huấn luyện.")

    autotune = tf.data.AUTOTUNE

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
    ])

    def prepare_train(images, labels):
        images = augmentation(images, training=True)
        return images / 255.0, labels

    def prepare_test(images, labels):
        return images / 255.0, labels

    train_ds = train_ds.map(
        prepare_train, num_parallel_calls=autotune
    ).prefetch(autotune)

    test_ds = test_ds.map(
        prepare_test, num_parallel_calls=autotune
    ).prefetch(autotune)

    model = models.Sequential([
        layers.Input(shape=(*CNN_IMAGE_SIZE, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=CNN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    model.save(CNN_MODEL_PATH)

    with open(CNN_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump({
            "dataset_signature": dataset_signature,
            "class_names": class_names,
            "test_accuracy": float(test_accuracy),
        }, file, ensure_ascii=False, indent=2)

    st.success(f"CNN đã huấn luyện xong. Test accuracy: {test_accuracy:.2%}")

    return model, class_names


def recognize_car_image(uploaded_file, model, class_names):
    # Hỗ trợ cả ảnh upload và ảnh PIL đã mở
    if isinstance(uploaded_file, Image.Image):
        image = uploaded_file.convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    image = image.resize(CNN_IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence

def main():
    st.title("Support System for Car Recognition, Valuation, and Recommendation in Vietnam")
    st.caption("Tìm xe theo nhu cầu và phân tích giá cả trên thị trường Việt Nam.")

    cnn_model = None
    cnn_class_names = []

    try:
        dataset_signature = get_cnn_dataset_signature()

        with st.spinner("Đang kiểm tra mô hình nhận diện ảnh..."):
            cnn_model, cnn_class_names = load_or_train_cnn(dataset_signature)

    except Exception as error:
        st.warning(f"Chưa thể khởi tạo CNN: {error}")

    st.caption(f"Model: {MODEL_NAME.split('/')[-1]}")

    try:
        dataset = load_dataset()
    except Exception as error:
        st.error(f"Không thể tải dataset: {error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "image_result":
                st.write(f"**Xe được nhận diện:** {message['car_name']}")
                st.write(f"**Độ tin cậy:** {message['confidence']:.1%}")
            else:
                st.markdown(message["content"])

            if message.get("image") is not None:
                st.image(
                    message["image"],
                    caption="Ảnh xe đã tải lên",
                    width=300,
                )

            if message.get("type") == "text_result":
                with st.spinner("Đang hiển thị kết quả..."):
                    process_text_question(message["question"], dataset)

    # Text prompt and image upload share the same chat input.
    prompt_data = st.chat_input(
        "Nhập yêu cầu tìm xe hoặc tải ảnh xe...",
        accept_file="multiple",
        file_type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if prompt_data is None:
        return

    question = (prompt_data.text or "").strip()
    uploaded_files = prompt_data.files

    if not question and not uploaded_files:
        return

    image = None
    car_name = None
    confidence = None

    if uploaded_files:
        uploaded_file = uploaded_files[0]

        try:
            image = Image.open(uploaded_file).convert("RGB")
            image.load()
        except Exception:
            st.error("File tải lên không phải ảnh hợp lệ.")
            return

        if cnn_model is None:
            st.error("Mô hình CNN chưa sẵn sàng để nhận diện ảnh.")
            return

        try:
            with st.spinner("Đang nhận diện xe từ ảnh..."):
                car_name, confidence = recognize_car_image(
                    image.copy(),
                    cnn_model,
                    cnn_class_names,
                )
        except Exception as error:
            st.error(f"Không thể nhận diện ảnh: {error}")
            return

        if not question:
            question = "Nhận diện xe trong ảnh"

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "image": image.copy() if image is not None else None,
    })

    with st.chat_message("user"):
        if question:
            st.markdown(question)
        if image is not None:
            st.image(image, caption="Ảnh xe đã tải lên", width=300)

    with st.chat_message("assistant"):
        if image is not None:
            st.success(f"**Xe được nhận diện:** {car_name}")
            st.write(f"**Độ tin cậy:** {confidence:.1%}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Xe được nhận diện: {car_name}",
                "type": "image_result",
                "car_name": car_name,
                "confidence": confidence,
            })

        # Chỉ tìm xe khi không upload ảnh.
        if image is None and question:
            process_text_question(question, dataset)

            st.session_state.messages.append({
                "role": "assistant",
                "content": "Đã tìm và hiển thị các xe phù hợp cùng biểu đồ Linear Regression.",
                "type": "text_result",
                "question": question,
            })

if __name__ == "__main__":
    main()

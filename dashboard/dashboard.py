import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency

sns.set(style='whitegrid')

# Helper functions untuk menyiapkan berbagai dataframe yang dibutuhkan
@st.cache_data
def create_daily_orders_df(df):
    daily_orders_df = df.drop_duplicates(subset=['order_id', 'payment_sequential']).groupby(by=df['order_purchase_timestamp'].dt.date).agg({
        "order_id": "nunique",
        "payment_value": "sum"
    }).reset_index()
    daily_orders_df.rename(columns={
        "order_purchase_timestamp": "order_date",
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    return daily_orders_df

@st.cache_data
def create_ytd_comparison_df(df):
    # Filter hanya bulan Jan-Sep untuk tahun 2017 & 2018
    ytd_df = df[(df['order_purchase_timestamp'].dt.month <= 9) & 
                (df['order_purchase_timestamp'].dt.year.isin([2017, 2018]))].copy()
    ytd_df['year'] = ytd_df['order_purchase_timestamp'].dt.year
    return ytd_df

@st.cache_data
def create_rfm_df(df):
    rfm_data = df.drop_duplicates(subset=['order_id', 'payment_sequential'])
    rfm_df = rfm_data.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max", 
        "order_id": "nunique",            
        "payment_value": "sum"            
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    
    recent_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    # Scoring sesuai logic di Notebook (rank based normalization)
    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

    rfm_df['r_score'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max() * 5).round(0)
    rfm_df['f_score'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max() * 5).round(0)
    rfm_df['m_score'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max() * 5).round(0)

    # Menentukan segmen pelanggan
    def segment_customer(df):
        if df['r_score'] >= 4 and df['f_score'] >= 4 and df['m_score'] >= 4:
            return 'Champions'
        elif df['r_score'] >= 3 and df['f_score'] >= 3:
            return 'Loyal Customers'
        elif df['r_score'] >= 3 and df['m_score'] >= 4:
            return 'Big Spenders'
        elif df['r_score'] <= 2:
            return 'At Risk / Hibernating'
        else:
            return 'Others'

    rfm_df['customer_segment'] = rfm_df.apply(segment_customer, axis=1)
    return rfm_df

# Load cleaned data
@st.cache_data
def get_data():
    df = pd.read_csv("main_data.csv")
    datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for column in datetime_columns:
        df[column] = pd.to_datetime(df[column])
    
    df.sort_values(by="order_purchase_timestamp", inplace=True)
    df['delivery_duration_days'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days
    df.reset_index(drop=True, inplace=True)
    return df

all_df = get_data()

# Sidebar
with st.sidebar:
    st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png")
    
    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=all_df['order_purchase_timestamp'].min().date(),
        max_value=all_df['order_purchase_timestamp'].max().date(),
        value=[all_df['order_purchase_timestamp'].min().date(), all_df['order_purchase_timestamp'].max().date()]
    )

# Filter data berdasarkan sidebar
main_filtered_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                          (all_df["order_purchase_timestamp"] <= str(end_date))]

# Menyiapkan berbagai dataframe
daily_orders_df = create_daily_orders_df(main_filtered_df)
rfm_df = create_rfm_df(main_filtered_df)
ytd_df = create_ytd_comparison_df(main_filtered_df)

# Header Dashboard
st.header('Olist E-Commerce Dashboard :sparkles:')

# --- Penjelasan Metodologi ---
with st.expander("ℹ️ Methodology Note"):
    st.write("""
    Dashboard ini menggunakan pendekatan **Year-to-Date (YTD) Comparison** (Januari-September 2017 vs 2018).
    Hal ini dilakukan untuk memastikan perbandingan yang *Apple-to-Apple*, menghindari bias musiman akhir tahun, 
    dan menyesuaikan dengan keterbatasan data tahun 2018 yang hanya tersedia hingga Oktober.
    """)

st.subheader('Daily Orders')
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Orders", value=daily_orders_df.order_count.sum())
with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(), "BRL", locale='pt_BR') 
    st.metric("Total Revenue", value=total_revenue)
with col3:
    avg_review = main_filtered_df.drop_duplicates(subset=['order_id', 'product_id']).review_score.mean()
    st.metric("Avg Review Score", value=f"{avg_review:.2f} ⭐")

fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(data=daily_orders_df, x="order_date", y="order_count", color="#90CAF9", ax=ax)
ax.set_title("Tren Pesanan Harian", fontsize=20)
st.pyplot(fig)

# --- Section 1: Revenue Comparison (Q1) ---
st.write("---")
st.subheader("1. Pertumbuhan Pendapatan di 5 Negara Bagian Teratas")
top_5_states = ytd_df.groupby('customer_state')['payment_value'].sum().nlargest(5).index.tolist()
state_revenue = ytd_df[ytd_df['customer_state'].isin(top_5_states)].groupby(['customer_state', 'year'])['payment_value'].sum().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=state_revenue, x='customer_state', y='payment_value', hue='year', palette='viridis', ax=ax)
ax.set_title("Perbandingan Pendapatan YoY (Jan-Sep)", fontsize=15)
ax.set_ylabel("Total Pendapatan (BRL)")
st.pyplot(fig)
st.info("Insight: Sao Paulo (SP) mendominasi pasar dengan pertumbuhan tertinggi mencapai 127% YoY.")

# --- Section 2: Review Score Drop (Q2) ---
st.write("---")
st.subheader("2. Kategori Produk dengan Penurunan Skor Ulasan Terbesar")
cat_scores = ytd_df.groupby(['product_category_name_english', 'year']).agg({'review_score': ['mean', 'count']}).reset_index()
cat_scores.columns = ['category', 'year', 'avg_score', 'count']
valid_cats = cat_scores[cat_scores['count'] >= 30]['category'].unique()
compare_cat = cat_scores[cat_scores['category'].isin(valid_cats)].pivot(index='category', columns='year', values='avg_score').dropna()
compare_cat['drop'] = compare_cat[2017] - compare_cat[2018]
top_drops = compare_cat.sort_values('drop', ascending=False).head(10).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_drops, x='drop', y='category', palette='Reds_r', ax=ax)
ax.set_title("Top 10 Penurunan Skor Rata-rata Ulasan", fontsize=15)
ax.set_xlabel("Besar Penurunan Skor (Poin)")
st.pyplot(fig)
st.warning("Insight: Kategori 'cine_photo' mengalami penurunan kualitas persepsi pelanggan paling tajam.")

# --- Section 3: Delivery Efficiency (Q3) ---
st.write("---")
st.subheader("3. Efisiensi Waktu Pengiriman (Delivery Time)")
delivery_stats = ytd_df[ytd_df['order_status']=='delivered'].groupby('year')['delivery_duration_days'].mean().reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
    data=delivery_stats, 
    x='year', 
    y='delivery_duration_days', 
    palette=['#4C72B0', '#C44E52'],
    ax=ax
)
ax.set_title("Rata-rata Waktu Pengiriman (Hari)", fontsize=15)
st.pyplot(fig)
st.write(f"Rata-rata durasi: **{delivery_stats.iloc[0,1]:.2f} hari** (2017) vs **{delivery_stats.iloc[1,1]:.2f} hari** (2018).")

# --- Section 4: Customer Loyalty (Q4) ---
st.write("---")
st.subheader("4. Distribusi Frekuensi Pembelian (Loyalitas)")
freq_data = ytd_df.groupby(['year', 'customer_unique_id']).order_id.nunique().reset_index()
freq_dist = freq_data.groupby(['year', 'order_id']).size().reset_index(name='count')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=freq_dist[freq_dist['order_id'] <= 3], x='order_id', y='count', hue='year', palette='viridis', ax=ax)
ax.set_yscale('log')
ax.set_title("Distribusi Frekuensi Pembelian (Log Scale)", fontsize=15)
ax.set_xlabel("Jumlah Pembelian")
ax.set_ylabel("Jumlah Pelanggan")
st.pyplot(fig)
st.error("Insight: Mayoritas pelanggan hanya berbelanja 1x. Rasio repeat order menurun di 2018.")

# --- Section 5: RFM Analysis ---
st.write("---")
st.subheader("Advanced Analysis: Customer Segmentation (RFM)")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Avg Recency (days)", value=round(rfm_df.recency.mean(), 1))
with col2:
    st.metric("Avg Frequency", value=round(rfm_df.frequency.mean(), 2))
with col3:
    avg_monetary = format_currency(rfm_df.monetary.mean(), "BRL", locale='pt_BR') 
    st.metric("Avg Monetary", value=avg_monetary)

segment_counts = rfm_df['customer_segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Customer Count']
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(
    data=segment_counts, 
    x='Customer Count', 
    y='Segment', 
    palette='viridis', ax=ax
)
ax.set_title("Distribusi Segmen Pelanggan", fontsize=15)
st.pyplot(fig)

# --- Conclusion & Recommendation ---
st.write("---")
st.write("### Conclusion & Recommendations")
st.markdown("""
1. **Performa Wilayah:** Negara bagian SP tetap menjadi pilar utama pendapatan dengan pertumbuhan eksponensial. Strategi retensi harus difokuskan di wilayah ini.
2. **Kualitas Produk:** Perlu investigasi pada kategori dengan penurunan skor ulasan tajam (seperti perlengkapan natal dan furnitur kantor) untuk memperbaiki standar QC atau deskripsi produk.
3. **Logistik:** Waktu pengiriman stagnan (11.6 hari). Optimasi jalur distribusi di wilayah luar SP sangat disarankan untuk meningkatkan efisiensi.
4. **Loyalitas:** Rendahnya repeat order menunjukkan bisnis sangat bergantung pada akuisisi pelanggan baru. Program loyalitas/cashback sangat direkomendasikan untuk segmen *At Risk*.
""")

st.caption('Copyright (c) Saputra Budiman 2024')
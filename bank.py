import os
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import datetime
from concurrent.futures import ThreadPoolExecutor
from pandas.api.types import is_datetime64tz_dtype

# ---------------------------
# Set page configuration
# ---------------------------
st.set_page_config(
    page_title="Global Banks Performance Tracker",  # Default title in English
    layout="wide",
    page_icon="🏦"
)

# ---------------------------
# Language selection
# ---------------------------
if "language" not in st.session_state:
    st.session_state.language = "English"

language = st.sidebar.radio("Select Language / اختر اللغة", ["English", "Arabic"], index=["English", "Arabic"].index(st.session_state.language))
st.session_state.language = language

if language == "Arabic":
    L = {
        "page_title": "متتبع أداء المصارف العالمية",
        "app_title": "متتبع أداء المصارف العالمية",
        "app_description": ("يقدم هذا التطبيق تحليلاً شاملاً لأداء أفضل 30 مصرفًا ماليًا عالميًا على مدى السنوات الثلاث الماضية.\n\n"
                            "**مصدر البيانات:** Yahoo Finance\n"
                            "**التطبيع:** يتم تطبيع الأسعار باستخدام مقياس MinMax للمقارنة.\n"
                            "_مرر المؤشر فوق التسميات للمزيد من التفاصيل._"),
        "filter_options": "خيارات التصفية",
        "select_company": "اختر مصرفًا:",
        "select_date_range": "اختر نطاق التاريخ:",
        "data_last_30": "البيانات (آخر 30 يومًا)",
        "stock_price": "سعر السهم",
        "normalized_comparison": "مقارنة سعر السهم المُطبع",
        "efficiency_comparison": "مقارنة الكفاءة",
        "load_efficiency": "تحميل مقارنة الكفاءة",
        "kpi": "المؤشرات الرئيسية",
        "num_companies": "عدد المصارف",
        "average_growth_rate": "متوسط معدل النمو",
        "top_growing_company": "المصرف ذو النمو الأعلى",
        "summary": "ملخص",
        "summary_text": ("يوفر هذا التطبيق نظرة شاملة على أداء أفضل 30 مصرفًا ماليًا عالميًا على مدى السنوات الثلاث الماضية، "
                         "حيث يقدم أدوات تفاعلية لاستكشاف بيانات السوق، مقارنة الأداء، واستخراج مؤشرات النمو الرئيسية. "
                         "يدعم هذا النهج القائم على البيانات اتخاذ قرارات استثمارية مبنية على معلومات دقيقة."),
        "ml_section": "التنبؤ باستخدام التعلم الآلي والتوصية",
        "ml_forecast": "توقع سعر الإغلاق لليوم التالي باستخدام نموذج Gradient Boosting Regression",
        "recommendation_buy": "التوصية: بناءً على توقعات النمو، يُنصح بشراء سهم",
        "recommendation_sell": "التوصية: بناءً على توقعات النمو، يُنصح ببيع سهم",
        "note": "ملاحظة: يعتمد هذا التوقع على نموذج Gradient Boosting مع خصائص التأخر الزمني ويُستخدم لأغراض العرض التوضيحي فقط. يُنصح بإجراء المزيد من التحليل قبل اتخاذ قرارات استثمارية.",
        "single_day_warning": "نطاق التاريخ المختار هو يوم واحد. قد لا تعرض بعض الرسوم البيانية اتجاهات ذات مغزى.",
        "info_efficiency": "اضغط على الزر أعلاه لتحميل رسم مقارنة الكفاءة."
    }
else:
    L = {
        "page_title": "Global Banks Performance Tracker",
        "app_title": "Global Banks Performance Tracker",
        "app_description": ("This application provides a comprehensive analysis of the performance of the top 30 global banks over the past three years.\n\n"
                            "**Data Source:** Yahoo Finance\n"
                            "**Normalization:** Prices are normalized using MinMax scaling for comparison purposes.\n"
                            "_Hover over labels for more details._"),
        "filter_options": "Filter Options",
        "select_company": "Select a bank:",
        "select_date_range": "Select date range:",
        "data_last_30": "Data (Last 30 Days)",
        "stock_price": "Stock Price",
        "normalized_comparison": "Normalized Stock Price Comparison",
        "efficiency_comparison": "Efficiency Comparison",
        "load_efficiency": "Load Efficiency Comparison",
        "kpi": "Key Performance Indicators",
        "num_companies": "Number of Banks",
        "average_growth_rate": "Average Growth Rate",
        "top_growing_company": "Top Growing Bank",
        "summary": "Summary",
        "summary_text": ("This application provides a comprehensive overview of the performance of the top 30 global banks over the past three years, "
                         "offering interactive tools for exploring market data, comparing performance, and extracting key growth indicators. "
                         "This data-driven approach supports informed investment decisions."),
        "ml_section": "Machine Learning Forecast & Recommendation",
        "ml_forecast": "Forecasting Next Day's Closing Price using Gradient Boosting Regression",
        "recommendation_buy": "Buy Recommendation: Based on the forecast, consider buying",
        "recommendation_sell": "Sell Recommendation: Based on the forecast, consider selling",
        "note": "Note: This ML forecast uses a Gradient Boosting Regressor with lag features and should be used for demonstration purposes only. Further analysis is recommended before making investment decisions.",
        "single_day_warning": "Selected date range is a single day. Some charts might not display meaningful trends.",
        "info_efficiency": "Click the button above to load the Efficiency Comparison chart."
    }

# ---------------------------
# Data fetching function
# ---------------------------
@st.cache_data(show_spinner="Fetching data from Yahoo Finance...")
def fetch_data_yahoo(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="3y")
        data.reset_index(inplace=True)
        # Use a robust check for timezone awareness
        if not is_datetime64tz_dtype(data["Date"]):
            data["Date"] = data["Date"].dt.tz_localize("UTC")
        data["Date"] = data["Date"].dt.tz_convert("America/New_York")
        return data
    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance for {ticker}: {e}")
        return pd.DataFrame()

def fetch_all_data(tickers):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_data_yahoo, tickers))
    return {ticker: df for ticker, df in zip(tickers, results) if not df.empty}

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess_data(data_dict, start_date, end_date):
    df_all = pd.concat(
        [df.assign(Company=ticker) for ticker, df in data_dict.items()],
        ignore_index=True
    )
    # Filter the combined DataFrame by the selected date range
    df_filtered = df_all[(df_all["Date"] >= start_date) & (df_all["Date"] <= end_date)].copy()
    scaler = MinMaxScaler()
    df_filtered.loc[:, "Close_Scaled"] = scaler.fit_transform(df_filtered[["Close"]])
    return df_filtered

# ---------------------------
# Machine Learning Forecast
# ---------------------------
def ml_forecast(data_dict, start_date, end_date):
    forecast_results = {}
    for ticker, df in data_dict.items():
        # Filter each bank's data based on the selected date range
        df_ml = df.copy()
        df_ml = df_ml[(df_ml["Date"] >= start_date) & (df_ml["Date"] <= end_date)]
        df_ml.sort_values("Date", inplace=True)
        if df_ml.empty or df_ml.shape[0] < 10:
            continue
        
        # Create lag feature and convert dates for regression
        df_ml["Date_ordinal"] = df_ml["Date"].apply(lambda x: x.timestamp())
        df_ml["Close_lag1"] = df_ml["Close"].shift(1)
        df_ml = df_ml.dropna(subset=["Date_ordinal", "Close", "Close_lag1"])
        if df_ml.shape[0] < 10:
            continue
        
        X = df_ml[["Date_ordinal", "Close_lag1"]].values
        y = df_ml["Close"].values
        
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X, y)
        
        last_date_ordinal = df_ml["Date_ordinal"].iloc[-1]
        last_close = df_ml["Close"].iloc[-1]
        # Predict next day: adding 86400 seconds (one day)
        X_next = np.array([[last_date_ordinal + 86400, last_close]])
        predicted_price = model.predict(X_next)[0]
        predicted_growth = (predicted_price - last_close) / last_close
        forecast_results[ticker] = predicted_growth
    
    return forecast_results

# ---------------------------
# Main Application
# ---------------------------
def main():
    st.title(L["app_title"])
    st.markdown(L["app_description"])

    # Full names of top 30 global banks
    banks_full = {
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corporation",
        "WFC": "Wells Fargo & Company",
        "C": "Citigroup Inc.",
        "HSBC": "HSBC Holdings plc",
        "GS": "The Goldman Sachs Group, Inc.",
        "MS": "Morgan Stanley",
        "UBS": "UBS Group AG",
        "DB": "Deutsche Bank AG",
        "BNP.PA": "BNP Paribas SA",
        "SAN": "Banco Santander, S.A.",
        "ING": "ING Groep N.V.",
        "BBVA": "Banco Bilbao Vizcaya Argentaria, S.A.",
        "CS": "Credit Suisse Group AG",
        "BCS": "Barclays PLC",
        "RY": "Royal Bank of Canada",
        "TD": "The Toronto-Dominion Bank",
        "BNS": "The Bank of Nova Scotia",
        "BMO": "Bank of Montreal",
        "CIB": "BanColombia S.A.",
        "ITUB": "Itaú Unibanco Holding S.A.",
        "BSBR": "Banco Santander (Brasil) S.A.",
        "BBD": "Banco Bradesco S.A.",
        "HDB": "HDFC Bank Limited",
        "ICICIBC.NS": "ICICI Bank Limited",
        "AXP": "American Express Company",
        "COF": "Capital One Financial Corporation",
        "USB": "U.S. Bancorp",
        "PNC": "The PNC Financial Services Group, Inc.",
        "STT": "State Street Corporation"
    }

    # Fetch data for all banks
    tickers = list(banks_full.keys())
    with st.spinner("Fetching data..."):
        data_dict = fetch_all_data(tickers)

    if not data_dict:
        st.error("No data available for the selected banks.")
        st.stop()

    # Sidebar filters
    st.sidebar.header(L["filter_options"])
    available_tickers = sorted(data_dict.keys())
    display_names = [f"{ticker} - {banks_full.get(ticker, '')}" for ticker in available_tickers]
    selected_display = st.sidebar.selectbox(L["select_company"], display_names)
    selected_ticker = selected_display.split(" - ")[0]

    # Determine the overall min and max dates from the fetched data
    min_date = pd.to_datetime(min(df["Date"].min() for df in data_dict.values())).date()
    max_date = pd.to_datetime(max(df["Date"].max() for df in data_dict.values())).date()
    date_range = st.sidebar.date_input(L["select_date_range"], [min_date, max_date])
    # Convert the selected dates to timezone-aware datetime objects using a list comprehension
    start_date, end_date = [pd.to_datetime(date).tz_localize("America/New_York") for date in date_range]

    # Preprocess data based on the selected date range
    df_filtered = preprocess_data(data_dict, start_date, end_date)

    if start_date == end_date:
        st.warning(L["single_day_warning"])

    # Filter the data for the selected ticker for display
    ticker_data_filtered = df_filtered[df_filtered["Company"] == selected_ticker]
    st.subheader(f"{selected_ticker} - {banks_full.get(selected_ticker, '')} {L['data_last_30']}")
    if not ticker_data_filtered.empty:
        st.dataframe(ticker_data_filtered.tail(30))
    else:
        st.info("No data available for the selected date range.")

    st.subheader(f"{selected_ticker} - {banks_full.get(selected_ticker, '')} {L['stock_price']}")
    fig_line = px.line(
        data_dict[selected_ticker],
        x="Date",
        y="Close",
        title=f"{selected_ticker} {L['stock_price']}",
        labels={"Close": L["stock_price"], "Date": "Date"}
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader(L["normalized_comparison"])
    fig_compare = px.line(
        df_filtered,
        x="Date",
        y="Close_Scaled",
        color="Company",
        title=L["normalized_comparison"],
        labels={"Close_Scaled": L["normalized_comparison"], "Date": "Date"}
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    if st.button(L["load_efficiency"]):
        st.subheader(L["efficiency_comparison"])
        efficiency_df = df_filtered.groupby("Company")["Close_Scaled"].mean().reset_index()
        efficiency_df = efficiency_df.sort_values(by="Close_Scaled", ascending=False)
        fig_bar = px.bar(
            efficiency_df,
            x="Company",
            y="Close_Scaled",
            title="Average Normalized Stock Price per Bank",
            labels={"Close_Scaled": "Average Normalized Price", "Company": "Bank"},
            color="Close_Scaled",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(L["info_efficiency"])

    # Machine Learning Forecast and Recommendation
    with st.expander(L["ml_section"], expanded=False):
        st.markdown(f"#### {L['ml_forecast']}")
        forecast_results = ml_forecast(data_dict, start_date, end_date)
        if forecast_results:
            forecast_df = pd.DataFrame.from_dict(forecast_results, orient="index", columns=["Predicted Growth Rate"])
            forecast_df.sort_values(by="Predicted Growth Rate", ascending=False, inplace=True)
            forecast_df["Predicted Growth Rate (%)"] = forecast_df["Predicted Growth Rate"] * 100
            st.dataframe(forecast_df[["Predicted Growth Rate (%)"]].style.format("{:.2f}"))
            
            best_ticker = forecast_df.index[0]
            best_growth = forecast_df.iloc[0]["Predicted Growth Rate (%)"]
            worst_ticker = forecast_df.index[-1]
            worst_growth = forecast_df.iloc[-1]["Predicted Growth Rate (%)"]

            st.success(f"**{L['recommendation_buy']} {best_ticker} - {banks_full.get(best_ticker, '')}** with a predicted growth of {best_growth:.2f}%.")
            st.error(f"**{L['recommendation_sell']} {worst_ticker} - {banks_full.get(worst_ticker, '')}** with a predicted growth of {worst_growth:.2f}%.")
        else:
            st.warning("Insufficient data for machine learning forecast.")
        
        st.info(L["note"])

    # Add credit line with updated spelling for the name
    st.markdown("<center><small>Designed by Chief Engineer Tareq Majeed alkarimi - Iraqi Ministry of Oil</small></center>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

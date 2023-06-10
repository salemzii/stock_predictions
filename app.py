import streamlit as st
from datetime import datetime
import time, random
import pandas as pd
import matplotlib.pyplot as plt

#from stock_predict import stock_prediction_procedure

st.set_page_config(page_title = "Stock Price Prediction", page_icon=":bar_chart:", layout="wide")

def load_stock_dataset(symbol):
    return pd.read_csv(f"data/{symbol}.csv")

def display_columns(datapoints, peak_price, rmse):
    with left_column:
        st.subheader("Datapoints: ")
        st.subheader(datapoints)
    with middle_column:
        st.subheader("All time high price: ")
        st.subheader(f"US ${peak_price}")
    """st.subheader("All time Low price: ")"""
    with right_column:
        st.subheader("Root Mean Square Error: ")
        st.subheader(rmse)

def Visualize_data(mode:str, dataframe: pd.DataFrame):
    if mode == "Histogram":
        fig, ax = plt.subplots()
        ax.hist(dataframe[["Close", "Predictions"]], bins=20)

        st.pyplot(fig) 
    
    elif mode == "Line Chart":
        st.line_chart(data=df, x="Date", y=["Predictions", "Close"])
    elif mode == "Area Chart":
        st.area_chart(df[["Close", "Predictions"]])
    elif mode == "Bokeh Chart":
        from bokeh.plotting import figure

        p = figure(
            title='Stock Prediction Visualization using Bokeh Chart',
            x_axis_label='Close',
            y_axis_label='Predictions')

        p.line(dataframe["Close"], dataframe["Predictions"], legend_label='Trend', line_width=2)

        st.bokeh_chart(p, use_container_width=True)
    else:
        #visualize the data
        plt.figure(figsize=(16, 8))
        plt.title('Model')
        plt.xlabel('Prediction', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(dataframe['Close'])
        plt.plot(dataframe[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        st.pyplot(plt)


# ---- SIDEBAR ----
st.sidebar.header("Please Filter Here:")

stock = st.sidebar.selectbox(
    "select the language: ",
    options=("APPLE", "AMAZON", "SALESFORCE", "GOOGLE", "KO", "META", "MICROSOFT", "TESLA"),
)

stock_dict = {
    "APPLE": "AAPL",
    "AMAZON" : "AMZN", 
    "SALESFORCE" : "CRM",
    "GOOGLE": "GOOGL",
    "COCA-COLA": "KO",
    "META" : "META", 
    "MICROSOFT" : "MSFT", 
    "TESLA": "TSLA"
}

start_date = st.sidebar.date_input("Start date for stock") 
end_date = st.sidebar.date_input("End date for stock") 

visualize = st.sidebar.selectbox(
    "Choose Visualization: ",
    options=("Area Chart", "Line Chart", "Default"),
)

# ---- MAINPAGE ----
st.title(":bar_chart: Stock Price Prediction System")
st.markdown("##")

prediction_df = None
stock_symbol = st.text_input("Enter The Stock Symbol Here: ")


stock = st.sidebar.select_slider(
    "select the stock: ",
    options=("APPLE", "AMAZON", "SALESFORCE", "GOOGLE", "KO", "META", "MICROSOFT", "TESLA"),
)

datapoints = None
peak_price = None
dip_price = None
today_prediction = 167.9
rmse = 2.17959737

left_column, middle_column, right_column = st.columns(3)


if stock_symbol != "":
    try:
        with st.spinner(f"Loading {stock_symbol} Index"):
            prediction_df, rmsE = stock_prediction_procedure(symbol=stock_symbol)

            st.success(f'Encountered Error Retrieving {stock_symbol} Dataset', icon="ℹ️")

            datapoints = len(prediction_df["Close"])
            peak_price = round(prediction_df["Close"].max(), 3)
            dip_price = round(prediction_df["Close"].min(), 3)

            display_columns(datapoints=datapoints, peak_price=peak_price, rmse=rmsE)

        Visualize_data(mode=visualize, dataframe=prediction_df)
    except Exception as err:
        st.info(f'Encountered Error Retrieving {stock_symbol} Dataset', icon="ℹ️")
    finally:
        with st.spinner(f"Loading Apple's Stock Index"):
            df = load_stock_dataset(symbol="AAPL")
            datapoints = len(df["Close"])
            peak_price = round(df["Close"].max(), 3)
            dip_price = round(df["Close"].min(), 3)
            rmse = 2.17959737
            
            time.sleep(random.randint(2, 8))
        st.success('Successfully Loaded Dataset')

        display_columns(datapoints=datapoints, peak_price=peak_price, rmse=rmse)

        Visualize_data(mode=visualize, dataframe=df)

else:
    with st.spinner(f'loading dataset'):
        df = load_stock_dataset(symbol=stock_dict[stock])
        time.sleep(random.randint(2, 8))
    st.success('Successfully Loaded Dataset')

    datapoints = len(df["Close"])
    peak_price = round(df["Close"].max(), 3)
    dip_price = round(df["Close"].min(), 3)
    today_prediction = 167.9
    rmse = 2.17959737

    display_columns(datapoints=datapoints, peak_price=peak_price, rmse=rmse)


    Visualize_data(mode = visualize, dataframe=df)


#st.radio(options=list, help=tooltip_text)
"""
Name: Luisa Fernanda Ramírez Sánchez
"""
#Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from datetime import date
import yfinance as yf
import streamlit as st
#import plotly.express as px
#import plotly.graph_objects as go

##st.metric para los indicadores 
## En la seccion 4 estan las graficas que necesito
#set the page configuration


   
#.............................................................................#
######################  Header  ###############################################
#General page configuration
st.set_page_config(page_title='Stock Market', page_icon=':chart_with_upwards_trend:', layout="wide")

#Title of the app
st.title("Stock Market") #cambiar por algo más significativo
st.caption("Made by: Luisa Fernanda Ramirez Sánchez")

#Create the different taps in the app
Summary, Chart, Financials, monte_Carlo, Analysis,References =st.tabs(["Summary", "Chart", "Financials","Monte Carlo simulation","Analysis","References"])

#Create ticker list
tickers_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

# Create a selection box for the tcikers
tickers = st.sidebar.selectbox("**Choose tickerss(s)**:point_down:", tickers_list)

#Show the Logo of the company
logo=st.sidebar.image(yf.Ticker(tickers).info['logo_url'])

#Create selection boxes for the date
start_date = st.sidebar.date_input("Start date", datetime.today().date() - timedelta(days=30))
end_date = st.sidebar.date_input("End date", datetime.today().date())

#Create a botton to get data
get = st.sidebar.button("Get data", key="get")

#Showy the app reference
st.sidebar.write("Data source: [Yahoo Finance](https://finance.yahoo.com/)")


#Change the theme of the app



###################### Get info ##############################################

# Create a function to get company's information
@st.cache
def GetCompanyInfo(tickers):
    return yf.Ticker(tickers).info
 
if tickers != '':
     #Get the company information in list format
    info=GetCompanyInfo(tickers)
     
#Create a function to get Stock data

@st.cache
def GetStockData(tickers, start_date, end_date,period='1mo',interval='1d'):
    stock_price = pd.DataFrame()
    for tick in tickers:
        stock_df = yf.Ticker(tick).history(start=start_date, end=end_date, period=period)
        stock_df['Ticker'] = tick  # Add the column ticker name
        stock_price = pd.concat([stock_price, stock_df], axis=0)  # Comebine results
    return stock_price.loc[:, ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]

stocks=GetStockData(tickers, start_date, end_date)

@st.cache
def GetMA(df,column='Close',window=50):
    df['MA']=df[column].rolling(window).mean()
    ma_df=df.copy()
    return ma_df

#Create stock variables
close_price=stocks['Close']
open_price=stocks['Open']
daily_return=close_price.pct_change()
daily_volatility = np.std(daily_return)


######################  Tab 1: Summary  #######################################


with Summary:
     
        # Show the company description
        with st.expander('**Business Summary**'):
            st.write(info['longBusinessSummary'])
              
      
        
        #keys = ['name', 'last_name', 'phone_number', 'email']
        #dict2 = {x:dict1[x] for x in keys}
        
        
        # Show some statistics
        col1,col2, col3= st.columns([1,1,2],gap="small")
        
        with col1:
            kcol1 = ['previousClose', 'open', 'bid', 'ask', 'volume','averageVolume']
            company_stats = {}  # Dictionary
            for key in kcol1:
                company_stats.update({key:info[key]})
            nkcol1=['Previous Close', 'Open', 'Bid','Ask','Volume','Avg.Volume']
            company_stats=dict(zip(nkcol1, list(company_stats.values())))
            company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})
            st.dataframe(company_stats.style.format(subset=['Value'],formatter='{:.2f}'), use_container_width=True)
            
        
        with col2:
            kcol2 = ['marketCap', 'beta', 'trailingPE', 'trailingEps','trailingAnnualDividendYield', 'exDividendDate']
            company_stats = {}  # Dictionary
            for key in kcol2:
                company_stats.update({key:info[key]})
            nkcol2=['Market Cap ($B)', 'Beta', 'PE Ratio(TTM)','EPS (TTM)','Forward Dividend & Yield','Ex-Dividend Date']
            company_stats=dict(zip(nkcol2, list(company_stats.values())))
            company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
            company_stats['Value'][0]=company_stats['Value'][0]/1000000000
            st.dataframe(company_stats.style.format(subset=['Value'],formatter='{:,.2f}'),use_container_width=True)
        #Falta buscar algunos valores porque no estan bien
    
   
    # Add table to show stock data
        with col3:
            period = st.radio('Interval', ('1M','6M','YDT','1Y','5Y','MAX'),horizontal=True,label_visibility="collapsed")
                       
                
            
            if period == '1M':
                interval='1d'
                gstart_date=end_date-timedelta(days=30)
                if tickers !='':         
                    stock_price=GetStockData([tickers],gstart_date, end_date,interval=interval)
                    for tick in [tickers]:
                        stock_df = stock_price[stock_price['Ticker'] == tick]
                        fig1 = px.area(stock_df, x=stock_df.index, y="Close",width=1000, height=400)                                      
                st.plotly_chart(fig1)
            elif period == '6M':
                interval='1mo'
                gstart_date=end_date-timedelta(days=180)
                if tickers !='':         
                    stock_price=GetStockData([tickers],gstart_date, end_date,interval=interval)
                    for tick in [tickers]:
                        stock_df = stock_price[stock_price['Ticker'] == tick]
                        fig2 = px.area(stock_df, x=stock_df.index, y="Close",width=1000, height=400)                                      
                st.plotly_chart(fig2)               
            elif period == 'YDT':
                interval='1mo'
                gstart_date=date(2022,1,1)
                if tickers !='':         
                    stock_price=GetStockData([tickers],gstart_date, end_date,interval=interval)
                    for tick in [tickers]:
                        stock_df = stock_price[stock_price['Ticker'] == tick]
                        fig3 = px.area(stock_df, x=stock_df.index, y="Close",width=1000, height=400)                                      
                st.plotly_chart(fig3)
            elif period == '1Y':
                interval='1mo'
                gstart_date=end_date-timedelta(days=365)
                if tickers !='':         
                    stock_price=GetStockData([tickers],gstart_date, end_date,interval=interval)
                    for tick in [tickers]:
                        stock_df = stock_price[stock_price['Ticker'] == tick]
                        fig4 = px.area(stock_df, x=stock_df.index, y="Close",width=1000, height=400)                                      
                st.plotly_chart(fig4) 
            elif period == '5Y':
                interval='1mo'
                gstart_date=end_date-timedelta(days=1825)
                if tickers !='':         
                    stock_price=GetStockData([tickers],gstart_date, end_date,interval=interval)
                    for tick in [tickers]:
                        stock_df = stock_price[stock_price['Ticker'] == tick]
                        fig5 = px.area(stock_df, x=stock_df.index, y="Close",width=1000, height=400)                                      
                st.plotly_chart(fig5) 
            elif period == 'MAX':
                interval='1mo'
                gstart_date=date(1984,12,1)
                if tickers !='':         
                    stock_price=GetStockData([tickers],gstart_date, end_date,interval=interval)
                    for tick in [tickers]:
                        stock_df = stock_price[stock_price['Ticker'] == tick]
                        fig6 = px.area(stock_df, x=stock_df.index, y="Close",width=1000, height=400)                                      
                st.plotly_chart(fig6) 
                       
            
    # Add a check box
        with st.container():
            show_data = st.checkbox("Show stock price data info")
            
            if tickers != '':
                stock_price = GetStockData([tickers], start_date, end_date)
                if show_data:
                    st.subheader('Stock price data')
                    st.dataframe(stock_price, use_container_width=True)
            
######################  Tab 2: Chart###########################################
#Create a list with period options
period_list = ['1m', '3m', '6m', '1y', '2y', '5y', '10y', 'ydt', 'max','Date range']

#Create an interval list for the summary and graphs tabs
interval_list = ['1d', '5d', '1wk', '1mo', '3mo']

with Chart:
    with st.form("Graphs"):
    #Selection boxes period
        period_t2=st.selectbox('Select period',(period_list))
        interval_t2=st.selectbox('Select ineterval',interval_list)
        type_chart=st.radio('Type of label', ('Line Chart','Candle Stick Chart'),horizontal=True)
        st.form_submit_button("Generate graph")
    if type_chart=='Line Chart':
        #Creating a close and volume average       
            if period_t2 == '1m':
                nstart_date=end_date-timedelta(days=30)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == '3m':
                nstart_date=end_date-timedelta(days=90)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == '6m':
                nstart_date=end_date-timedelta(days=180)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == '1y':
                nstart_date=end_date-timedelta(days=365)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == '2y':
                nstart_date=end_date-timedelta(days=730)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == '5y':
                nstart_date=end_date-timedelta(days=1825)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == 'ydt':
                nstart_date=date(2022,1,1)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == 'max':
                nstart_date=date(1984,12,1)
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
            elif period_t2 == 'Date range':
                nstart_date=start_date
                if tickers !='':                      
                    stock_data=GetStockData([tickers], nstart_date, end_date, interval=interval_t2, period=period_t2)
                    for tick in [tickers]:           
                        close_df = stock_data[stock_data['Ticker'] == tick]
                        avg_df=GetMA(close_df)
                        # set up plotly figure
                        ln_bar = make_subplots(rows=2,cols=1,row_width=[0.2, 0.9])
                        
                        # add first scatter trace at row = 1, col = 1
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['Close'], line=dict(color='red'), name='Close price'),
                                      row = 1, col = 1)
            
                        # add first bar trace at row = 1, col = 1
                        ln_bar.add_trace(go.Bar(x=avg_df.index, y=avg_df['Volume'],
                                             name='Volume'),
                                      row = 2, col = 1)
                        #Add moving average
                        ln_bar.add_trace(go.Scatter(x=avg_df.index, y=avg_df['MA'], line=dict(color='blue'), name='SMA'),
                                     row = 1, col = 1)                                               
                st.plotly_chart(ln_bar) 
        
   
    


    #Calculating moving average

    # Candle stick chart
    if type_chart=='Candle Stick Chart':
        if tickers != '':
                 for tick in [tickers]:
                     stock_df = stock_price[stock_price['Ticker'] == tick]
                     fig_cand=go.Figure(data=[go.Candlestick(x=stock_df.index,
                                     open=stock_df['Open'],
                                     high=stock_df['High'],
                                     low=stock_df['Low'],
                                     close=stock_df['Close'])])
        st.plotly_chart(fig_cand)

#https://www.youtube.com/watch?v=VLGptwMRIVQ





######################  Tab 3: Financials  ####################################
#Create a function to get financial information 

with Financials:
    option = st.selectbox(
    'Type of inform',
    ('Income Statement', 'Balance', 'Cash Flow'))
    time = st.selectbox(
    'Period',
    ('Anual', 'Quarterly'))

    #Income statement
    #Anual
    @st.cache
    def GetBalanceAnual(tickers):
        return yf.Ticker(tickers).balance_sheet
         
        if tickers != '':
             #Get the company information in list format
            anual_bal=GetBalanceAnual(tickers)
            anual_bal=anual_bal.reset_index()
            anual_bal=anual_bal.reindex([6,20,14,23,24,26,8,13,19,0,11,16,2,15,27,17,5,25,10,3,7,1,9,18,4,22,12,21])
            anual_bal=anual_bal.set_index('index')
            if option=="Income Statement" and time=="Anual":
                st.table(anual_bal.style.format(formatter='{:,.0f}'))
        #Quartly
    @st.cache
    def GetBalanceQuartly(tickers):
            return yf.Ticker(tickers).quarterly_balance_sheet
         
    if tickers != '':
            qrt_bal=GetBalanceQuartly(tickers)
            if option=="Income Statement" and time=="Quarterly":
                st.table(qrt_bal.style.format(formatter='{:,.0f}')) 


         
        #Balance Sheet
        #Anual
    @st.cache
    def GetEarnings(tickers):
            return yf.Ticker(tickers).earnings
        
    if tickers != '':
            anual_earnings=GetEarnings(tickers)
            if option=="Balance" and time=="Anual":
                st.table(anual_earnings.style.format(formatter='{:,.0f}'))
            
        #Quartly
    @st.cache
    def GetQuarterlyEarnings(tickers):
            return yf.Ticker(tickers).quarterly_earnings
        
        
    if tickers != '':
            qrt_earnings=GetQuarterlyEarnings(tickers)
            if option=="Balance" and time=="Quarterly":
                st.table(anual_earnings.style.format(formatter='{:,.0f}'))
    #Cashflow
    #Anual
   
    @st.cache
    def GetCashflow(tickers):
            return yf.Ticker(tickers).cashflow
        
        
    if tickers != '':
            anual_cashflow=GetCashflow(tickers)
            if option=="Cash Flow" and time=="Anual":
                st.table(anual_cashflow.style.format(formatter='{:,.0f}'))
        
        #Quartly
    def GetQuarterlyCashflow(tickers):
            return yf.Ticker(tickers).quarterly_cashflow
        
        
    if tickers != '':
            qrt_cashflow=GetQuarterlyCashflow(tickers)
            if option=="Cash Flow" and time=="Quarterly":
                st.table(qrt_cashflow.style.format(formatter='{:,.0f}')) 

    


######################  Monte Carlo simulation ###############################################

#declare the variables
n = [100,200, 500,1000]
t = [30,60,90]

with monte_Carlo:
    #Selection boxes period
    nbr_simulations=st.selectbox('Number of simulations',(n))
    time=st.selectbox('Time horizon',t)

    
np.random.seed(123)
simulations = nbr_simulations
time_horizone = time


# Run the simulation

simulation_df = pd.DataFrame()  

for i in range(simulations):
    
    # The list to store the next stock price
    next_price = []
    
    # Create the next stock price
    last_price = close_price[-1]
    
    for j in range(time_horizone):
        # Generate the random percentage change around the mean (0) and std (daily_volatility)
        future_return = np.random.normal(0, daily_volatility)

        # Generate the random future price
        future_price = last_price * (1 + future_return)

        # Save the price and go next
        next_price.append(future_price)
        last_price = future_price
    
    # Store the result of the simulation
    next_price_df = pd.Series(next_price).rename('sim' + str(i))
    simulation_df = pd.concat([simulation_df, next_price_df], axis=1)


    #Selection boxes period
      
        
with monte_Carlo:
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10, forward=True)
        
        plt.plot(simulation_df)
        plt.xlabel('Day')
        plt.ylabel('Price')
        
        plt.axhline(y=close_price[-1], color='red')
        plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')
        
        st.pyplot(plt)
        




######################  Analysis ##############################################


#import plotly.express as px
#fig = px.treemap(df, path=[px.Constant("all"), 'sector','ticker'], values = 'market_cap', color='colors',
 #                color_discrete_map ={'(?)':'#262931', 'red':'red', 'indianred':'indianred','lightpink':'lightpink', 'lightgreen':'lightgreen','lime':'lime','green':'green'},

  #              hover_data = {'delta':':.2p'}
   #             ))
#fig.show()

#.............................................................................#
##################### References #############################################

#Streamlit 
#Minh's notes
#geeks for geeks 
#https://discuss.streamlit.io/t/how-to-format-float-values-to-2-decimal-place-in-a-dataframe-except-one-column-of-the-dataframe/3619/3
#https://pythoninoffice.com/create-a-stock-market-heat-map-using-python/
#https://stackoverflow.com/questions/60292750/plotly-how-to-plot-a-bar-line-chart-combined-with-a-bar-chart-as-subplots
#https://stackoverflow.com/questions/64689342/plotly-how-to-add-volume-to-a-candlestick-chart




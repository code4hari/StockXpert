import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import requests

# Function definitions
def get_stock_data(ticker_symbol, period='1y'):
    stock = yf.Ticker(ticker_symbol)
    history = stock.history(period=period)
    return stock, history

def create_stock_chart(data, selected_stocks):
    fig = go.Figure()
    
    for ticker in selected_stocks:
        fig.add_trace(
            go.Scatter(
                x=data[ticker].index,
                y=data[ticker]['Close'],
                name=ticker,
                mode='lines'
            )
        )
    
    fig.update_layout(
        title="Stock Price Comparison",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        template="plotly_dark"
    )
    return fig

def display_stock_metrics(stock, history):
    info = stock.info
    
    # Basic Metrics (keep existing code)
    current_price = history['Close'].iloc[-1]
    previous_price = history['Close'].iloc[-2]
    price_change = current_price - previous_price
    price_change_percent = (price_change / previous_price) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}", f"{price_change_percent:.2f}%")
    with col2:
        st.metric("Volume", f"{history['Volume'].iloc[-1]:,.0f}")
    with col3:
        st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")

    # Valuation Metrics
    with st.expander("Valuation Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Market Cap:", f"${info.get('marketCap', 0)/1e12:.2f}T")
            st.write("Enterprise Value:", f"${info.get('enterpriseValue', 0)/1e12:.2f}T")
            st.write("Trailing P/E:", info.get('trailingPE', 'N/A'))
            st.write("Forward P/E:", info.get('forwardPE', 'N/A'))
            st.write("PEG Ratio:", info.get('pegRatio', 'N/A'))
        with col2:
            st.write("Price/Sales:", info.get('priceToSalesTrailing12Months', 'N/A'))
            st.write("Price/Book:", info.get('priceToBook', 'N/A'))
            st.write("Enterprise Value/Revenue:", info.get('enterpriseToRevenue', 'N/A'))
            st.write("Enterprise Value/EBITDA:", info.get('enterpriseToEbitda', 'N/A'))


# Financial Advisor API Call Function
def call_financial_advisor_api(query):
    try:
        response = requests.post(
            "http://localhost:5000/analyze", 
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # Clean the response text immediately when receiving it
            analysis_text = data.get('analysis', 'No response from advisor')
            # Remove literal \n\n string sequences
            analysis_text = analysis_text.replace('\\n\\n', ' ')
            # Remove actual newlines
            analysis_text = analysis_text.replace('\n\n', ' ')
            analysis_text = analysis_text.replace('\n', ' ')
            # Clean up any multiple spaces
            analysis_text = ' '.join(analysis_text.split())
            return analysis_text
        else:
            return "Error connecting to financial advisor service"
    except Exception as e:
        return f"Error: {str(e)}"

def format_advisor_response(response):
    """Clean and format the advisor's response"""
    if "TextBlock" in response:
        # Extract the actual text content between text=" and the final quotation
        start_idx = response.find('text="') + 6
        end_idx = response.rfind('"')
        response = response[start_idx:end_idx]
    
    # Clean any remaining literal \n\n sequences
    response = response.replace('\\n\\n', ' ')
    # Clean actual newlines
    response = response.replace('\n\n', ' ')
    response = response.replace('\n', ' ')
    # Clean up multiple spaces
    response = ' '.join(response.split())
    
    return response
# Page configuration
st.set_page_config(layout="wide", page_title="Investment Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Market Dashboard", "Portfolio", "Financial Advisor"])

with tab1:
    st.title("Stock Market Dashboard")
    
    # Sidebar for stock selection
    st.sidebar.header("Stock Selection")
    
    stock_input = st.sidebar.text_input("Enter stock symbols (comma-separated)", "AAPL, MSFT, GOOGL")
    selected_stocks = [symbol.strip() for symbol in stock_input.split(",")]

    period_options = {
        "1 Week": "1wk",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "1 Year": "1y",
        "5 Years": "5y"
    }
    selected_period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))

    if selected_stocks:
        try:
            stock_data = {}
            
            for symbol in selected_stocks:
                if symbol:
                    stock, history = get_stock_data(symbol, period_options[selected_period])
                    stock_data[symbol] = history
                    
                    st.subheader(f"{symbol} Metrics")
                    display_stock_metrics(stock, history)
            
            chart = create_stock_chart(stock_data, selected_stocks)
            st.plotly_chart(chart, use_container_width=True)
            
            with st.expander("View Raw Data"):
                for symbol in selected_stocks:
                    if symbol:
                        st.write(f"{symbol} Historical Data")
                        st.dataframe(stock_data[symbol])
                    
        except Exception as e:
            st.error(f"Error occurred: {str(e)}")

with tab2:
    st.title("Investment Portfolio")

    # Sample portfolio data
    portfolio_stocks = {
        'AAPL': {'shares': 150, 'avg_cost': 150.25},
        'MSFT': {'shares': 100, 'avg_cost': 245.30},
        'GOOGL': {'shares': 50, 'avg_cost': 2800.75},
        'AMZN': {'shares': 75, 'avg_cost': 3300.50},
        'NVDA': {'shares': 200, 'avg_cost': 180.25},
        'TSM': {'shares': 300, 'avg_cost': 95.50},
        'META': {'shares': 120, 'avg_cost': 280.75},
        'TSLA': {'shares': 80, 'avg_cost': 250.30},
        'JPM': {'shares': 150, 'avg_cost': 140.25},
        'V': {'shares': 100, 'avg_cost': 220.50}
    }

    # Portfolio calculations
    current_prices = {}
    total_value = 0
    total_cost = 0
    portfolio_performance = []

    for symbol in portfolio_stocks.keys():
        try:
            stock = yf.Ticker(symbol)
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            current_prices[symbol] = current_price
            
            current_value = portfolio_stocks[symbol]['shares'] * current_price
            cost_basis = portfolio_stocks[symbol]['shares'] * portfolio_stocks[symbol]['avg_cost']
            
            total_value += current_value
            total_cost += cost_basis
            
            gain_loss = current_value - cost_basis
            gain_loss_percent = (gain_loss / cost_basis) * 100
            
            portfolio_performance.append({
                'Symbol': symbol,
                'Shares': portfolio_stocks[symbol]['shares'],
                'Avg Cost': portfolio_stocks[symbol]['avg_cost'],
                'Current Price': current_price,
                'Current Value': current_value,
                'Gain/Loss': gain_loss,
                'Gain/Loss %': gain_loss_percent
            })
        except:
            st.warning(f"Could not fetch data for {symbol}")

    # Portfolio Summary
    st.header("Portfolio Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Total Cost Basis", f"${total_cost:,.2f}")
    with col3:
        total_gain_loss = total_value - total_cost
        st.metric("Total Gain/Loss", f"${total_gain_loss:,.2f}")
    with col4:
        total_gain_loss_percent = (total_gain_loss / total_cost) * 100
        st.metric("Total Return", f"{total_gain_loss_percent:.2f}%")

    # Portfolio Allocation Chart
    st.subheader("Portfolio Allocation")
    allocation_fig = go.Figure(data=[go.Pie(
        labels=[p['Symbol'] for p in portfolio_performance],
        values=[p['Current Value'] for p in portfolio_performance],
        hole=.3
    )])
    allocation_fig.update_layout(height=400)
    st.plotly_chart(allocation_fig, use_container_width=True)

    # Portfolio Performance Table
    st.subheader("Holdings Detail")
    df_portfolio = pd.DataFrame(portfolio_performance)
    df_portfolio = df_portfolio.round(2)
    st.dataframe(df_portfolio, use_container_width=True)

    # Historical Performance Chart
    st.subheader("Historical Portfolio Performance")
    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    base_value = total_value * 0.7
    historical_values = [base_value]
    
    for i in range(1, len(dates)):
        change = random.uniform(-0.02, 0.02)
        historical_values.append(historical_values[-1] * (1 + change))

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(
        x=dates,
        y=historical_values,
        mode='lines',
        name='Portfolio Value'
    ))
    hist_fig.update_layout(
        title="Historical Portfolio Value",
        xaxis_title="Date",
        yaxis_title="Value (USD)",
        height=500,
        template="plotly_dark"
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    # Additional Portfolio Metrics
    with st.expander("Portfolio Metrics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Number of Holdings:", len(portfolio_stocks))
            st.write("Most Valuable Holding:", max(portfolio_performance, key=lambda x: x['Current Value'])['Symbol'])
            st.write("Best Performer:", max(portfolio_performance, key=lambda x: x['Gain/Loss %'])['Symbol'])
            st.write("Average Position Size:", f"${total_value/len(portfolio_stocks):,.2f}")
        with col2:
            st.write("Portfolio Beta:", round(random.uniform(0.8, 1.2), 2))
            st.write("Sharpe Ratio:", round(random.uniform(1.5, 2.5), 2))
            st.write("Dividend Yield:", f"{random.uniform(1.5, 3.5):.2f}%")
            st.write("Worst Performer:", min(portfolio_performance, key=lambda x: x['Gain/Loss %'])['Symbol'])

    # Sector Allocation
    with st.expander("Sector Allocation"):
        sectors = {
            'Technology': 35,
            'Financial Services': 20,
            'Healthcare': 15,
            'Consumer Cyclical': 10,
            'Communication Services': 8,
            'Industrials': 7,
            'Consumer Defensive': 5
        }
        
        sector_fig = go.Figure(data=[go.Bar(
            x=list(sectors.keys()),
            y=list(sectors.values())
        )])
        sector_fig.update_layout(
            title="Sector Allocation (%)",
            xaxis_title="Sector",
            yaxis_title="Allocation (%)",
            height=400
        )
        st.plotly_chart(sector_fig, use_container_width=True)

with tab3:
    st.title("🤖 Investment Advisor AI")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Advisor:** {message['content']}")
            st.markdown("---")
    
    # Input area
    with st.form(key='advisor_input_form'):
        user_input = st.text_input("Ask your financial question:", key="advisor_input")
        submit_button = st.form_submit_button("Send")
    
    # Process user input
    if submit_button and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            'role': 'user', 
            'content': user_input
        })
        
        # Show loading spinner
        with st.spinner('Generating response...'):
            # Call API
            advisor_response = call_financial_advisor_api(user_input)
        
        # Format response
        formatted_response = format_advisor_response(advisor_response)
        
        # Add advisor response to chat history
        st.session_state.chat_history.append({
            'role': 'advisor', 
            'content': formatted_response
        })
        
        # Rerun to update the chat
        st.rerun()
    
    # Clear chat button
    if st.button('Clear Chat'):
        st.session_state.chat_history = []
        st.rerun()

# Add custom CSS for chat styling
st.markdown("""
<style>
.stContainer {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 20px;
}
.stMarkdown {
    font-size: 14px;
    line-height: 1.6;
}
.stTextInput > div > div > input {
    border-radius: 20px;
    padding: 10px 15px;
}
</style>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    st.runtime.scriptrunner.add_script_run_ctx()
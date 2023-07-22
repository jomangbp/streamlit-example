import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para descargar los datos y realizar las simulaciones
def simulate(ticker_symbol, start_date, end_date, model, num_simulations):
    # Descargar los datos históricos
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    close_prices = data['Close']
    ret = np.log(1+close_prices.pct_change())

    mean = ret.mean()
    std = ret.std()

    # Obtener el último precio de cierre como el precio de inicio
    starting_stock_price = close_prices[-1]

    if model == "Monte Carlo":
        simulations_mc = []
        for i in range(num_simulations):
            simulated_data = np.random.normal(mean, std, ret.shape[0])
            sim_stock_price = starting_stock_price * (simulated_data + 1).cumprod()
            df_mc = pd.DataFrame(sim_stock_price, columns=['Price'])
            simulations_mc.append(df_mc)
        return simulations_mc

    elif model == "GBM":
        n = 1000  # Number of intervals
        T = 4  # Time in years

        # Calculate each time step
        dt = T / n

        # Simulation using numpy arrays
        np.random.seed(42)
        St = np.exp((mean - std ** 2 / 2) * dt + std * np.random.normal(0, np.sqrt(dt), size=(num_simulations, n)).T)
        St = starting_stock_price * St.cumprod(axis=0)
        
        simulations_gbm = []
        for i in range(num_simulations):
            df = pd.DataFrame(St[:, i], columns=['Price'])
            simulations_gbm.append(df)
        return simulations_gbm

    elif model == "Heston":
        kappa = 2  # Mean reversion speed of variance
        theta = std ** 2  # Long-term average variance
        sigma = std  # Volatility of volatility
        rho = -0.5  # Correlation between the stock price and its volatility
        r = 0.05  # Risk-free interest rate
        T = 1  # Time to maturity (in years)
        N = 860  # Number of time steps
        dt = T / N  # Time increment

        simulations_hm = []
        for i in range(num_simulations):
            V = np.zeros(N+1)
            V[0] = theta
            for t in range(1, N+1):
                dZ1 = np.random.normal(0, np.sqrt(dt))
                dZ2 = rho * dZ1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
                V[t] = V[t-1] + kappa * (theta - V[t-1]) * dt + sigma * np.sqrt(V[t-1]) * dZ1

            S = np.zeros(N+1)
            S[0] = starting_stock_price
            for t in range(1, N+1):
                dW = np.random.normal(0, np.sqrt(dt))
                S[t] = S[t-1] * np.exp((r - 0.5 * V[t]) * dt + np.sqrt(V[t]) * dW)

            df_heston = pd.DataFrame(S, columns=['Price'])
            simulations_hm.append(df_heston)
        return simulations_hm

# Función para graficar las simulaciones
def plot_simulations(simulations):
    for i in range(len(simulations)):
        plt.plot(simulations[i])
    st.pyplot()

# Utiliza streamlit para crear la UI
st.title("Stock Price Simulation")

# Campo de entrada para el símbolo de ticker
ticker_symbol = st.text_input("Enter ticker symbol:", value="AAPL")

# Campos de entrada para las fechas de inicio y fin
start_date = st.date_input("Start date:", value=pd.to_datetime("2008-01-01"))
end_date = st.date_input("End date:", value=pd.to_datetime("2011-01-01"))

# Campo de entrada para el modelo de simulación
model = st.selectbox("Select simulation model:", options=["Monte Carlo", "GBM", "Heston"])

# Campo de entrada para el número de simulaciones
num_simulations = st.number_input("Number of simulations:", min_value=1, max_value=1000, value=100)

# Cuando se presiona el botón, realiza la simulación y muestra el resultado
if st.button("Simulate"):
    simulations = simulate(ticker_symbol, start_date, end_date, model, num_simulations)
    plot_simulations(simulations)

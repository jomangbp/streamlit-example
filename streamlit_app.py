import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para descargar los datos y realizar las simulaciones
def simulate(ticker_symbol, start_date, end_date, model, num_simulations=1000, kappa=2, theta=0.01, sigma=0.1, rho=-0.5, r=0.05):
    # Descargar los datos históricos
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    close_prices = data['Close']

    # Calcular los retornos logarítmicos
    ret = np.log(1 + close_prices.pct_change())
    
    mean = ret.mean()
    std = ret.std()

    # Obtener el último precio de cierre como el precio de inicio
    starting_stock_price = close_prices[-1]

    # Descargar los datos de dividendos
    dividends = yf.Ticker(ticker_symbol).dividends

    # Asegúrate de que ambos índices de tiempo sean conscientes de la zona horaria
    dividends.index = dividends.index.tz_localize(None)
    close_prices.index = close_prices.index.tz_localize(None)

    # Calcular el rendimiento de los dividendos
    dividend_yield = dividends / close_prices

    # Rellenar los valores faltantes en dividend_yield con 0
    dividend_yield = dividend_yield.fillna(0)

    # Calcular los retornos teniendo en cuenta los dividendos y la tasa de interés
    interest_rate = 0.01  # Asumiendo una tasa de interés del 1%
    ret = np.log(1 + close_prices.pct_change() + dividend_yield) - interest_rate

    if model == "Heston":
        n = 1000  # Number of intervals
        T = 4  # Time in years
        M = num_simulations  # Number of simulations

        # Calculate each time step
        dt = T / n

        # Simulation using numpy arrays
        np.random.seed(42)
        St = np.exp((mean - std ** 2 / 2) * dt + std * np.random.normal(0, np.sqrt(dt), size=(M, n)).T)
        St = starting_stock_price * St.cumprod(axis=0)
        column_names = ['Simulation_{}'.format(i) for i in range(M)]
        simulations_gbm = pd.DataFrame(St, columns=column_names)

        return [simulations_gbm]

    elif model == "Heston":
        N = 860  # Number of time steps
        T = 1  # Time to maturity (in years)
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

# Utiliza streamlit para crear la UI
st.title("Stock Price Simulation")

# Campo de entrada para el símbolo de ticker
ticker_symbol = st.text_input("Enter ticker symbol:")

# Campos de entrada para las fechas de inicio y fin
start_date = st.date_input("Start date:")
end_date = st.date_input("End date:")

# Campo de entrada para el número de simulaciones
num_simulations = st.number_input("Number of simulations:", min_value=100, max_value=10000)

# Campo de entrada para el modelo de simulación
model = st.selectbox("Select simulation model:", options=["GBM", "Heston"])

# Si el usuario selecciona el modelo de Heston, muestra campos de entrada adicionales para los parámetros del modelo
if model == "Heston":
    kappa = st.number_input("Enter kappa (mean reversion speed of variance):", min_value=0.0, value=2.0)
    theta = st.number_input("Enter theta (long-term average variance):", min_value=0.0, value=0.01)
    sigma = st.number_input("Enter sigma (volatility of volatility):", min_value=0.0, value=0.1)
    rho = st.number_input("Enter rho (correlation between the stock price and its volatility):", min_value=-1.0, max_value=1.0, value=-0.5)
    r = st.number_input("Enter r (risk-free interest rate):", min_value=0.0, value=0.05)

# Cuando se presiona el botón, realiza la simulación y muestra el resultado
if st.button("Simulate"):
    simulations = simulate(ticker_symbol, start_date, end_date, model, num_simulations, kappa=kappa, theta=theta, sigma=sigma, rho=rho, r=r)
        
    # Concatenar todas las simulaciones en un solo DataFrame
    all_simulations = pd.concat(simulations, axis=1)

    # Crear el gráfico de líneas con todas las simulaciones
    st.subheader("Line chart of all simulations")
    st.line_chart(all_simulations)

    # Crear el gráfico de pastel de los resultados finales de las simulaciones
    st.subheader("Pie chart of final simulation results")
    final_results = all_simulations.iloc[-1]

    # Define los rangos de precios basados en los precios de las acciones
    min_price = final_results.min()
    max_price = final_results.max()
    bins = np.linspace(min_price, max_price, 11)  # Crea 10 rangos de precios igualmente espaciados
    names = ['{0:.2f}-{1:.2f}'.format(bins[i], bins[i+1]) for i in range(len(bins)-1)]  # Crea los nombres de los rangos

    # Agrupa los resultados finales en los rangos de precios
    final_results_grouped = pd.cut(final_results, bins, labels=names).value_counts()

    plt.figure(figsize=(10,6))
    plt.pie(final_results_grouped, labels=final_results_grouped.index, autopct='%1.1f%%')
    plt.title('Pie chart of final simulation results')
    st.pyplot(plt)

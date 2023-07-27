import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función para descargar los datos y realizar las simulaciones
def simulate(ticker_symbol, start_date, end_date, model, num_simulations=1000):
    # Descargar los datos históricos
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    close_prices = data['Close']

    # Calcular los retornos logarítmicos
    ret = np.log(1 + close_prices.pct_change())
    
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
        kappa = 2  # Mean reversion speed of variance
        theta = std ** 2  # Long-term average variance
        sigma = std  # Volatility of volatility
        rho = -0.5  # Correlation between the stock price and its volatility
        r = 0.01  # Risk-free interest rate
        T = 1  # Time to maturity (in years)
        N = 860  # Number of time steps
        dt = T / N  # Time increment
        num_simulations = num_simulations  # Number of simulations

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
model = st.selectbox("Select simulation model:", options=["Monte Carlo", "GBM", "Heston"])

# Cuando se presiona el botón, realiza la simulación y muestra el resultado
if st.button("Simulate"):
    # Descargar los datos históricos
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Realizar la simulación en los datos de entrenamiento
    simulations = simulate(ticker_symbol, train_data.index[0], train_data.index[-1], model, num_simulations)

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

    # Guardar las simulaciones en el estado de la sesión
    if 'simulations' not in st.session_state:
        st.session_state['simulations'] = simulations

    # Realizar el backtesting comparando los resultados de la simulación con los datos de prueba
    st.subheader("Backtesting results")
    simulation_index = st.slider("Select a simulation:", 0, len(simulations)-1, 0)
    simulation = st.session_state['simulations'][simulation_index]
    st.line_chart(pd.DataFrame({'Simulation': simulation['Price'], 'Real': test_data['Close']}))

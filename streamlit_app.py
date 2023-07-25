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
    dividends = data['Dividends']
    interest_rate = data['Interest Rate']  # Asumiendo que los datos de la tasa de interés están disponibles

    # Calcular los retornos logarítmicos
    ret = np.log(1 + close_prices.pct_change() + dividends) - interest_rate
    
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

    
    if model == "Heston":
        # Parámetros del modelo Heston
        kappa = 2  # Mean reversion speed of variance
        theta = std ** 2  # Long-term average variance
        sigma = std  # Volatility of volatility
        rho = -0.5  # Correlation between the stock price and its volatility
        v0 = std ** 2  # Initial variance

        # Parámetros de la opción
        r = 0.05  # Risk-free interest rate
        T = 1  # Time to maturity (in years)
        S0 = starting_stock_price  # Initial stock price
        K = S0  # Strike price

        # Crear el proceso de Heston
        day_count = ql.Actual365Fixed()
        calendar = ql.UnitedStates()
        ql.Settings.instance().evaluationDate = ql.Date(1, 1, 2020)

        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql.Settings.instance().evaluationDate, r, day_count))
        dividend_yield = ql.YieldTermStructureHandle(ql.FlatForward(ql.Settings.instance().evaluationDate, 0.0, day_count))
        heston_process = ql.HestonProcess(flat_ts, dividend_yield, spot_handle, v0, kappa, theta, sigma, rho)

        # Crear el generador de números aleatorios
        rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(2, ql.UniformRandomGenerator()))
        seq = ql.GaussianPathGenerator(heston_process, T, num_simulations, rng, False)

        # Simular los precios de las acciones
        simulations_hm = []
        for i in range(num_simulations):
            sample_path = seq.next()
            path = sample_path.value()
            simulated_stock_prices = [path[j][0] for j in range(len(path))]
            df_heston = pd.DataFrame(simulated_stock_prices, columns=['Price'])
            simulations_hm.append(df_heston)

        return simulations_hm
        
    elif model == "Markov":
        data["daily_return"] = data["Adj Close"].pct_change()
        data["state"] = np.where(data["daily_return"] >= 0, "up", "down")

        up_counts = len(data[data["state"] == "up"])
        down_counts = len(data[data["state"] == "down"])
        up_to_up = len(data[(data["state"] == "up") & (data["state"].shift(-1) == "up") ]) / len(data.query('state=="up"'))
        down_to_up = len(data[(data["state"] == "up") & (data["state"].shift(-1) == "down")]) / len(data.query('state=="up"'))
        up_to_down = len(data[(data["state"] == "down") & (data["state"].shift(-1) == "up")]) / len(data.query('state=="down"'))
        down_to_down = len(data[(data["state"] == "down") & (data["state"].shift(-1) == "down")]) / len(data.query('state=="down"'))
        transition_matrix = pd.DataFrame({
         "up": [up_to_up, up_to_down],
         "down": [down_to_up, down_to_down]
         }, index=["up", "down"])
        return [transition_matrix]
        
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
model = st.selectbox("Select simulation model:", options=["Monte Carlo", "GBM", "Heston", "Markov"])

# Cuando se presiona el botón, realiza la simulación y muestra el resultado
if st.button("Simulate"):
    simulations = simulate(ticker_symbol, start_date, end_date, model, num_simulations)

    if model != "Markov":
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
    else:
        st.write(simulations)


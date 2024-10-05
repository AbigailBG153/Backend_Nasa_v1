from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import geocoder
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.experimental import enable_iterative_imputer  # Habilitar IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Inicializar la aplicación FastAPI
app = FastAPI()

# Parámetros de predicción
horas_prediccion = 6  # Horas para predecir (en intervalos de 2 horas)

# Definir un modelo de datos para el cuerpo de la solicitud
class PrediccionRequest(BaseModel):
    lat: float = None
    lon: float = None
    use_current_location: bool = False
    variable: str = 'T2M'

# Función para obtener la predicción y calcular la precisión para una variable
def obtener_prediccion(lat, lon, variable, variable_name, unidad, horas_prediccion=6):
    # Definir el rango de fechas (último año para datos horarios)
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')  # NASA solo da datos hasta ayer
    one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')  # Hace 1 año (365 días)

    # Definir la URL de la NASA POWER API para obtener los datos horarios de la variable seleccionada
    url = f'https://power.larc.nasa.gov/api/temporal/hourly/point?parameters={variable}&community=AG&longitude={lon}&latitude={lat}&format=JSON&start={one_year_ago}&end={end_date}'
    
    # Hacer la solicitud a la API
    response = requests.get(url)
    data = response.json()

    # Procesar los datos si existen
    if 'properties' in data:
        # Obtener los datos de la variable
        if variable == 'PRECTOT':
            valores = data['properties']['parameter']['PRECTOTCORR']
        else:
            valores = data['properties']['parameter'][variable]
        
        # Convertir los datos a una serie temporal
        dates = list(valores.keys())
        values = list(valores.values())

        # Crear un DataFrame
        df = pd.DataFrame({
            'Fecha': dates, 
            'Valor': values
        })

        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y%m%d%H')
        df.set_index('Fecha', inplace=True)
        df.replace(-999.0, np.nan, inplace=True)  # Manejar valores faltantes

        # Usar Iterative Imputation para valores faltantes
        iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
        df = pd.DataFrame(iterative_imputer.fit_transform(df), columns=df.columns, index=df.index)

        # ----------- MODELO: PROPHET ------------
        df_prophet = df.reset_index().rename(columns={'Fecha': 'ds', 'y': 'Valor'})
        model_prophet = Prophet()
        model_prophet.fit(df_prophet)

        future_dates = model_prophet.make_future_dataframe(periods=horas_prediccion*2, freq='2H')
        forecast_prophet = model_prophet.predict(future_dates)

        # Evaluar la precisión del modelo
        y_true = df_prophet['y']
        y_pred = forecast_prophet['yhat'][:len(y_true)]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Extraer las predicciones para las próximas horas
        forecast_prophet = forecast_prophet[['ds', 'yhat']].tail(horas_prediccion)
        prediccion_fechas = forecast_prophet['ds'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        prediccion_valores = forecast_prophet['yhat'].round(2).tolist()

        resultados = {
            'variable': variable_name,
            'unidad': unidad,
            'predicciones': [
                {'fecha': fecha, 'valor': f"{valor}{unidad}"}
                for fecha, valor in zip(prediccion_fechas, prediccion_valores)
            ],
            'rmse': round(rmse, 2),
            'mae': round(mae, 2)
        }
        return resultados
    else:
        raise HTTPException(status_code=404, detail=f"No se encontraron datos para la variable {variable_name}.")

# Ruta para la predicción
@app.post("/prediccion")
def prediccion(request: PrediccionRequest):
    if request.use_current_location:
        g = geocoder.ip('me')
        lat, lon = g.latlng
    else:
        lat, lon = request.lat, request.lon

    variable = request.variable
    variables = {
        'T2M': ('Temperatura Media', '°C'),
        'PRECTOT': ('Precipitación', 'mm'),
        'RH2M': ('Humedad Relativa', '%'),
        'ALLSKY_SFC_SW_DWN': ('Radiación Solar', 'W/m²'),
        'WS2M': ('Velocidad del Viento', 'm/s')
    }

    if variable in variables:
        variable_name, unidad = variables[variable]
        resultado = obtener_prediccion(lat, lon, variable, variable_name, unidad)
        return resultado
    else:
        raise HTTPException(status_code=400, detail="Variable no soportada.")

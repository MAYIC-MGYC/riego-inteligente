import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Predicción de Riego",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
    }
    .prediction-result {
        background: linear-gradient(90deg, #90EE90, #98FB98);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .realtime-card {
        background: linear-gradient(90deg, #FFE4B5, #F0E68C);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF8C00;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Función para obtener datos en tiempo real
@st.cache_data(ttl=300)  # Cache por 5 minutos
def obtener_datos_tiempo_real():
    """
    Obtiene datos climáticos en tiempo real de Huánuco usando Open-Meteo API
    """
    try:
        # Coordenadas de Huánuco
        lat, lon = -9.9306, -76.2422
        
        # URL de Open-Meteo (datos actuales y pronóstico de hoy)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation&daily=temperature_2m_max,temperature_2m_min&timezone=America/Lima&forecast_days=1"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extraer valores actuales
        current = data['current']
        daily = data['daily']
        
        datos_actuales = {
            'temp_media': current['temperature_2m'],
            'humedad_media': current['relative_humidity_2m'],
            'precipitacion': current['precipitation'],
            'temp_max': daily['temperature_2m_max'][0] if daily['temperature_2m_max'] else current['temperature_2m'],
            'temp_min': daily['temperature_2m_min'][0] if daily['temperature_2m_min'] else current['temperature_2m'],
            'timestamp': current['time'],
            'status': 'online'
        }
        
        return datos_actuales
        
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ Error de conexión: {str(e)}")
        return {'status': 'offline', 'error': str(e)}
    except Exception as e:
        st.error(f"⚠️ Error obteniendo datos: {str(e)}")
        return {'status': 'offline', 'error': str(e)}

# Título principal
st.markdown("""
<div class="main-header">
    <h1>💧 Sistema Inteligente de Predicción de Riego Agrícola</h1>
    <p>Predicción basada en Machine Learning con datos reales de Huánuco, Perú</p>
    <p>🌐 <strong>Ahora con datos en tiempo real!</strong></p>
</div>
""", unsafe_allow_html=True)

# Función para cargar el modelo
@st.cache_resource
def cargar_modelo():
    if os.path.exists("modelo_riego.pkl"):
        return joblib.load("modelo_riego.pkl")
    else:
        st.error("❌ No se encontró el archivo 'modelo_riego.pkl'")
        st.info("📝 Asegúrate de que el archivo del modelo entrenado esté en el mismo directorio")
        return None

# Función para cargar datos históricos
@st.cache_data
def cargar_datos():
    if os.path.exists("riego_historico_huanuco.csv"):
        df = pd.read_csv("riego_historico_huanuco.csv")
        # Convertir fecha a datetime si no lo está
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    else:
        st.error("❌ No se encontró el archivo 'riego_historico_huanuco.csv'")
        st.info("📝 Asegúrate de que el archivo CSV esté en el mismo directorio")
        return None

# Cargar modelo y datos
modelo = cargar_modelo()
df = cargar_datos()

# Verificar que ambos archivos estén disponibles
if modelo is None or df is None:
    st.stop()

# Obtener datos en tiempo real
datos_tiempo_real = obtener_datos_tiempo_real()

# Sidebar con información del dataset
st.sidebar.header("📊 Información del Dataset")
st.sidebar.info(f"""
**Total de registros:** {len(df)}
**Período:** {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}
**Variables:** {', '.join(df.columns.tolist())}
""")

# Estado de conexión en tiempo real
st.sidebar.header("🌐 Estado de Conexión")
if datos_tiempo_real['status'] == 'online':
    st.sidebar.markdown('<p class="status-online">🟢 Conectado - Datos en tiempo real</p>', unsafe_allow_html=True)
    st.sidebar.info(f"📡 Última actualización: {datetime.now().strftime('%H:%M:%S')}")
else:
    st.sidebar.markdown('<p class="status-offline">🔴 Desconectado - Usando datos históricos</p>', unsafe_allow_html=True)
    st.sidebar.error("⚠️ No se pueden obtener datos actuales")

# Estadísticas generales en sidebar
st.sidebar.header("📈 Estadísticas Generales")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Temp. Promedio", f"{df['temp_media'].mean():.1f}°C")
    st.metric("Humedad Promedio", f"{df['humedad_media'].mean():.1f}%")
with col2:
    st.metric("Precipitación Total", f"{df['precipitacion'].sum():.1f}mm")
    st.metric("Riego Promedio", f"{df['riego_estimado'].mean():.2f}L/m²")

# Mostrar datos en tiempo real si están disponibles
if datos_tiempo_real['status'] == 'online':
    st.markdown("""
    <div class="realtime-card">
        <h3>🌡️ Condiciones Climáticas Actuales en Huánuco</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "🌡️ Temperatura", 
            f"{datos_tiempo_real['temp_media']:.1f}°C",
            delta=f"{datos_tiempo_real['temp_media'] - df['temp_media'].mean():.1f}°C vs promedio"
        )
    with col2:
        st.metric(
            "💧 Humedad", 
            f"{datos_tiempo_real['humedad_media']:.0f}%",
            delta=f"{datos_tiempo_real['humedad_media'] - df['humedad_media'].mean():.0f}% vs promedio"
        )
    with col3:
        st.metric(
            "🌧️ Precipitación", 
            f"{datos_tiempo_real['precipitacion']:.1f}mm",
            delta=f"{datos_tiempo_real['precipitacion'] - df['precipitacion'].mean():.1f}mm vs promedio"
        )
    with col4:
        # Predicción automática con datos actuales
        if 'temp_media' in datos_tiempo_real:
            datos_prediccion = pd.DataFrame({
                'temp_media': [datos_tiempo_real['temp_media']],
                'humedad_media': [datos_tiempo_real['humedad_media']],
                'precipitacion': [datos_tiempo_real['precipitacion']]
            })
            
            try:
                prediccion_actual = modelo.predict(datos_prediccion)[0]
                st.metric(
                    "🚰 Riego Recomendado", 
                    f"{prediccion_actual:.2f}L/m²",
                    delta="Tiempo Real"
                )
            except:
                st.metric("🚰 Riego Recomendado", "Error", delta="No disponible")

# Layout principal con columnas
col1, col2 = st.columns([2, 1])

with col1:
    # Gráficos de datos históricos
    st.subheader("📈 Análisis de Datos Históricos")
    
    # Pestañas para diferentes visualizaciones
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tendencia de Riego", "Variables Climáticas", "Correlaciones", "Datos Recientes", "Comparación Tiempo Real"])
    
    with tab1:
        # Gráfico de tendencia de riego
        fig_riego = px.line(df, x='fecha', y='riego_estimado', 
                           title='Evolución del Riego Estimado',
                           labels={'riego_estimado': 'Riego Estimado (L/m²)', 'fecha': 'Fecha'})
        fig_riego.update_traces(line_color='#2E8B57', line_width=3)
        fig_riego.update_layout(height=400)
        st.plotly_chart(fig_riego, use_container_width=True)
    
    with tab2:
        # Gráfico de variables climáticas
        fig_clima = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperatura Media', 'Humedad Media', 'Precipitación', 'Temp. Max vs Min'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Temperatura media
        fig_clima.add_trace(
            go.Scatter(x=df['fecha'], y=df['temp_media'], name='Temp. Media', line=dict(color='orange')),
            row=1, col=1
        )
        
        # Humedad media
        fig_clima.add_trace(
            go.Scatter(x=df['fecha'], y=df['humedad_media'], name='Humedad', line=dict(color='blue')),
            row=1, col=2
        )
        
        # Precipitación
        fig_clima.add_trace(
            go.Bar(x=df['fecha'], y=df['precipitacion'], name='Precipitación', marker_color='lightblue'),
            row=2, col=1
        )
        
        # Temp max vs min
        if 'temp_max' in df.columns and 'temp_min' in df.columns:
            fig_clima.add_trace(
                go.Scatter(x=df['fecha'], y=df['temp_max'], name='Temp. Máx', line=dict(color='red')),
                row=2, col=2
            )
            fig_clima.add_trace(
                go.Scatter(x=df['fecha'], y=df['temp_min'], name='Temp. Mín', line=dict(color='lightcoral')),
                row=2, col=2
            )
        
        fig_clima.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_clima, use_container_width=True)
    
    with tab3:
        # Matriz de correlación
        st.subheader("🔗 Matriz de Correlación")
        variables_numericas = ['temp_media', 'humedad_media', 'precipitacion', 'riego_estimado']
        corr_matrix = df[variables_numericas].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True, 
                            aspect="auto",
                            title="Correlación entre Variables",
                            color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        # Mostrar datos más recientes
        st.subheader("📅 Datos de los Últimos 10 Días")
        datos_recientes = df.tail(10).copy()
        if 'fecha' in datos_recientes.columns:
            datos_recientes['fecha'] = datos_recientes['fecha'].dt.strftime('%Y-%m-%d')
        st.dataframe(datos_recientes, use_container_width=True)
    
    with tab5:
        # Comparación con datos en tiempo real
        if datos_tiempo_real['status'] == 'online':
            st.subheader("🔄 Comparación: Datos Actuales vs Históricos")
            
            # Crear DataFrame para comparación
            comparacion_data = {
                'Variable': ['Temperatura (°C)', 'Humedad (%)', 'Precipitación (mm)'],
                'Actual': [
                    datos_tiempo_real['temp_media'],
                    datos_tiempo_real['humedad_media'],
                    datos_tiempo_real['precipitacion']
                ],
                'Promedio Histórico': [
                    df['temp_media'].mean(),
                    df['humedad_media'].mean(),
                    df['precipitacion'].mean()
                ]
            }
            
            df_comp = pd.DataFrame(comparacion_data)
            df_comp['Diferencia'] = df_comp['Actual'] - df_comp['Promedio Histórico']
            
            # Gráfico de barras comparativo
            fig_comp = px.bar(df_comp.melt(id_vars='Variable', value_vars=['Actual', 'Promedio Histórico']),
                             x='Variable', y='value', color='variable',
                             title='Condiciones Actuales vs Promedio Histórico',
                             barmode='group')
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Tabla de comparación
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.warning("⚠️ No hay datos en tiempo real disponibles para comparar")

with col2:
    # Sección de predicción
    st.subheader("🔮 Predicción de Riego")
    
    # Opción para usar datos en tiempo real o manuales
    modo_prediccion = st.radio(
        "Selecciona el modo de predicción:",
        ["🌐 Usar datos en tiempo real", "✋ Ingresar datos manualmente"],
        disabled=(datos_tiempo_real['status'] != 'online')
    )
    
    if modo_prediccion == "🌐 Usar datos en tiempo real" and datos_tiempo_real['status'] == 'online':
        # Predicción automática con datos actuales
        st.info("🔄 Usando datos climáticos actuales de Huánuco")
        
        # Mostrar los datos que se están usando
        st.write("**Datos actuales:**")
        st.write(f"🌡️ Temperatura: {datos_tiempo_real['temp_media']:.1f}°C")
        st.write(f"💧 Humedad: {datos_tiempo_real['humedad_media']:.0f}%")
        st.write(f"🌧️ Precipitación: {datos_tiempo_real['precipitacion']:.1f}mm")
        
        # Botón para actualizar predicción
        if st.button("🔄 Actualizar Predicción", use_container_width=True):
            # Limpiar cache para obtener datos frescos
            st.cache_data.clear()
            st.rerun()
        
        # Realizar predicción automática
        datos_prediccion = pd.DataFrame({
            'temp_media': [datos_tiempo_real['temp_media']],
            'humedad_media': [datos_tiempo_real['humedad_media']],
            'precipitacion': [datos_tiempo_real['precipitacion']]
        })
        
        try:
            prediccion = modelo.predict(datos_prediccion)[0]
            
            # Mostrar resultado
            st.markdown(f"""
            <div class="prediction-result">
                🌿 <strong>Riego Recomendado (Tiempo Real):</strong><br>
                <span style="font-size: 2rem; color: #2E8B57;">{prediccion:.2f} L/m²</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretación del resultado
            if prediccion < df['riego_estimado'].quantile(0.33):
                interpretacion = "💚 **Riego Bajo** - Condiciones favorables"
            elif prediccion < df['riego_estimado'].quantile(0.66):
                interpretacion = "💙 **Riego Moderado** - Condiciones normales"
            else:
                interpretacion = "💪 **Riego Alto** - Condiciones secas"
            
            st.markdown(interpretacion)
            
            # Comparación con históricos
            percentil = (df['riego_estimado'] <= prediccion).mean() * 100
            st.info(f"📊 Esta predicción está en el percentil {percentil:.0f}% de los datos históricos")
            
            # Mostrar timestamp de los datos
            st.caption(f"📅 Datos obtenidos: {datos_tiempo_real['timestamp']}")
            
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
    
    else:
        # Predicción manual (modo original)
        with st.form("prediccion_form"):
            st.write("**Ingresa los parámetros climáticos:**")
            
            # Obtener rangos de los datos históricos para los sliders
            temp_min, temp_max = float(df['temp_media'].min()), float(df['temp_media'].max())
            hum_min, hum_max = float(df['humedad_media'].min()), float(df['humedad_media'].max())
            prec_min, prec_max = 0.0, float(df['precipitacion'].max())
            
            # Valores por defecto (usar datos en tiempo real si están disponibles)
            temp_default = datos_tiempo_real.get('temp_media', df['temp_media'].mean()) if datos_tiempo_real['status'] == 'online' else df['temp_media'].mean()
            hum_default = datos_tiempo_real.get('humedad_media', df['humedad_media'].mean()) if datos_tiempo_real['status'] == 'online' else df['humedad_media'].mean()
            prec_default = datos_tiempo_real.get('precipitacion', df['precipitacion'].mean()) if datos_tiempo_real['status'] == 'online' else df['precipitacion'].mean()
            
            temperatura = st.slider(
                "🌡️ Temperatura Media (°C)", 
                min_value=temp_min-5, 
                max_value=temp_max+5, 
                value=float(temp_default),
                step=0.1
            )
            
            humedad = st.slider(
                "💧 Humedad Relativa (%)", 
                min_value=max(0.0, hum_min-10), 
                max_value=min(100.0, hum_max+10), 
                value=float(hum_default),
                step=1.0
            )
            
            precipitacion = st.slider(
                "🌧️ Precipitación (mm)", 
                min_value=prec_min, 
                max_value=prec_max+10, 
                value=float(prec_default),
                step=0.1
            )
            
            # Botón de predicción
            submitted = st.form_submit_button("🚀 Predecir Necesidad de Riego", use_container_width=True)
            
            if submitted:
                # Preparar datos para predicción
                datos_nuevos = pd.DataFrame({
                    'temp_media': [temperatura],
                    'humedad_media': [humedad],
                    'precipitacion': [precipitacion]
                })
                
                # Realizar predicción
                try:
                    prediccion = modelo.predict(datos_nuevos)[0]
                    
                    # Mostrar resultado
                    st.markdown(f"""
                    <div class="prediction-result">
                        🌿 <strong>Riego Recomendado:</strong><br>
                        <span style="font-size: 2rem; color: #2E8B57;">{prediccion:.2f} L/m²</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interpretación del resultado
                    if prediccion < df['riego_estimado'].quantile(0.33):
                        interpretacion = "💚 **Riego Bajo** - Condiciones favorables"
                    elif prediccion < df['riego_estimado'].quantile(0.66):
                        interpretacion = "💙 **Riego Moderado** - Condiciones normales"
                    else:
                        interpretacion = "💪 **Riego Alto** - Condiciones secas"
                    
                    st.markdown(interpretacion)
                    
                    # Comparación con históricos
                    percentil = (df['riego_estimado'] <= prediccion).mean() * 100
                    st.info(f"📊 Esta predicción está en el percentil {percentil:.0f}% de los datos históricos")
                    
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")

# Sección de análisis estadístico expandible
with st.expander("📊 Análisis Estadístico Detallado"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Estadísticas Descriptivas:**")
        st.dataframe(df[['temp_media', 'humedad_media', 'precipitacion', 'riego_estimado']].describe())
    
    with col2:
        st.write("**Distribución de Riego Estimado:**")
        fig_hist = px.histogram(df, x='riego_estimado', nbins=20, 
                               title='Distribución del Riego Estimado')
        fig_hist.update_traces(marker_color='#2E8B57')
        st.plotly_chart(fig_hist, use_container_width=True)

# Footer con información técnica
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666;">
    <p><strong>Sistema de Predicción de Riego Agrícola</strong></p>
    <p>Desarrollado con Machine Learning • Datos de Huánuco, Perú • Modelo: Random Forest</p>
    <p>🌐 Datos en tiempo real: Open-Meteo API • Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)



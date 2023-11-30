import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import plotly.graph_objects as go
import plotly.express as px

def main():
    st.title("Démonstration de maintenance prédictive IA")

    st.sidebar.title("Lecture des capteurs")
    st.sidebar.markdown("Valeurs actuelles des capteurs")

    st.markdown("Prédiction de la probabilité de défaillance au fil du temps:")
    prediction_chart_placeholder = st.empty()

    st.markdown("Journal d'événements d'anomalie:")
    log_placeholder = st.empty()

    st.markdown("Contrôle de la simulation:")
    if st.button("Démarrer la simulation"):
        simulation(prediction_chart_placeholder, log_placeholder)

def get_sensor_data():
    return pd.DataFrame(np.random.uniform(low=10.0, high=200.0, size=(60, 5)), columns=['Capteur Température', 'Capteur Pression', 'Capteur Vibration', 'Capteur Humidité', 'Capteur Usure'])

def simulation(prediction_chart_placeholder, log_placeholder):
    sensor_data = get_sensor_data()
    predicted_failures = []
    timestamps = []
    failures_timestamps = []

    n_failure = 0
    start_time = datetime.now()

    for i in range(60):
        if (datetime.now() - start_time).total_seconds() > 30 or n_failure >= 2:
            break

        time.sleep(0.5)

        for sensor, data in sensor_data.items():
            st.sidebar.plotly_chart(go.Figure(go.Indicator(
                mode = "gauge+number",
                value = float(data[i]),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': sensor}),
                layout=dict(height= 100, margin=dict(t=25, b=25))))

        current_prediction = np.random.uniform(0, 1)
        predicted_failures.append(current_prediction)
        current_time = datetime.now()
        timestamps.append(current_time)

        if current_prediction > 0.5:
            n_failure += 1
            explanation = f"Probabilité élevée de défaillance {current_prediction}! Anomalie détectée dans le capteur de température et le capteur de vibration."
            log_placeholder.error(explanation)
            failures_timestamps.append(current_time)
            show_modal()
            if st.button("Reprendre la simulation"):
                continue
            else:
                break
        else:
            explanation = "Les anomalies sont dans la plage acceptable. Le système est sain."
            log_placeholder.info(explanation)

        prediction_chart = pd.DataFrame({"Probabilité Prévue de Défaillance": predicted_failures}, index=timestamps)
        fig = px.line(prediction_chart, y="Probabilité Prévue de Défaillance", title='Probabilité de défaillance au fil du temps')
        
        for failure_time in failures_timestamps:
            fig.add_shape(type="line",
                x0=failure_time, y0=0, x1=failure_time, y1=1,
                line=dict(color="Orange",width=2, dash="dot"))
            
        fig.add_shape(type="line",
            x0=timestamps[0], y0=0.5, x1=timestamps[-1], y1=0.5,
            line=dict(color="Red",width=2, dash="dot"))
        
        prediction_chart_placeholder.plotly_chart(fig)

def show_modal():
    st.markdown("**Défaillance détectée.**")
    st.markdown("""
        Veuillez choisir une action :
        1. **Envoyer une équipe de maintenance** - Envoyer une équipe d'intervention immédiate pour effectuer une maintenance d'urgence.
        2. **Planifier une période d'arrêt du système** - Réserver un créneau horaire pour effectuer une maintenance du système.
        3. **Commander des pièces de rechange** - Vérifier l'inventaire des pièces de rechange et passer une commande si nécessaire.
        """)
    st.button("Envoyer une équipe de maintenance")
    st.button("Planifier une période d'arrêt du système")
    st.button("Commander des pièces de rechange")

if __name__ == "__main__":
    main()
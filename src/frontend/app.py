# src/frontend/app.py

import streamlit as st
import requests

# Titlu aplicație
st.title("📚 BookBot – Recomandări AI de cărți")
st.write("Descoperă cărți în funcție de temele care te pasionează.")

# Input de la utilizator
user_input = st.text_input("🔎 Ce fel de carte cauți?")

# Buton de trimis întrebarea
if st.button("Generează recomandare"):
    if not user_input.strip():
        st.warning("Introdu o întrebare mai întâi.")
    else:
        with st.spinner("Caut cea mai potrivită carte..."):
            try:
                # Trimite întrebarea către backend FastAPI
                response = requests.post(
                    "http://localhost:8000/api/chat",
                    json={"question": user_input}
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"📖 Recomandare: **{data['recommendation']}**")

                    st.markdown(f"🧠 **Motivare:** {data['reasoning']}")
                    st.markdown(f"📖 **Rezumat detaliat:** {data['detailed_summary']}")

                    if data.get("audio_url"):
                        st.audio(data["audio_url"])

                    if data.get("image_url"):
                        st.image(data["image_url"])
                else:
                    st.error(response.json().get("detail", "Eroare necunoscută"))

            except Exception as e:
                st.error(f"Eroare conexiune: {e}")

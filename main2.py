import streamlit as st
import langchain_helper as lch

st.title("Pet Name Generator")

name = st.sidebar.selectbox('What type of animal do you have',['cat','dog','bird','cow']);

if name : 
    response = lch.generate_pet_name(name)
    st.text(response)
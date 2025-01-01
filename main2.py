import streamlit as st
import langchain_helper as lch

st.title("Pet Name Generator")

animal_type = st.sidebar.selectbox("What kind of animal do you have?", ["Dog", "Cat", "Fish", "Bird"])

if animal_type : 
    response = lch.generate_pet_name(animal_type)
    st.text(response)
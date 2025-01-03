import streamlit as st


def main(): 
    st.set_page_config(page_title="Bank ChatBot", page_icon=":bank:")

    st.header("Bank ChatBot :bank:")
    st.text_input("Talk with our chabot")

if __name__ == '__main__':
    main()
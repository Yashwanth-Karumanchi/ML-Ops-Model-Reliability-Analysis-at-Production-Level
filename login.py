import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title='Login', page_icon='👨🏻‍💻', layout='wide', initial_sidebar_state="collapsed")
df = pd.read_csv('creds.csv')
st.markdown("<h1 style='text-align: center;'>ML Ops: Model Reliability Analysis at Production Level</h1>", unsafe_allow_html=True)
mark1 = st.markdown("<p style='font-size: 20px;text-align: center;'>An interactive dashboard for automating the process of monitoring the model health</p>", unsafe_allow_html=True)

user = st.text_input("Username")
passwrd = st.text_input("Password",type="password")
st.write('##')
if st.button('Login'):
    if len(user) == 0 or len(passwrd) == 0:
        st.error('Please enter credentials') 
    elif user in df['Username'].tolist() and (passwrd == str(df['Password'].tolist()[df['Username'].tolist().index(user)])):
        st.success("Login Successful")
        details = df[df['Username'] == user] 
        st.session_state['username'] = user
        st.session_state['occupation'] = details['Occupation'].tolist()[0]
        st.session_state['id'] = details['ID'].tolist()[0]
        st.session_state['email'] = details['Email'].tolist()[0]
        switch_page('home')
    else:
        st.error("Login Failed")

logout = st.sidebar.button('Logout')
if logout and len(st.session_state) != 0:
    st.session_state.clear()
    st.sidebar.success("Sign out successful")
elif logout and len(st.session_state) == 0:
    st.sidebar.error("Please login first")

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
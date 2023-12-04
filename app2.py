import streamlit as st
from streamlit_webrtc import webrtc_streamer


st.title("Hello 2021!")

webrtc_streamer(key="sample")

import streamlit as st


home_page = st.Page(
    "pages/home.py",
    title = "Dashboard",
    default = True
)

sec_a = st.Page(
    "pages/sck2-section-a.py",
    title = "Section A"
)

sec_b = st.Page(
    "pages/sck2-section-b.py",
    title = "Section B"
)

pages = st.navigation({
    "Home"            : [home_page],
    "Practicals"      : [sec_a, sec_b]
})

st.set_page_config(page_title="GGY 168 Grading App", layout="wide")
pages.run()

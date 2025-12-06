import streamlit as st
import langchain_utils as lc  

st.set_page_config(page_title="Restaurant Name Generator", page_icon="🍽️")

st.title("🍽️ Restaurant Name & Menu Generator")

cuisine = st.sidebar.selectbox(
    "Pick a cuisine",
    ("Indian", "Italian", "Mexican", "Arabic")
)

if cuisine:
    response = lc.generate_restaurant_name_and_items(cuisine)

    st.header(response["restaurant_name"].strip())

    menu_items = response["menu_items"].strip().split(",")

    st.subheader(" Menu Items")
    for item in menu_items:
        st.write("- " + item.strip())

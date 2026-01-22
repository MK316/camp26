import streamlit as st
import streamlit.components.v1 as components

def main():
    st.caption("💙 Project Board: Padlet")
    st.write("➡️ Click the '+' sign to write.")
    # Padlet embed URL (you need to replace this with your actual Padlet embed URL)
    padlet_url = "https://padlet.com/mirankim316/bootcamp26"

    # Create an iframe to embed the Padlet
    padlet_iframe = f"<iframe src='{padlet_url}' width='100%' height='600' frameborder='0' allow='autoplay'></iframe>"

    # Display the iframe in Streamlit
    components.html(padlet_iframe, height=600)

if __name__ == "__main__":
    main()

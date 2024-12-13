import streamlit as st
from PIL import Image, ImageStat
import pytesseract
import cv2
import openai

# Set your OpenAI API key here
openai.api_key = "YOUR_API_KEY"

# Streamlit UI
st.title("Visual Accessibility Checker")
uploaded_file = st.file_uploader("Upload an image or document", type=["jpg", "png", "pdf"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded File", use_column_width=True)
    st.write("Analyzing the file...")

    # Extract text from image
    img = Image.open(uploaded_file)
    text = pytesseract.image_to_string(img)
    st.write("Extracted Text:")
    st.text(text)

    # Readability Analysis
    st.write("### Readability Analysis")
    # Grayscale contrast check
    img_cv = cv2.cvtColor(cv2.imread(uploaded_file.name), cv2.COLOR_BGR2GRAY)
    mean_brightness = img_cv.mean()
    st.write(f"Average brightness: {mean_brightness}")
    if mean_brightness < 50:
        st.warning("Low contrast: Consider increasing text brightness.")

    # Color Contrast Check
    stat = ImageStat.Stat(img)
    brightness = sum(stat.mean[:3]) / 3  # Average of R, G, B
    if brightness < 125:
        st.warning("Low color contrast: Consider using lighter colors for better readability.")

    # AI Accessibility Suggestions
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Analyze the following text for readability and suggest improvements:\n{text}",
        max_tokens=150
    )
    st.write("AI Suggestions:")
    st.text(response.choices[0].text)

    # AI Alternative Text Generation
    response_alt = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Generate an alternative text description for this image based on its content.",
        max_tokens=100
    )
    st.write("Suggested Alt Text:")
    st.text(response_alt.choices[0].text)
import streamlit as st
st.title("Test App")
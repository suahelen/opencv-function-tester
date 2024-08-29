import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
from cv_functions import opencv_functions
from utils import (
    initialize_session_state,
    save_current_state,
    revert_state,
    forward_state,
    generate_parameter_ui,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    st.set_page_config(layout="wide")  # Use the wide layout
    st.title("OpenCV Image Processing App")
    initialize_session_state()

    # Handle file upload and check if it's a new file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if "file" not in st.session_state or uploaded_file != st.session_state.file:
            image = Image.open(uploaded_file)
            image = np.array(image)
            logger.info(f"Image shape: {image.shape}")

            if image.dtype == bool:
                image = image.astype(np.uint8) * 255

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            st.session_state.image = image
            save_current_state(image)
            st.session_state.file = (
                uploaded_file  # Track the uploaded file in session state
            )

    if st.session_state.image is not None:
        selected_function = st.selectbox(
            "Select OpenCV function", list(opencv_functions.keys())
        )
        params = generate_parameter_ui(selected_function)

        try:

            start = time.time()

            processed_image = opencv_functions[selected_function].process(
                st.session_state.image.copy(), **params
            )

            end = time.time()

            # displaying the time taken to process the image
            st.write(f"Time taken to process the image: {end-start} seconds")

            # Make the image display wider
            col1, col2 = st.columns([2, 2])
            with col1:
                st.header("Input Image")
                st.image(
                    cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB),
                    use_column_width=True,
                )
            with col2:
                st.header("Processed Image")
                st.image(
                    cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                    use_column_width=True,
                )

            col3, col4, col5 = st.columns(3)
            with col3:
                if st.button("Revert", use_container_width=True):
                    revert_state()
                    st.rerun()  # Rerun the script to update the image

            with col4:
                if st.button("Accept", use_container_width=True):
                    st.session_state.image = processed_image
                    save_current_state(processed_image)
                    st.success(
                        "The processed image has been accepted and set as the new input."
                    )
                    st.rerun()

            with col5:
                if st.button("Forward", use_container_width=True):
                    forward_state()
                    st.rerun()  # Rerun the script to update the image

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.write("Please upload an image to get started.")


if __name__ == "__main__":
    main()

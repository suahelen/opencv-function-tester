import streamlit as st
import cv2
import numpy as np
from PIL import Image


def get_ui_parameters(selected_function):
    params = {}
    param_info = selected_function.get_params()
    secondary_input_required = "requires_secondary_image" in param_info
    if secondary_input_required:
        param_info.pop("requires_secondary_image")

    # Organize controls into rows with up to 3 columns
    param_names = list(param_info.keys())
    for i in range(0, len(param_names), 3):
        cols = st.columns(3)
        for j, param in enumerate(param_names[i : i + 3]):
            config = param_info[param]
            with cols[j]:
                if isinstance(config, list):  # Dropdown options (including enums)
                    params[param] = st.selectbox(param, config)
                elif isinstance(config, tuple):  # Sliders
                    min_val, max_val, default, step = config
                    params[param] = st.slider(param, min_val, max_val, default, step)
                elif isinstance(config, bool):  # Checkbox
                    params[param] = st.checkbox(param, config)
                elif isinstance(config, int):  # Integer inputs
                    params[param] = st.number_input(param, value=config, step=1)
                elif isinstance(config, float):  # Float inputs
                    params[param] = st.number_input(param, value=config, format="%.4f")

    secondary_image = None
    if secondary_input_required:
        st.write("Please upload a second image (if required by the function):")
        uploaded_file_secondary = st.file_uploader(
            "Choose a second image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file_secondary is not None:
            secondary_image = Image.open(uploaded_file_secondary)
            secondary_image = np.array(secondary_image)
            if secondary_image.shape[2] == 4:  # If the image has an alpha channel
                secondary_image = cv2.cvtColor(secondary_image, cv2.COLOR_RGBA2BGR)
            else:
                secondary_image = cv2.cvtColor(secondary_image, cv2.COLOR_RGB2BGR)

    return params, secondary_image


def initialize_session_state():
    if "image" not in st.session_state:
        st.session_state.image = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "history_index" not in st.session_state:
        st.session_state.history_index = -1
    if "file" not in st.session_state:
        st.session_state.file = None


def save_current_state(image):
    """Save the current image state and reset forward history."""
    # If the current index isn't at the end of the history, truncate history
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[
            : st.session_state.history_index + 1
        ]

    # Add the new image to history and update the index
    st.session_state.history.append(image)
    st.session_state.history_index = len(st.session_state.history) - 1


def revert_state():
    """Revert to the previous image state."""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.image = st.session_state.history[
            st.session_state.history_index
        ]


def forward_state():
    """Move forward to the next image state."""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        st.session_state.image = st.session_state.history[
            st.session_state.history_index
        ]

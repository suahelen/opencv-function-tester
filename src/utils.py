import streamlit as st
from cv_functions import opencv_functions
from enum import Enum


def generate_parameter_ui(selected_function):
    param_info = opencv_functions[selected_function].get_params()
    params = {}

    # Organize controls into rows with up to 3 columns
    param_names = list(param_info.keys())
    for i in range(0, len(param_names), 3):
        cols = st.columns(3)
        for j, param in enumerate(param_names[i : i + 3]):
            config = param_info[param]
            with cols[j]:
                if isinstance(config, list) and isinstance(config[0], Enum):  # Enums
                    options = [e.name for e in config]
                    selected = st.selectbox(param, options)
                    params[param] = next(e for e in config if e.name == selected)
                elif isinstance(config, list):  # Dropdown options
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

    return params


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

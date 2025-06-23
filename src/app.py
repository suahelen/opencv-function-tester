import streamlit as st
import cv2
import time
import numpy as np
from PIL import Image
from functions import opencv_functions
from utils import (
    initialize_session_state,
    save_current_state,
    revert_state,
    forward_state,
    get_ui_parameters,
    get_function_history_json,
    display_function_history,
    import_and_apply_json,
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
        # If a new file is uploaded or session state is empty
        if "file" not in st.session_state or uploaded_file != st.session_state.file:
            image = Image.open(uploaded_file)
            image = np.array(image)
            logger.info(f"Image shape: {image.shape}")

            # Handle different image types and convert to BGR
            if image.dtype == bool:
                image = image.astype(np.uint8) * 255

            if len(image.shape) == 2:  # If grayscale, convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:  # If RGBA, convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:  # Assume RGB and convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            st.session_state.image = image
            save_current_state(image)  # Reset function history for new image
            st.session_state.file = (
                uploaded_file  # Track the uploaded file in session state
            )

    if "image" in st.session_state:
        # Create tabs for main processing and function history
        tab1, tab2 = st.tabs(["Image Processing", "Function History"])
        
        with tab1:
            selected_function_string = st.selectbox(
                "Select OpenCV function", list(opencv_functions.keys())
            )

            selected_function = opencv_functions[selected_function_string]
            # Get UI parameters and secondary image if required
            params, secondary_image = get_ui_parameters(selected_function)

            try:
                start = time.time()
                current_img = st.session_state.image
                # Process the image using the selected function
                if secondary_image is not None:
                    processed_image = selected_function.process(
                        current_img.copy(), secondary_image, **params
                    )

                    st.image(
                        cv2.cvtColor(secondary_image, cv2.COLOR_BGR2RGB),
                        caption="Secondary Image",
                        use_column_width=False,
                    )
                elif current_img is not None:
                    processed_image = selected_function.process(
                        current_img.copy(), **params
                    )
                else:
                    st.error("Please upload an image to get started.")
                    return

                end = time.time()

                # Displaying the time taken to process the image
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

                # Control buttons
                col3, col4, col5 = st.columns(3)
                with col3:
                    if st.button("Revert", use_container_width=True):
                        revert_state()
                        st.rerun()  # Rerun the script to update the image

                with col4:
                    if st.button("Accept", use_container_width=True):
                        st.session_state.image = processed_image
                        save_current_state(processed_image, selected_function_string, params)
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
        
        with tab2:
            st.header("Function History")
            
            # Display current function history
            display_function_history()
            
            # JSON export section
            st.subheader("Export Function History")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("Show JSON", use_container_width=True):
                    json_data = get_function_history_json()
                    st.code(json_data, language="json")
            
            with col2:
                json_data = get_function_history_json()
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"opencv_processing_history_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            st.divider()
            
            # JSON import section
            st.subheader("Import Function History")
            
            uploaded_json = st.file_uploader(
                "Choose a JSON history file...", 
                type=["json"],
                help="Upload a previously exported function history JSON file"
            )
            
            if uploaded_json is not None:
                st.write("**Import Options:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Add to Current History", use_container_width=True, type="primary"):
                        with st.spinner("Applying functions..."):
                            success, message, count = import_and_apply_json(
                                uploaded_json, opencv_functions, mode="add"
                            )
                        
                        if success:
                            st.success(f"✅ {message}")
                            time.sleep(1)  # Brief pause to show success
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                
                with col2:
                    if st.button("Restart History", use_container_width=True, type="secondary"):
                        with st.spinner("Restarting and applying functions..."):
                            success, message, count = import_and_apply_json(
                                uploaded_json, opencv_functions, mode="restart"
                            )
                        
                        if success:
                            st.success(f"✅ {message}")
                            time.sleep(1)  # Brief pause to show success
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                
                st.info("""
                **Import Modes:**
                - **Add to Current History**: Applies the imported functions after your current processing steps
                - **Restart History**: Clears current history and applies the imported functions from the original image
                
                If any error occurs during import, all changes will be reverted and your original image restored.
                """)
            
            st.divider()
            
            # Clear history option
            if st.button("Clear Function History", type="secondary"):
                st.session_state.function_history = []
                st.session_state.function_history_index = -1
                st.success("Function history cleared!")
                st.rerun()
                
    else:
        st.write("Please upload an image to get started.")


if __name__ == "__main__":
    main()

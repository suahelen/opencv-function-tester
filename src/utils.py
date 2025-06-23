import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import datetime
import traceback
from functions.enums import (
    ColorSpaceConversionType, ThresholdType, MorphShape, MorphOperation,
    Interpolation, BorderType, CornerRefineMethod, DictType,
    AkazeDescriptorType, DiffusivityType, OrbHarrisScore, FastFeatureType,
    TemplateMatchingMethod
)


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
    if "function_history" not in st.session_state:
        st.session_state.function_history = []
    if "function_history_index" not in st.session_state:
        st.session_state.function_history_index = -1


def save_current_state(image, function_name=None, function_params=None):
    """Save the current image state and reset forward history."""
    # If the current index isn't at the end of the history, truncate history
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[
            : st.session_state.history_index + 1
        ]
        st.session_state.function_history = st.session_state.function_history[
            : st.session_state.function_history_index + 1
        ]

    # Add the new image to history and update the index
    st.session_state.history.append(image)
    st.session_state.history_index = len(st.session_state.history) - 1
    
    # Track function application if provided
    if function_name is not None:
        function_record = {
            "step": len(st.session_state.function_history) + 1,
            "function_name": function_name,
            "parameters": serialize_params(function_params) if function_params else {},
            "timestamp": datetime.datetime.now().isoformat(),
        }
        st.session_state.function_history.append(function_record)
        st.session_state.function_history_index = len(st.session_state.function_history) - 1


def serialize_params(params):
    """Convert parameters to JSON-serializable format."""
    serialized = {}
    for key, value in params.items():
        if hasattr(value, 'name'):  # Handle enum values
            serialized[key] = value.name
        elif hasattr(value, 'value'):  # Handle enum values with .value
            serialized[key] = value.value
        elif isinstance(value, (int, float, str, bool, list, tuple)):
            serialized[key] = value
        else:
            serialized[key] = str(value)  # Fallback to string representation
    return serialized


def deserialize_params(params, opencv_functions):
    """Convert JSON parameters back to their proper types including enums."""
    # List of all enum classes
    enum_classes = [
        ColorSpaceConversionType, ThresholdType, MorphShape, MorphOperation,
        Interpolation, BorderType, CornerRefineMethod, DictType,
        AkazeDescriptorType, DiffusivityType, OrbHarrisScore, FastFeatureType,
        TemplateMatchingMethod
    ]
    
    deserialized = {}
    for key, value in params.items():
        # Try to convert enum string names back to enum objects
        if isinstance(value, str):
            # Check all enum types
            for enum_class in enum_classes:
                try:
                    if hasattr(enum_class, value):
                        deserialized[key] = getattr(enum_class, value)
                        break
                except:
                    pass
            else:
                deserialized[key] = value
        else:
            deserialized[key] = value
    
    return deserialized


def revert_state():
    """Revert to the previous image state."""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.image = st.session_state.history[
            st.session_state.history_index
        ]
        
        # Update function history index to match image history
        # Function history index should be history_index - 1 (since first image has no function)
        st.session_state.function_history_index = max(-1, st.session_state.history_index - 1)


def forward_state():
    """Move forward to the next image state."""
    if st.session_state.history_index < len(st.session_state.history) - 1:
        st.session_state.history_index += 1
        st.session_state.image = st.session_state.history[
            st.session_state.history_index
        ]
        
        # Update function history index to match image history
        # Function history index should be history_index - 1 (since first image has no function)
        st.session_state.function_history_index = min(
            st.session_state.history_index - 1,
            len(st.session_state.function_history) - 1
        )


def get_function_history_json():
    """Return the function history as a JSON string."""
    if not st.session_state.function_history or st.session_state.function_history_index < 0:
        return json.dumps({"message": "No functions applied yet"}, indent=2)
    
    # Only show functions up to current position
    current_functions = st.session_state.function_history[:st.session_state.function_history_index + 1]
    
    history_data = {
        "processing_pipeline": {
            "total_steps": len(current_functions),
            "current_step": st.session_state.function_history_index + 1,
            "functions_applied": current_functions
        },
        "export_timestamp": datetime.datetime.now().isoformat()
    }
    
    return json.dumps(history_data, indent=2)


def validate_json_structure(data):
    """Validate the structure of imported JSON data."""
    try:
        if not isinstance(data, dict):
            return False, "JSON must be an object"
        
        if "processing_pipeline" not in data:
            return False, "Missing 'processing_pipeline' key"
        
        pipeline = data["processing_pipeline"]
        if not isinstance(pipeline, dict):
            return False, "'processing_pipeline' must be an object"
        
        if "functions_applied" not in pipeline:
            return False, "Missing 'functions_applied' key"
        
        functions = pipeline["functions_applied"]
        if not isinstance(functions, list):
            return False, "'functions_applied' must be a list"
        
        for i, func in enumerate(functions):
            if not isinstance(func, dict):
                return False, f"Function {i+1} must be an object"
            
            if "function_name" not in func:
                return False, f"Function {i+1} missing 'function_name'"
            
            if "parameters" not in func:
                return False, f"Function {i+1} missing 'parameters'"
        
        return True, "Valid JSON structure"
    
    except Exception as e:
        return False, f"JSON validation error: {str(e)}"


def apply_function_sequence(opencv_functions, functions_to_apply, mode="add"):
    """Apply a sequence of functions from JSON data."""
    try:
        # Store original state for rollback
        original_image = st.session_state.image.copy()
        original_history = st.session_state.history.copy()
        original_history_index = st.session_state.history_index
        original_function_history = st.session_state.function_history.copy()
        original_function_history_index = st.session_state.function_history_index
        
        # If restarting, go back to the original image
        if mode == "restart":
            if len(st.session_state.history) > 0:
                st.session_state.image = st.session_state.history[0].copy()
                st.session_state.history = [st.session_state.history[0]]
                st.session_state.history_index = 0
                st.session_state.function_history = []
                st.session_state.function_history_index = -1
        
        # Apply each function in sequence
        applied_count = 0
        for func_data in functions_to_apply:
            function_name = func_data["function_name"]
            
            if function_name not in opencv_functions:
                raise ValueError(f"Unknown function: {function_name}")
            
            function_class = opencv_functions[function_name]
            parameters = deserialize_params(func_data.get("parameters", {}), opencv_functions)
            
            # Apply the function
            current_image = st.session_state.image.copy()
            processed_image = function_class.process(current_image, **parameters)
            
            # Save the state
            st.session_state.image = processed_image
            save_current_state(processed_image, function_name, parameters)
            applied_count += 1
        
        return True, f"Successfully applied {applied_count} functions", applied_count
    
    except Exception as e:
        # Rollback on error
        st.session_state.image = original_image
        st.session_state.history = original_history
        st.session_state.history_index = original_history_index
        st.session_state.function_history = original_function_history
        st.session_state.function_history_index = original_function_history_index
        
        error_msg = f"Error applying functions: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        return False, error_msg, 0


def import_and_apply_json(uploaded_json, opencv_functions, mode="add"):
    """Import JSON and apply the function sequence."""
    try:
        # Parse JSON
        json_data = json.loads(uploaded_json.getvalue().decode("utf-8"))
        
        # Validate structure
        is_valid, validation_msg = validate_json_structure(json_data)
        if not is_valid:
            return False, f"Invalid JSON structure: {validation_msg}", 0
        
        # Extract functions to apply
        functions_to_apply = json_data["processing_pipeline"]["functions_applied"]
        
        if not functions_to_apply:
            return False, "No functions found in JSON", 0
        
        # Apply the functions
        success, message, count = apply_function_sequence(opencv_functions, functions_to_apply, mode)
        
        return success, message, count
    
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON format: {str(e)}", 0
    except Exception as e:
        return False, f"Import error: {str(e)}", 0


def display_function_history():
    """Display the current function history in the UI."""
    if not st.session_state.function_history or st.session_state.function_history_index < 0:
        st.info("No functions have been applied yet.")
        return
    
    # Only show functions up to current position
    current_functions = st.session_state.function_history[:st.session_state.function_history_index + 1]
    
    st.subheader(f"Applied Functions ({len(current_functions)} steps)")
    
    # Show total available functions if there are more
    total_functions = len(st.session_state.function_history)
    if total_functions > len(current_functions):
        st.info(f"Showing {len(current_functions)} of {total_functions} total functions (use Forward to see more)")
    
    for i, func_record in enumerate(current_functions, 1):
        with st.expander(f"Step {i}: {func_record['function_name']}"):
            st.write(f"**Function:** {func_record['function_name']}")
            st.write(f"**Timestamp:** {func_record['timestamp']}")
            if func_record['parameters']:
                st.write("**Parameters:**")
                for param, value in func_record['parameters'].items():
                    st.write(f"  - {param}: {value}")
            else:
                st.write("**Parameters:** None")

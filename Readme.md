

# Image Processing with Streamlit and OpenCV

This project is a modular image processing application built with Streamlit and OpenCV. It allows users to apply various image processing techniques through an interactive web interface. The project is designed to be easily extensible, with each image processing function encapsulated in its own class.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Adding New Functions](#adding-new-functions)
- [Project Structure](#project-structure)

## Getting Started

### Prerequisites
- Python 3.x
- `pixi` (Python package manager)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/image-processing-app.git
   cd image-processing-app
   ```

2. **Install the required packages:**

   ```bash
   pixi install
   ```

   add new packages:
   ```bash
   pixi add <package-name>
   ```

3. **Run the application:**

   ```bash
   streamlit run app.py
   ```

   This will start the Streamlit application, which you can access in your web browser at `http://localhost:8501`.

## Usage

Once the application is running, you can:

1. **Upload an Image:**
   - Click on the "Choose an image..." button to upload an image file.

2. **Select an Image Processing Function:**
   - Use the dropdown menu to select the image processing function you want to apply.

3. **Adjust Parameters:**
   - Each function has its own set of adjustable parameters. The UI will update dynamically based on the selected function.

4. **Process the Image:**
   - The processed image will be displayed alongside the original image.
   - You can use the "Accept" button to apply the processed image as the new input for further processing.
   - Use the "Revert" button to undo the last action, or the "Forward" button to redo it.

## Adding New Functions

### 1. **Create a New Function Class**

To add a new image processing function:

1. **Create a new file** in the `functions/` directory for your function. For example, `new_function.py`.

2. **Define the function class** with a `process` method and a `get_params` method:
Certainly! Below is an enhanced section for the README that includes information on the available parameter types and an example implementation.

---

### 2. **Define the Function Class**

To add a new image processing function, you'll define a class that includes two methods: `process` and `get_params`.

- **`process` method**: This method takes the image and the required parameters, applies the processing, and returns the processed image.
- **`get_params` method**: This method returns a dictionary where the keys are the parameter names, and the values are the configuration for UI controls.

Note: The names of the parameters in the `process` method must match the keys in the `get_params` method.


#### Available Parameter Types

In the `get_params` method, you can define parameters with the following types:

- **Slider**: A tuple `(min, max, default, step)` that creates a slider.
  - Example: `"param1": (0, 100, 50, 1)` creates a slider with a range from 0 to 100, default value of 50, and a step of 1.
- **Dropdown (List of Values)**: A list of possible values.
  - Example: `"param2": [1, 2, 3, 4]` creates a dropdown with options 1, 2, 3, and 4.
- **Enum Dropdown**: A list of Enum values. The dropdown will display the names of the enums.
  - Example: `["EnumType.VALUE1", "EnumType.VALUE2"]` will display as "VALUE1" and "VALUE2".
- **Checkbox**: A boolean value (`True` or `False`).
  - Example: `"param3": True` creates a checkbox that is checked by default.
- **Integer Input**: An integer value.
  - Example: `"param4": 10` creates an input box with a default integer value of 10.
- **Float Input**: A float value.
  - Example: `"param5": 0.5` creates an input box with a default float value of 0.5.

#### Example: Adding a New Function

Here's how you might implement a new function, `NewFunction`, that applies a simple thresholding operation to an image:

```python
import cv2

class NewFunction:
    @staticmethod
    def process(image, thresh, maxval, thresh_type):
        _, processed_image = cv2.threshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), thresh, maxval, thresh_type
        )
        return cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def get_params():
        return {
            "thresh": (0, 255, 127, 1),  # Slider: min=0, max=255, default=127, step=1
            "maxval": (0, 255, 255, 1),  # Slider: min=0, max=255, default=255, step=1
            "thresh_type": [
                cv2.THRESH_BINARY,
                cv2.THRESH_BINARY_INV,
                cv2.THRESH_TRUNC,
                cv2.THRESH_TOZERO,
                cv2.THRESH_TOZERO_INV,
            ],  # Dropdown: Select threshold type
        }
```

**Explanation**:
- **`thresh` and `maxval`**: These are defined as sliders with ranges from 0 to 255.
- **`thresh_type`**: This is a dropdown list where users can choose the type of thresholding operation.


### 2. **Register the Function**

1. **Import your new function** in the `functions/__init__.py` file:

   ```python
   from .new_function import NewFunction
   ```

2. **Add your function** to the `opencv_functions` dictionary in `opencv_functions.py`:

   ```python
   opencv_functions = {
       # Other functions...
       "New Function": NewFunction,
   }
   ```

### 3. **Run the Application**

Run `streamlit run app.py` again. Your new function will appear in the dropdown list, and you can test it via the web interface.

## Project Structure

```
src/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # List of required packages
├── cv_functions.txt        # Dictionary of all available functions
├── functions/              # Folder containing function classes
│   ├── __init__.py         # Import all classes from this folder
│   ├── basic_operations.py # Class with some basic operations
│   ├── aruco.py            # Class with some aruco operations
│   ├── new_function.py     # Example: Class for New Function
```

## Contributing

Feel free to fork this repository and contribute by adding new functions or improving existing ones. Pull requests are welcome!



# Image Processing with Streamlit and OpenCV

This project is a modular image processing application built with Streamlit and OpenCV. It allows users to apply various image processing techniques through an interactive web interface. The project is designed to be easily extensible, with each image processing function encapsulated in its own class.

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Adding New Functions](#adding-new-functions)
- [Project Structure](#project-structure)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- `pip` (Python package installer)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/image-processing-app.git
   cd image-processing-app
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   Make sure `requirements.txt` includes the necessary packages:

   ```text
   streamlit
   opencv-python
   numpy
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

   ```python
   import cv2

   class NewFunction:
       @staticmethod
       def process(image, param1, param2):
           # Your image processing code here
           return processed_image

       @staticmethod
       def get_params():
           return {
               "param1": (min, max, default, step),
               "param2": (min, max, default, step)
           }
   ```

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

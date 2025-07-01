import os
import sys
import json

# Determine the base path for the application.
# This is crucial for finding external files, especially with --onefile.
# sys._MEIPASS is the path to the temporary folder where PyInstaller extracts files.
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    bundle_dir = sys._MEIPASS
else:
    # Running in a regular Python environment (for development)
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to your settings file.
# We assume settings.json will be in the same directory as the executable.
settings_file_path = os.path.join(bundle_dir, 'settings.json')

# --- Provide default settings ---
# This is important for the first run or if the settings file is missing/corrupt.
default_settings = {
    "setting_a": "default_value",
    "number_of_retries": 3,
    "enable_feature_x": False,
    "output_directory": os.path.join(os.path.expanduser("~"), "MyApplicationOutput")
}

# --- Load settings ---
settings = default_settings.copy() # Start with defaults

try:
    with open(settings_file_path, 'r', encoding='utf-8') as f:
        loaded_settings = json.load(f)
        settings.update(loaded_settings) # Override defaults with loaded settings
    print(f"Loaded settings from: {settings_file_path}")
except FileNotFoundError:
    print(f"Settings file not found at {settings_file_path}. Using default settings.")
    # Optionally, create a default settings file for the client
    try:
        with open(settings_file_path, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, indent=4)
        print(f"Created a default settings file at {settings_file_path}.")
    except IOError as e:
        print(f"Could not write default settings file: {e}")
except json.JSONDecodeError:
    print(f"Error decoding JSON from {settings_file_path}. Using default settings.")
    # You might want to back up the corrupt file and create a new default here
except Exception as e:
    print(f"An unexpected error occurred while loading settings: {e}")

# --- Now you can use the settings in your application ---
print(f"Setting A: {settings['setting_a']}")
print(f"Retries: {settings['number_of_retries']}")
print(f"Feature X Enabled: {settings['enable_feature_x']}")
print(f"Output Directory: {settings['output_directory']}")

# Example application logic using settings
if settings['enable_feature_x']:
    print("Feature X is enabled. Doing something special!")

# Your main application logic would follow here

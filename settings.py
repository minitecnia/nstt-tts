# Settings file
# Francisco Jose Ochando

from pathlib import Path
import sys
from PIL import Image

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
STREAMING = 'Streaming'
RTSP = 'RTSP'
YOUTUBE = 'Youtube'

# Images config
IMAGES_DIR = ROOT / 'images'

# Path models
path_models = 'models/'

# Main title of streamlit application
main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:30px; 
                    font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Audio iNSTT 
                    </h1></div>"""

im = Image.open("logos/speechgpt.ico")

# Subtitle of streamlit application
sub_title_cfg = """<div><h4 style="color:#FF64DA; text-align:center; font-size:20px;
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Iberian Noisy Speech to text transcription 🚀</h4>
                    </div>"""

# Background image
# background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# background-image: url("https://img.freepik.com/foto-gratis/puesta-sol-misurina_181624-34793.jpg");
background_image = """<style>
                        [data-testid="stAppViewContainer"] > .main {                          
                        background-image: url("https://cdn.worldvectorlogo.com/logos/nato-8.svg")
                        background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
                        background-position: center;  
                        background-repeat: no-repeat;
                        }
                        </style>"""

# Append the custom HTML
input_style = """<style>
                input[type="text"] {
                background-color: transparent;
                color: #a19eae;  // This changes the text color inside the input box
                }
                div[data-baseweb="base-input"] {
                background-color: transparent !important;
                }
                [data-testid="stAppViewContainer"] {
                background-color: transparent !important;
                }
                </style>"""

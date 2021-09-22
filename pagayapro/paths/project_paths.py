import pathlib
import os

_config_path = pathlib.Path(__file__).resolve().parent

PROJECT_ROOT = _config_path.parents[1]

PACKAGE_PATH = os.path.join(PROJECT_ROOT, "pagayapro")
MODELS_PATH = os.path.join(PACKAGE_PATH, "models")
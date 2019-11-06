import logging
from os.path import join
from pathlib import Path
from typing import Any

# Package imports
import pypso

# Create custom console logger
FORMAT: str = '[%(asctime)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger: Any = logging.getLogger(__name__)

# # Read VERSION file and import on module load
# root = Path(pypso.__file__).resolve().parent
# with open(join(root, 'VERSION')) as version_file:
#     __version__ = version_file.read().strip()
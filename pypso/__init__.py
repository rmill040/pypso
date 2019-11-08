import logging
import re
from typing import Any
from ._version import get_versions

# Create custom console logger
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s - %(message)s'
    )
logger: Any = logging.getLogger(__name__)

# Define version
__version__ = re.findall(r"\d+\.\d+\.\d+", get_versions()['version'])[0]
del get_versions
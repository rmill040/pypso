from os.path import join
from pathlib import Path

# Package imports
import pypso

# Read VERSION file and import on module load
root = Path(pypso.__file__).resolve().parent
with open(join(root, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
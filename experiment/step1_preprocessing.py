# %%
import os
from pathlib import Path

# %%
PATH_ROOT = Path(os.getcwd()).parent.absolute()
print(PATH_ROOT)

# %%
PATH_RAW = os.path.join(PATH_ROOT, 'data', 'raw')
print(PATH_RAW)

# %%




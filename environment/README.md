# Contents

- `jupyter_setup.sh` - Jupyter extensions setup, run this only after installing the requirements
- `requirements.txt` - Project Python requirements

# Setting up the environment

To set up the environment:

 1. Install all libraries from `requirements.txt` (`pip install -r environment/requirements.txt.`).
 2. (Optional) run `jupyter_setup.sh` to configure jupyter extentions.
 3. Add the main directory to your PYTHONPATH (if using virtualenvwrapper, use `add2virtualenv .` from within the main
    directory).
 4. Go to `conf.py` and change the `DATA_DIR` to the absolute path of the data directory.   

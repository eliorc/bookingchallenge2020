# Contents

- `jupyter_setup.sh` - Jupyter extensions setup, run this only after installing the requirements
- `requirements.txt` - Project Python requirements

# Setting up the environment

To set up the environment:

 1. Install all libraries from `requirements.txt` (`pip install -r environment/requirements.txt.`).
 2. (Optional) run `./environment/jupyter_setup.sh` to configure jupyter extensions.
 3. (Necessary for running experiments) run `./environment/build_images.sh` to build the images     
 4. Add the main directory to your PYTHONPATH (if using virtualenvwrapper, use `add2virtualenv .` from within the main
    directory).
 5. Go to `conf.py` and change the `DATA_DIR` to the absolute path of the data directory.   

# To run experiments

This project experiments were executed with Trains, so in order to run them you will need a trains-server deployed 
on default ports.
If you are using an agent (running tasks with `--enqueue`) make sure:

 1. Your agent is in docker mode, and using the `booking:ubuntu20.04-gpu` base image (built locally in setup step 3)
 2. In extra arguments, make sure you map your root data dir to `/opt/data`
    For example, your `agent.default_docker` section should look like
    
    ```
    default_docker: {
        # default docker image to use when running in docker mode
        image: "booking:ubuntu20.04-gpu"

        # optional arguments to pass to docker image
        arguments: ["-v /home/elior/Dev/Projects/bookingchallenge2020/data:/opt/data"]
    }
    ```
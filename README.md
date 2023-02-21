# Setup Repository

Make a virtualenv and install python packages:

```
# make python virtual environment
python -m venv env
# activate it
. env/bin/activate
# install dependencies
pip install -r requirements.txt
# Tell jupyter about the virtual environment
python -m ipykernel install --user --name=env
```

Run the notebook

```
jupyter notebook
```

In the notebook, in the navigation menu select Kernel -> Change Kernel -> env to use the virtual environment.

NOTE: before commiting anything make sure you clean the output of the notebooks to avoid gunking up the repository:

```
nbstripout *.ipynb
```

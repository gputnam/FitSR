# Setup Repository

Make a virtualenv and install python packages:

```
python -m venv env
. env/bin/activate
pip install -r requirements.txt
```

Run the notebook

```
jupyter notebook
```

NOTE: before commiting anything make sure you clean the output of the notebooks to avoid gunking up the repository:

```
nbstripout *.ipynb
```

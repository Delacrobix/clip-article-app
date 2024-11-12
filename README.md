# Clip article Streamlit app

# Setup
Create and Activate a Virtual Environment:

```bash
python -m venv myenv
source myenv/bin/activate
```

## Install Dependencies:

```bash
pip install -r requirements.txt
```

### Run the Streamlit App:

```bash
streamlit run app.py
```

## Index images

Create a `/data` folder in the root directory and add the images you want to index.

run the following command to index the images:

```bash
python index_images_folder.py
```
in the file `index_images_folder.py` you can change the folder path to the folder containing the images you want to index and the index names that will be used to search the images.
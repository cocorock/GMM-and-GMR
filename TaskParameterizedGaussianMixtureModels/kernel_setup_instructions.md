# Kernel Setup Instructions

## 1. Install Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## 2. Install the Package

Install the `tpgmm` package in editable mode. This allows you to make changes to the source code and have them reflected in your environment without reinstalling.

```bash
pip install -e .
```

## 3. Create and Install the Jupyter Kernel

Install `ipykernel` and create a new Jupyter kernel for this project:

```bash
pip install ipykernel
python -m ipykernel install --user --name=tpgmm_kernel

python -m ipykernel install --user --name tpgmm_kernel_311 --display-name "Python 3.11"

jupyter kernelspec list 
```

## 4. Download and Extract the Data

The example notebook requires a dataset that needs to be downloaded and extracted.

1.  **Download the data:**
    *   **Using PowerShell (Windows):**
        ```powershell
        Invoke-WebRequest -Uri "https://kilthub.cmu.edu/ndownloader/files/28625121" -OutFile "examples\data\28625121.zip"
        ```
    *   **Using curl (Linux/macOS):**
        ```bash
        curl -L -o "examples/data/28625121.zip" https://kilthub.cmu.edu/ndownloader/files/28625121
        ```

2.  **Unzip the data:**
    Unzip the downloaded file into the `examples/data` directory. After unzipping, you should have a `7days1` folder inside `examples/data`.

## 5. Modify the Notebook for Correct Pathing

To ensure the notebook can find the data, you need to add a code cell at the beginning of the `examples/example.ipynb` notebook with the following code:

```python
import os
os.chdir('..')
```

This will change the notebook's working directory to the root of the project, allowing it to find the data in `examples/data`.

After completing these steps, you should be able to run the `examples/example.ipynb` notebook successfully using the `tpgmm_kernel`.
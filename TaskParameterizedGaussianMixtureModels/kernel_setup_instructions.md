# Setting up the Jupyter Kernel for the `tpgmm` Project

This document outlines the steps to resolve import errors in the project's Jupyter Notebooks by creating and selecting the correct kernel.

## 1. Diagnosis: Confirming the Environment Tools

The first step is to determine if your Python environment has the necessary tool to communicate with Jupyter Notebook. This tool is the `ipykernel` package.

*   **Action:** Run the command `pip show ipykernel`.
*   **Result:** If the command successfully finds the package, you have the tool required to create and manage custom Jupyter kernels.

## 2. Solution: Creating a Dedicated Kernel

To ensure the Jupyter Notebook uses the correct Python environment where the `tpgmm` library is installed, we create a dedicated kernel.

*   **Action:** Execute the command `python -m ipykernel install --user --name=tpgmm_kernel`.
*   **Result:** This command creates a new Jupyter kernel named `tpgmm_kernel`. This new kernel acts as a direct link to the Python environment where you successfully installed the `tpgmm` package.

## 3. Configuration: Connecting the Notebook to the Correct Kernel

Finally, you need to instruct your specific notebook to use this newly created kernel.

*   **Action:** Perform the following manual steps inside your Jupyter Notebook:
    1.  Open the `example.ipynb` notebook.
    2.  In the top menu, click on **Kernel**.
    3.  From the dropdown, select **Change kernel**.
    4.  Choose the new **`tpgmm_kernel`** from the list.
    5.  Rerun the cells in your notebook.

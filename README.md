- Author: Jun Xiong <junxiong360@gmail.com>  
- Date: 2023-06-09

# ag-seed-prod

## Overview

This project is for Corteva Agriscience Case Study, which aims to predict predicting product units using machine learning models and showcase the intergration of genAI. It includes modules for data processing, model training, and a dashboard for visualization and interaction. The project proposal a XGBoost tree approach + residual correction layer as a PoC.

The codebase is built for PoC purpose only. It is not intended for production use. Check the `Todo` section for more details.

## File Structure

The project is organized as follows:

-   [`README.md`](README.md): This file, providing an overview of the project.
-   `config/`: Contains configuration files (TOML format) for different models and pipelines.
    -   [`residual.toml`](config/residual.toml): Configuration for the residual model.
    -   [`xgb.toml`](config/xgb.toml): Configuration for the XGBoost model.
-   `src/`: Contains the source code for the project.
    -   [`__init__.py`](src/__init__.py): Initializes the `src` directory as a Python package.
    -   `utils.py`: Includes utility functions for model loading, evaluation, and confidence interval calculation.
    -   `dashboard/`: Contains code for the Streamlit dashboard.
        -   [`app.py`](src/dashboard/app.py): Main application file for the dashboard.
    -   `pipelines/`: Contains code for the data processing and model training pipelines.
        -   [`xgb.py`](src/pipelines/xgb.py): XGBoost pipeline.
        -   [`residue.py`](src/pipelines/residue.py): Residual pipeline.

## Model Performance
```
xgboost[after grid search]
Metric     Train           Test           
MSE        16.2983         37.7282        
MAE        2.2088          3.4257         
R2         0.8131          0.6748  

residual[dummy model]
Metric     Train           Test           
MSE        16.8397         35.8759
MAE        2.3359          3.3777
R2         0.8069          0.6908
```

## Installation

enable you have `uv` with `python 3.10` in your environment.

    ```bash
    uv init
    uv sync
    ```

## Usage

1.  Run the model training pipeline:
    ```bash
    uv run model_update.py --model residual
    ```

2.  Run the main.py to start the api server:

    ```bash
    PYTHONPATH=src uv run uvicorn main:app --reload --port 8000
    ```

3.  Run the Streamlit dashboard:

    ```bash
    uv run streamlit run src/dashboard/app.py
    ```
    

4.  Access the FastAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
5.  Access the Streamlit Dashboard: [http://localhost:8501](http://localhost:8501)

## Running Unit Tests

To run the unit tests, use the following command:

```bash
uv run pytest


## TODO

-   Apply Log (Units+1) conversation to make the prediction more robust
-   Scenarios analysis
-   Fix the predict API
-   Clean up the code; docstrings
-   Add more detailed documentation
-   Unit test
-   Dockerization for deployment
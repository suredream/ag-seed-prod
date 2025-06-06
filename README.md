# ag-seed-prod

## process

```
rm forecast_demo.db; uv run pred.py
uv run streamlit run forecast_explorer_app.py
uv run streamlit run dash.py
uv run uvicorn src/api:app --reload --port 8000 
```


## Perf
# (3.871694550208945, 45.068638115046674, 0.4429956793847415) one-hot
# 3.1283773639900305, 47.785631084564194, 0.4094163016533944, grid search, log
# (3.4393788435350467, 37.710091280289916, 0.5339401266064745) grid search
# {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}

MSE: 43.12749374926374
MAE: 3.787579685019346
R2: 0.46698632662637807

43.12749374926374 3.787579685019346 0.46698632662637807
37.110825059414765 3.7159928421717363 0.5413464714215577


## Todo
- scenarios analysis
- GeoAI output



## Structure
ag-seed-prod
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── docker-compose.yml
├── src/
│   ├── __init__.py
│   ├── chat_utils.py  # 通用聊天工具
│   ├── utils.py
│   ├── etl.py
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── xgb/
│   │   │   ├── __init__.py
│   │   │   ├── v1.py
│   ├── genai/                    # 轻量级GenAI模块
│   │   ├── __init__.py
│   │   ├── chat_service.py       # 单一聊天服务
│   │   ├── llm_wrapper.py        # LLM包装器
│   │   └── ml_context.py         # ML上下文增强
├── pipelines/
│   ├── __init__.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py
│   │   ├── hyperparameter_tuning.py
│   │   └── model_validation.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── etl_pipeline.py
│   │   └── feature_pipeline.py
│   └── monitoring/
│       ├── __init__.py
│       ├── model_monitoring.py
│       └── data_drift_detection.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│   │   └── routers/
│   │       └── chat.py           # 简化版聊天API
├── scripts/
│   ├── __init__.py
│   ├── setup_environment.py
│   ├── deploy_models.py
│   ├── run_training.py
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── pages/
│   │   └── chat.py              # 集成聊天页面
│   └── components/
│       └── embedded_chat.py     # 嵌入式聊天组件
├── notebooks/
├── tests/
├── docs/
│   ├── README.md
│   ├── api_documentation.md
│   ├── deployment_guide.md
│   ├── databricks_setup.md
│   └── mlflow_integration.md
└── data/
│   ├── case_study_data.csv

# 运行训练流水线
uv run streamlit run dashboard/app.py

# 运行训练流水线
python scripts/run_training.py --experiment-name "exp_001" --model-type "xgboost"

4. 监控和管理

访问Streamlit Dashboard: http://localhost:8501
访问FastAPI文档: http://localhost:8000/docs



train:
- create model
inference: shap
- ['pred']
- ['explain_shap']
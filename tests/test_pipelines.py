import pandas as pd
import toml
import pytest
import sys
import os

# Dynamically add project root (containing 'src') to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipelines.residue import preprocess_data, generate_pca_features, encode_categorical
from src.pipelines.explain import build_prompt, call_explain
import os
from dotenv import load_dotenv
load_dotenv()

class TestPipelines:
    @pytest.fixture
    def config(self):
        return toml.load("config/residual.toml")

    @pytest.fixture
    def sample_dataframe(self):
        return pd.read_csv('data/case_study_data.csv')

    def test_preprocess_data(self, sample_dataframe, config):
        df = preprocess_data(sample_dataframe.copy(), config)
        assert df is not None
        assert 'PROTECTION_SCORE' in df.columns
        assert 'PRODUCT_AGE' in df.columns
        assert 'PREVIOUS_UNITS' in df.columns

    def test_generate_pca_features(self, sample_dataframe, config):
        df = preprocess_data(sample_dataframe.copy(), config)
        df, scaler, pca = generate_pca_features(df, config)
        assert df is not None
        assert 'PCA1' in df.columns
        assert 'PCA2' in df.columns
        assert scaler is not None
        assert pca is not None

    def test_encode_categorical(self, sample_dataframe, config):
        df = preprocess_data(sample_dataframe.copy(), config)
        df, scaler, pca = generate_pca_features(df, config)
        df = encode_categorical(df, config)
        assert df is not None
        for col in config['categorical']['one_hot_columns_flat']:
            assert col in df.columns

    def test_build_prompt(self, sample_dataframe, config):
        df = preprocess_data(sample_dataframe.copy(), config)
        df, scaler, pca = generate_pca_features(df, config)
        df = encode_categorical(df, config)
        feature_series = df.iloc[0][config['features']['final']]
        names = config['features']['final']
        shap_vals = [0.1] * len(names)
        predictions = [1.0, 2.0, 3.0]
        prompt = build_prompt(feature_series, names, shap_vals, predictions)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

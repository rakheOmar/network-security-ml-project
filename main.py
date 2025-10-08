# streamlit_app.py
import io
import os
import sys

import certifi
import pandas as pd
import pymongo
import streamlit as st
from dotenv import load_dotenv

from src.constants.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.pipelines.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object
from src.utils.ml_utils.model.estimator import NetworkModel

ca = certifi.where()
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

try:
    client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
    database = client[DATA_INGESTION_DATABASE_NAME]
    collection = database[DATA_INGESTION_COLLECTION_NAME]
except Exception as e:
    logging.exception(e)
    collection = None

st.set_page_config(page_title="Network Security ML App", layout="wide")

st.title("Network Security ML App")
st.caption("Train pipeline and run CSV predictions with saved preprocessor/model.")

tab_train, tab_predict = st.tabs(["Train", "Predict"])

with tab_train:
    st.subheader("Train Pipeline")
    st.write("Run the end-to-end training pipeline using project settings.")
    if st.button("Start Training"):
        try:
            train_pipeline = TrainingPipeline()
            with st.spinner("Training in progress..."):
                train_pipeline.run_pipeline()
            st.success("Training is successful")
        except Exception as e:
            st.error(f"Training failed: {e}")
            raise NetworkSecurityException(e, sys)

with tab_predict:
    st.subheader("Predict from CSV")
    st.write("Upload a CSV with the expected feature columns for inference.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    preview_rows = st.number_input(
        "Preview first N rows", min_value=0, max_value=100, value=5, step=1
    )
    preproc_path = "final_model/preprocessor.pkl"
    model_path = "final_model/model.pkl"
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if preview_rows > 0:
                st.write("Preview:")
                st.dataframe(df.head(preview_rows), use_container_width=True)
            preprocessor = load_object(preproc_path)
            final_model = load_object(model_path)
            network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
            y_pred = network_model.predict(df)
            df_out = df.copy()
            df_out["predicted_column"] = y_pred
            os.makedirs("prediction_output", exist_ok=True)
            out_path = os.path.join("prediction_output", "output.csv")
            df_out.to_csv(out_path, index=False)
            st.success("Prediction complete. See table below and download CSV.")
            st.dataframe(df_out, use_container_width=True)
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download predictions CSV",
                data=csv_buf.getvalue(),
                file_name="output.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            raise NetworkSecurityException(e, sys)

with st.expander("Environment & Config"):
    st.write(f"MONGODB_URL_KEY is {'set' if mongo_db_url else 'missing'}")
    st.write(f"Preprocessor path: {preproc_path}")
    st.write(f"Model path: {model_path}")

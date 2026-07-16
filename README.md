# Segregated Automobile Intelligence Project

This package separates the trained intelligence system by use case and layer.

## Main Use Cases

1. Purchase & Sales Intelligence
   - Predicts future purchase and sales opportunities.
   - Folder: `models/purchase_sales_intelligence/`

2. Retention & Service Intelligence
   - Predicts customer loyalty and retention risk.
   - Folder: `models/retention_service_intelligence/`

3. Value & Financial Intelligence
   - Estimates customer value and financial behavior.
   - Folder: `models/value_financial_intelligence/`

4. Customer Engagement & Marketing Intelligence
   - Predicts the best opportunity for permission-based customer outreach.
   - Folder: `models/engagement_marketing_intelligence/`

## Run

```powershell
.\start_chatbot.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## Train all customer records

The trainer uses every row in `preprocessed_automobile_dataset.csv` by default, including the Customer Engagement & Marketing model. This may take significant time and memory on the 1.5M-row dataset.

```powershell
python training\train_models.py --csv .\preprocessed_automobile_dataset.csv
```

## Hugging Face Token

Put `.hf_token` in the project root, beside `backend`, `frontend`, and `models`. If this is the `WITH_HF_TOKEN` ZIP, the file is already included. You can also set:

```powershell
$env:HF_TOKEN = "hf_your_token"
```

## Notes

- `artifacts/intelligence_models.joblib` keeps the original combined model bundle used by the backend.
- Each use case also has an individual model file under `models/`.
- `training/train_models.py` contains the training workflow.
- `shared/ft_transformer.py` contains the FT-Transformer implementation.

# Project Structure

```text
segregated_purchase_sales_intelligence_project/
  backend/
    app.py
  frontend/
    index.html
  training/
    train_models.py
  shared/
    ft_transformer.py
  models/
    purchase_sales_intelligence/
      model.joblib
      metadata.json
    retention_service_intelligence/
      model.joblib
      metadata.json
    value_financial_intelligence/
      model.joblib
      metadata.json
  artifacts/
    intelligence_models.joblib
    metrics.json
  requirements.txt
  start_chatbot.ps1
  README.md
```

## Use Case Folders

- `models/purchase_sales_intelligence`: future purchase and sales opportunity model.
- `models/retention_service_intelligence`: loyalty / retention model.
- `models/value_financial_intelligence`: high-value financial behavior model.

Each use-case folder has its own `model.joblib` and `metadata.json`.

"""Slim IBM attrition FastAPI for Render Free (512 MB)."""

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

DATA_PATH = Path(__file__).parent / "model" / "data" / "ibm_hr_attrition.csv"
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).parent / "model" / "ibm_attrition_model.joblib"),
    )
)

DESCRIPTION = """
Human Resources API for teaching deploys on Render.

## Previews
* `/previews` returns random employee rows from the dataset (this is just previews)
* `/unique-values` lists valid values for a column (useful before predicting)

## Predict
* `/predict-attrition` predicts whether an employee is likely to leave

### How to fill the request body in `/docs`
Swagger shows a placeholder like `"additionalProp1": {}` — **delete that** and paste real
column names and values instead.

1. Call **GET `/previews?rows=1`** and copy the returned employee object.
2. Remove the `"Attrition"` field (that is the label the model predicts).
3. Wrap the remaining fields inside `"features": { ... }`.
4. Keep every other field, including `"Unnamed: 0"`.
5. Use exact text values for categorical fields (check with `/unique-values`).

The **POST `/predict-attrition`** endpoint also includes a ready-to-run example you can
load from the **Example value** dropdown when you click **Try it out**.
"""

app = FastAPI(
    title="HR Attrition API",
    description=DESCRIPTION,
    version="0.2",
    contact={"name": "HR API - by Jedha", "url": "https://jedha.co"},
)

df: pd.DataFrame | None = None
classifier = None
feature_names: list[str] = []


SAMPLE_EMPLOYEE_FEATURES = {
    "Unnamed: 0": 332,
    "Age": 54,
    "BusinessTravel": "Travel_Frequently",
    "DailyRate": 928,
    "Department": "Research & Development",
    "DistanceFromHome": 20,
    "Education": 4,
    "EducationField": "Life Sciences",
    "EmployeeCount": 1,
    "EmployeeNumber": 450,
    "EnvironmentSatisfaction": 4,
    "Gender": "Female",
    "HourlyRate": 31,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Research Scientist",
    "JobSatisfaction": 3,
    "MaritalStatus": "Single",
    "MonthlyIncome": 4869,
    "MonthlyRate": 16885,
    "NumCompaniesWorked": 3,
    "Over18": "Y",
    "OverTime": "No",
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 4,
    "StandardHours": 80,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 20,
    "TrainingTimesLastYear": 4,
    "WorkLifeBalance": 2,
    "YearsAtCompany": 4,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 3,
}


class AttritionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"features": SAMPLE_EMPLOYEE_FEATURES}]}
    )

    features: dict[str, Any] = Field(
        description=(
            "Employee fields expected by the model. Do not include `Attrition`. "
            "Easiest workflow: copy one row from GET /previews, remove `Attrition`, "
            "and paste the rest here."
        )
    )


@app.on_event("startup")
async def startup() -> None:
    global df, classifier, feature_names
    df = pd.read_csv(DATA_PATH)
    artifact = joblib.load(MODEL_PATH)
    classifier = artifact["model"]
    feature_names = list(artifact["feature_names"])


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rows": 0 if df is None else len(df),
        "model_loaded": classifier is not None,
        "model_path": str(MODEL_PATH),
        "app_env": os.environ.get("APP_ENV", "unset"),
    }


@app.get("/previews")
async def preview(rows: int = 10):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    sample = df.sample(min(rows, len(df)))
    return sample.to_dict(orient="records")


@app.get("/unique-values")
async def unique_values(column: str):
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded.")
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Unknown column: {column}")
    return {"column": column, "values": df[column].dropna().unique().tolist()}


@app.post("/predict-attrition")
async def predict_attrition(
    payload: AttritionInput = Body(
        ...,
        openapi_examples={
            "sample_employee": {
                "summary": "Ready-to-run employee example",
                "description": (
                    "Copy a row from GET /previews, remove Attrition, and wrap the "
                    "remaining fields in features. This example works as-is."
                ),
                "value": {"features": SAMPLE_EMPLOYEE_FEATURES},
            }
        },
    ),
):
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    input_df = pd.DataFrame([payload.features])
    missing = [col for col in feature_names if col not in input_df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    ordered = input_df[feature_names]
    prediction = classifier.predict(ordered)[0]
    probability = float(classifier.predict_proba(ordered)[0][1])
    return {
        "Attrition": "Yes" if int(prediction) == 1 else "No",
        "Probability": round(probability, 3),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

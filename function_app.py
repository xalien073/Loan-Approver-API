# function_app.py
import azure.functions as func 
import joblib  # Import joblib for model loading
import pandas as pd
from collections import OrderedDict
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load the model
model = joblib.load('loan_model.pkl')
scaler = joblib.load('standardscaler.pkl')
label_encoder_Gender = joblib.load('Label_encoder_Gender.pkl')
label_encoder_Married = joblib.load('Label_encoder_Married.pkl')
label_encoder_Loan_Status = joblib.load('Label_encoder_Loan_Status.pkl')

# Define input model
class LoanRequest(BaseModel):
    Gender: str
    Married: str
    ApplicantIncome: int
    LoanAmount: int

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male",
                "Married": "Yes",
                "ApplicantIncome": 3500,
                "LoanAmount": 100000
            }
        }

fast_app = FastAPI()

# origins = [#os.getenv("ALLOWED_ORIGINS")
#     "http://localhost:3000",
# ]

fast_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fast_app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Loan API!"}

@fast_app.post("/predict")
async def predict_loan_status(loan_data: LoanRequest):
    gender = loan_data.Gender
    married = loan_data.Married
    applicant_income = loan_data.ApplicantIncome
    loan_amount = loan_data.LoanAmount
    print(f'gender {gender}, married {married}, applicant_income {applicant_income}, loan_amount {loan_amount}')
    # Create a list of lists for model prediction
    inputs = [[gender], [married], [applicant_income], [loan_amount]]
    columns = ["Gender", "Married", "ApplicantIncome", "LoanAmount"]
    zipped_data = list(zip(columns, inputs))

    # Create DataFrame, perform preprocessing, and predict
    df = pd.DataFrame(OrderedDict(zipped_data))
    df_transform = df.copy()
    df_transform["Gender"] = label_encoder_Gender.transform(df_transform["Gender"])
    df_transform["Married"] = label_encoder_Married.transform(df_transform["Married"])
    df_transform = scaler.transform(df_transform)
    prediction = model.predict(df_transform)

    # Decode predicted loan status
    predicted_status = label_encoder_Loan_Status.inverse_transform(prediction)[0]

    return {"predicted_loan_status": predicted_status}

app = func.AsgiFunctionApp(app=fast_app,
                           http_auth_level=func.AuthLevel.ANONYMOUS)

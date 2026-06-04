from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

ROOT_DIR = Path(__file__).resolve().parents[3]

TARGET_COLUMN = "Credit_Score"

DROP_COLUMNS = [
    "ID",
    "Customer_ID",
    "Name",
    "SSN",
    "Month",
    "Type_of_Loan",
]

NUMERIC_STRING_COLUMNS = [
    "Age",
    "Annual_Income",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

FLOAT_COLUMNS = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Num_Credit_Inquiries",
]

INTEGER_COLUMNS = [
    "Age",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Credit_History_Age",
]

CATEGORICAL_COLUMNS = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]
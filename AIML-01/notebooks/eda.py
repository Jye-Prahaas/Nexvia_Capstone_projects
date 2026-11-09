import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/raw/Transactions.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

df.head()

# print(df.isnull().sum())
# print("Duplicate rows:", df.duplicated().sum())
# print(df["isFraud"].value_counts())

# print(
#     (df["isFraud"].value_counts(normalize=True) * 100).round(4)
# )


# Analyze transaction types

# print("Transaction type counts:")
# print(df["type"].value_counts())

# print("\nFraud cases by transaction type:")
# print(
#     df.groupby("type")["isFraud"]
#     .agg(["count", "sum"])
#     .rename(columns={
#         "count": "total_transactions",
#         "sum": "fraud_transactions"
#     })
# )







# fraud_by_type = df.groupby("type")["isFraud"].sum()

# fraud_by_type.plot(
#     kind="bar",
#     figsize=(9, 5)
# )

# plt.title("Fraudulent Transactions by Transaction Type")
# plt.xlabel("Transaction Type")
# plt.ylabel("Number of Fraudulent Transactions")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()




# fraud_analysis = df.groupby("type").agg(
#     total_transactions=("isFraud", "count"),
#     fraud_transactions=("isFraud", "sum")
# )

# fraud_analysis["fraud_percentage"] = (
#     fraud_analysis["fraud_transactions"]
#     / fraud_analysis["total_transactions"]
#     * 100
# )

# fraud_analysis = fraud_analysis.sort_values(
#     "fraud_percentage",
#     ascending=False
# )

# print(fraud_analysis.round(4))





# amount_analysis = df.groupby("isFraud")["amount"].agg(
#     ["count", "mean", "median", "min", "max"]
# )

# print(amount_analysis.round(2))






# plt.figure(figsize=(9, 5))

# sns.boxplot(
#     data=df,
#     x="isFraud",
#     y="amount"
# )

# plt.title("Transaction Amounts: Legitimate vs Fraudulent")
# plt.xlabel("Fraud Label (0 = Legitimate, 1 = Fraud)")
# plt.ylabel("Transaction Amount")

# plt.tight_layout()
# plt.show()




# balance_columns = [
#     "oldbalanceOrg",
#     "newbalanceOrig",
#     "oldbalanceDest",
#     "newbalanceDest"
# ]

# print(
#     df.groupby("isFraud")[balance_columns]
#     .mean()
#     .round(2)
# )






fraud_df = df.copy()

print("Working DataFrame created.")
print("Shape:", fraud_df.shape)

fraud_df["origin_balance_change"] = (
    fraud_df["oldbalanceOrg"] -
    fraud_df["newbalanceOrig"]
)

fraud_df["destination_balance_change"] = (
    fraud_df["newbalanceDest"] -
    fraud_df["oldbalanceDest"]
)

print(
    fraud_df[
        [
            "amount",
            "origin_balance_change",
            "destination_balance_change"
        ]
    ].head()
)







balance_change_analysis = fraud_df.groupby("isFraud")[
    [
        "origin_balance_change",
        "destination_balance_change"
    ]
].agg(["mean", "median"])

print(balance_change_analysis.round(2))
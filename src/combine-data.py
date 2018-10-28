import numpy as np
import pandas as pd

profile = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\user_profile.csv")
engagement = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\user_engagement.csv")

applies = False

extra = ".1" if applies else ""

x1 = [
    "count_tradelines_condition_derogatory",
    "count_open_installment_accounts_24_months",
    "count_tradelines_open_collection_accounts",
    "count_tradelines_open_mortgages",
    "count_tradelines_open_student_loans",
    "count_tradelines_opened_accounts",
    "count_tradelines_open_secured_loans",
    "count_tradelines_open_unsecured_loans",
    "total_cc_open_balance",
    "total_mortgage_loans_balance",
    "total_auto_loans_balance",
    "total_student_loans_balance",
    "count_inquiries_12_months",
    "count_bankruptcy"]


x = """count_tradelines_closed_accounts
count_total_tradelines_opened_24_months
count_tradelines_cc_opened_24_months
count_tradelines_condition_derogatory
count_open_installment_accounts_24_months
count_tradelines_open_collection_accounts
count_tradelines_open_mortgages
count_tradelines_open_student_loans
count_tradelines_opened_accounts
count_tradelines_open_secured_loans
count_tradelines_open_unsecured_loans
total_tradelines_amount_past_due
total_open_cc_amount_past_due
total_cc_open_balance
total_tradelines_open_balance
max_cc_limit
total_mortgage_loans_amount
total_mortgage_loans_balance
total_auto_loans_balance
total_student_loans_balance
count_inquiries_3_months
count_inquiries_6_months
count_inquiries_12_months
count_bankruptcy""".split("\n")

y = [
    "click_count_credit_card" + extra,
    "click_count_personal_loan" + extra,
    "click_count_mortgage" + extra,
    "click_count_credit_repair" + extra,
    "click_count_banking" + extra,
    "click_count_auto_products" + extra]

profile = profile[["user_id"] + x]

agecred = pd.read_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset\user_profile_age_credit.csv")

profile = pd.merge(profile, agecred)

profile = profile.set_index("user_id")

engagement = engagement[["user_id"] + y]

engagement = engagement.groupby(["user_id"]).agg(np.sum)

data = profile.join(engagement)

data = data[data[y].sum(axis=1) > 0]

data = data.loc[:, data.max() > 0]

data -= data.min()
data /= data.max()

data[["age_recoded", "credit_score_recoded"] + x].to_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset" +
                                                         "\\training-x2{}.csv".format(extra), index=False)

dy = data[y]
dy = dy.div(dy.sum(axis=1), axis=0)
dy.to_csv(r"C:\Users\Brian\Desktop\Programming\datathon\dataset" + "\\training-y2{}.csv".format(extra), index=False)


# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 22:08:59 2021

@author: CHITTIBABU
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf











###################______________________PART 1 IMPORTING ALL DATA SETS____________#########################_____________________________

#getting train and test data
train_data = pd.read_excel('train_Data.xlsx')
test_data = pd.read_excel('test_Data.xlsx')

#combining train and test data
full_data = pd.concat([train_data, test_data], axis=0)




# getting train and test from Bureau
train_b = pd.read_excel('train_bureau.xlsx')
test_b = pd.read_excel('test_bureau.xlsx')

#combining train and test bureau data
bureau_combined = pd.concat([train_b, test_b], axis=0)


replace_cols = ['JLG Individual', 'Business Loan General','Loan Against Bank Deposits','Used Car Loan','Business Loan Priority Sector  Small Business',
'Loan Against Shares / Securities',
'Property Loan',                                                            
'Mudra Loans   Shishu / Kishor / Tarun',
'Business Loan Priority Sector  Others',                                    
'Business Non-Funded Credit Facility-Priority Sector-Agriculture',           
'Non-Funded Credit Facility',                                                
'Education Loan',                                                           
'Business Loan - Secured',                                                   
'Business Loan Against Bank Deposits',                                       
'Individual',                                                                
'Business Loan Unsecured',                                                   
'Microfinance Business Loan',                                                
'Loan to Professional',                                                      
'Microfinance Others',                                                       
'Business Non-Funded Credit Facility-Priority Sector- Small Business',       
'Secured Credit Card',                                                        
'Business Non-Funded Credit Facility General',                                
'Prime Minister Jaan Dhan Yojana - Overdraft',                                
'Pradhan Mantri Awas Yojana - CLSS',                                          
'JLG Group',                                                                  
'SHG Individual',                                                             
'Microfinance Personal Loan',                                                 
'Fleet Card',                                                                 
'Commercial Equipment Loan',                                                  
'Microfinance Housing Loan',                                                  
'Corporate Credit Card',                                                       
'Loan on Credit Card',                                                         
'Leasing',                                                                     
'Business Non-Funded Credit Facility-Priority Sector-Others',                  
'Telco Landline',                                                              
'Staff Loan',                                                                  
'SHG Group']                    


















#############################_________PART 2 WORKING EXCLUSIVELY ON BUREAU DATA___________#####################____________________________


df_e = bureau_combined.copy()


id_account = df_e.groupby(['ID', 'ACCT-TYPE'])

id_accounts_count = id_account['ACCT-TYPE'].count().reset_index(name='count')

two_column_sort = id_accounts_count.sort_values(['ID', 'count'], ascending=[True, False])

cust_second_mode = two_column_sort.groupby('ID', as_index=False).nth(1)

cust_second_mode = cust_second_mode.drop('count', axis=1).rename(columns={'ACCT-TYPE': 'cust_second_mode'})

cust_third_mode = two_column_sort.groupby('ID', as_index=False).nth(2)

cust_third_mode = cust_third_mode.drop('count', axis=1).rename(columns={'ACCT-TYPE': 'cust_third_mode'})

cust_fourth_mode = two_column_sort.groupby('ID', as_index=False).nth(3)

cust_fourth_mode = cust_fourth_mode.drop('count', axis=1).rename(columns={'ACCT-TYPE': 'cust_fourth_mode'})

cust_fifth_mode = two_column_sort.groupby('ID', as_index=False).nth(4)

cust_fifth_mode = cust_fifth_mode.drop('count', axis=1).rename(columns={'ACCT-TYPE': 'cust_fifth_mode'})





id_contrib = df_e.groupby(['ID', 'CONTRIBUTOR-TYPE'])

id_contrib_count = id_contrib['CONTRIBUTOR-TYPE'].count().reset_index(name='count')

contrib_sort = id_contrib_count.sort_values(['ID', 'count'], ascending=[True, False])

contrib_second_mode = contrib_sort.groupby('ID', as_index=False).nth(1)

contrib_second_mode = contrib_second_mode.drop('count', axis=1).rename(columns={'CONTRIBUTOR-TYPE': 'contrib_second_mode'})

contrib_third_mode = contrib_sort.groupby('ID', as_index=False).nth(2)

contrib_third_mode = contrib_third_mode.drop('count', axis=1).rename(columns={'CONTRIBUTOR-TYPE': 'contrib_third_mode'})

contrib_fourth_mode = contrib_sort.groupby('ID', as_index=False).nth(3)

contrib_fourth_mode = contrib_fourth_mode.drop('count', axis=1).rename(columns={'CONTRIBUTOR-TYPE': 'contrib_fourth_mode'})






#Account status counts
delinq = (df_e.groupby(['ID','ACCOUNT-STATUS'])
       .apply(lambda x: (x['ACCOUNT-STATUS'] == 'Delinquent').sum()).reset_index(name='count'))

delinq_group = delinq.groupby('ID')

delinq_accounts = pd.DataFrame(delinq_group['count'].sum()).rename(columns={'count': 'delinq_accounts'})


active = (df_e.groupby(['ID','ACCOUNT-STATUS'])
       .apply(lambda x: (x['ACCOUNT-STATUS'] == 'Active').sum()).reset_index(name='count'))

active_group = active.groupby('ID')

active_accounts = pd.DataFrame(active_group['count'].sum()).rename(columns={'count': 'active_accounts'})


closed = (df_e.groupby(['ID','ACCOUNT-STATUS'])
       .apply(lambda x: (x['ACCOUNT-STATUS'] == 'Closed').sum()).reset_index(name='count'))

closed_group = closed.groupby('ID')

closed_accounts = pd.DataFrame(closed_group['count'].sum()).rename(columns={'count': 'closed_accounts'})





#Account Type counts
tractor = (df_e.groupby(['ID','ACCT-TYPE'])
       .apply(lambda x: (x['ACCT-TYPE'] == 'Tractor Loan').sum()).reset_index(name='count'))

tractor_group = tractor.groupby('ID')

tractor_accounts = pd.DataFrame(tractor_group['count'].sum()).rename(columns={'count': 'tractor_accounts'})


gold = (df_e.groupby(['ID','ACCT-TYPE'])
       .apply(lambda x: (x['ACCT-TYPE'] == 'Gold Loan').sum()).reset_index(name='count'))

gold_group = gold.groupby('ID')

gold_accounts = pd.DataFrame(gold_group['count'].sum()).rename(columns={'count': 'gold_accounts'})


agbs = (df_e.groupby(['ID','ACCT-TYPE'])
       .apply(lambda x: (x['ACCT-TYPE'] == 'Business Loan Priority Sector  Agriculture').sum()).reset_index(name='count'))

agbs_group = agbs.groupby('ID')

agbs_accounts = pd.DataFrame(agbs_group['count'].sum()).rename(columns={'count': 'agbs_accounts'})


#Contributor type counts
nab = (df_e.groupby(['ID','CONTRIBUTOR-TYPE'])
       .apply(lambda x: (x['CONTRIBUTOR-TYPE'] == 'NAB').sum()).reset_index(name='count'))

nab_group = nab.groupby('ID')

nab_accounts = pd.DataFrame(nab_group['count'].sum()).rename(columns={'count': 'nab_accounts'})


nbf = (df_e.groupby(['ID','CONTRIBUTOR-TYPE'])
       .apply(lambda x: (x['CONTRIBUTOR-TYPE'] == 'NBF').sum()).reset_index(name='count'))

nbf_group = nbf.groupby('ID')

nbf_accounts = pd.DataFrame(nbf_group['count'].sum()).rename(columns={'count': 'nbf_accounts'})


prb = (df_e.groupby(['ID','CONTRIBUTOR-TYPE'])
       .apply(lambda x: (x['CONTRIBUTOR-TYPE'] == 'PRB').sum()).reset_index(name='count'))

prb_group = prb.groupby('ID')

prb_accounts = pd.DataFrame(prb_group['count'].sum()).rename(columns={'count': 'prb_accounts'})

#L&T accounts count
others = (df_e.groupby(['ID','SELF-INDICATOR'])
       .apply(lambda x: (x['SELF-INDICATOR'] == False).sum()).reset_index(name='count'))

other_group = others.groupby('ID')

other_accounts = pd.DataFrame(other_group['count'].sum()).rename(columns={'count': 'other_accounts'})





id_group = df_e.groupby('ID')

total_accounts = pd.DataFrame(id_group['ID'].count()).rename(columns={'ID': 'total_accounts'})

df_e['DISBURSED-AMT/HIGH CREDIT'] = df_e['DISBURSED-AMT/HIGH CREDIT'].str.replace(',','').astype(float)

cust_disb_amt = pd.DataFrame(id_group['DISBURSED-AMT/HIGH CREDIT'].sum()).rename(columns={'DISBURSED-AMT/HIGH CREDIT': 'cust_disb_amt'})

#df_e['ACCT-TYPE'] = df_e['ACCT-TYPE'].replace(to_replace=replace_cols, value='Other')








#Working with dates
id_date_group = df_e.groupby('ID')

# taking sum of all numerical columns for ID group
first_loan = id_date_group['DISBURSED-DT'].min()
last_contact = id_date_group['DATE-REPORTED'].max()


years_diff = np.array(last_contact - first_loan)

days = years_diff.astype('timedelta64[D]')

cust_cred_hist = pd.DataFrame(days / np.timedelta64(1, 'D')).rename(columns={0: 'cust_cred_hist'})






cust_report_hist = pd.DataFrame(df_e['REPORTED DATE - HIST'].str.split(',')).rename(columns={'REPORTED DATE - HIST': 'cust_report_hist'})

cust_report_hist['cust_report_hist'] = cust_report_hist['cust_report_hist'].replace(to_replace=np.nan, value=0)

cust_total_reports = []


for i in range(0, len(cust_report_hist)):
    if cust_report_hist.iloc[i, 0] == 0:
        cust_total_reports.append(0)
    else:
        cust_total_reports.append(len(cust_report_hist.iloc[i, 0]))
        
      






       

cust_amt_due_col = df_e['AMT OVERDUE - HIST'].str.split(',')

cust_amt_due_col = pd.DataFrame(cust_amt_due_col)

cust_amt_due_col['AMT OVERDUE - HIST'] = cust_amt_due_col['AMT OVERDUE - HIST'].replace(to_replace=np.nan, value=0)

for i in range(0, len(cust_amt_due_col)):
    if cust_amt_due_col.iloc[i, 0] != 0:
        cust_amt_due_col.iloc[i, 0] = [string for string in cust_amt_due_col.iloc[i, 0] if string != ""]
        

cust_amt_due_hist = []
cust_delayed_due_payments = []
cust_current_due = []

# adding all the numerical values in each cell
for i in range(0, len(cust_amt_due_col)):
    if cust_amt_due_col.iloc[i, 0] == 0:
        cust_amt_due_hist.append(0)
        cust_delayed_due_payments.append(0)
    elif len(cust_amt_due_col.iloc[i, 0]) == 0:
        cust_amt_due_hist.append(0)
        cust_delayed_due_payments.append(0)
    else:
        cust_amt_due_col.iloc[i, 0] = [float(numeric_string) for numeric_string in cust_amt_due_col.iloc[i, 0]]
        cust_amt_due_hist.append(np.mean(cust_amt_due_col.iloc[i, 0]))
        cust_delayed_due_payments.append(np.count_nonzero(cust_amt_due_col.iloc[i, 0]))
        cust_current_due.append(cust_amt_due_col.iloc[i, 0][0])









# CURRENT BAL HISTORY CLEAN UP
# separating values based on commas
cust_cur_bal_col = df_e['CUR BAL - HIST'].str.split(',')

cust_cur_bal_col = pd.DataFrame(cust_cur_bal_col)

cust_cur_bal_col['CUR BAL - HIST'] = cust_cur_bal_col['CUR BAL - HIST'].replace(to_replace=np.nan, value=0)

for i in range(0, len(cust_cur_bal_col)):
    if cust_cur_bal_col.iloc[i, 0] != 0:
        cust_cur_bal_col.iloc[i, 0] = [string for string in cust_cur_bal_col.iloc[i, 0] if string != ""]
        

cust_cur_bal_hist = []

# adding all the numerical values in each cell
for i in range(0, len(cust_cur_bal_col)):
    if cust_cur_bal_col.iloc[i, 0] == 0:
        cust_cur_bal_hist.append(0)
    elif len(cust_cur_bal_col.iloc[i, 0]) == 0:
        cust_cur_bal_hist.append(0)
    else:
        cust_cur_bal_col.iloc[i, 0] = [float(numeric_string) for numeric_string in cust_cur_bal_col.iloc[i, 0]]
        cust_cur_bal_hist.append(np.mean(cust_cur_bal_col.iloc[i, 0]))
        




        
cust_amt_paid_col = df_e['AMT PAID - HIST'].str.split(',')

cust_amt_paid_col = pd.DataFrame(cust_amt_paid_col)

cust_amt_paid_col['AMT PAID - HIST'] = cust_amt_paid_col['AMT PAID - HIST'].replace(to_replace=np.nan, value=0)

for i in range(0, len(cust_amt_paid_col)):
    if cust_amt_paid_col.iloc[i, 0] != 0:
        cust_amt_paid_col.iloc[i, 0] = [string for string in cust_amt_paid_col.iloc[i, 0] if string != ""]
        

cust_amt_paid_hist = []

# adding all the numerical values in each cell
for i in range(0, len(cust_amt_paid_col)):
    if cust_amt_paid_col.iloc[i, 0] == 0:
       cust_amt_paid_hist.append(0)
    elif len(cust_amt_paid_col .iloc[i, 0]) == 0:
        cust_amt_paid_hist.append(0)
    else:
        cust_amt_paid_col .iloc[i, 0] = [float(numeric_string) for numeric_string in cust_amt_paid_col .iloc[i, 0]]
        cust_amt_paid_hist.append(np.sum(cust_amt_paid_col .iloc[i, 0]))
        


cust_total_reports = pd.DataFrame(cust_total_reports).rename(columns={0: 'cust_total_reports'}).astype(int)

cust_cur_bal_hist = pd.DataFrame(cust_cur_bal_hist).rename(columns={0: 'cust_cur_bal_hist'})

cust_amt_due_hist = pd.DataFrame(cust_amt_due_hist).rename(columns={0: 'cust_amt_due_hist'})

cust_amt_paid_hist = pd.DataFrame(cust_amt_paid_hist).rename(columns={0: 'cust_amt_paid_hist'})

cust_current_due = pd.DataFrame(cust_current_due ).rename(columns={0: 'cust_current_due'})

cust_delayed_due_payments = pd.DataFrame(cust_delayed_due_payments).rename(columns={0: 'cust_payment_delays'})


df_e.reset_index(drop=True, inplace=True)
cust_total_reports.reset_index(drop=True, inplace=True)
cust_cur_bal_hist.reset_index(drop=True, inplace=True)
cust_amt_due_hist.reset_index(drop=True, inplace=True)
cust_amt_paid_hist.reset_index(drop=True, inplace=True)
cust_current_due.reset_index(drop=True, inplace=True)
cust_delayed_due_payments.reset_index(drop=True, inplace=True)


cust_cred_hist['cust_cred_hist'] = cust_cred_hist['cust_cred_hist'].fillna(np.mean(cust_cred_hist['cust_cred_hist']))

cust_total_reports['cust_total_reports'] = cust_total_reports['cust_total_reports'].fillna(np.mean(cust_total_reports['cust_total_reports']))

cust_cur_bal_hist['cust_cur_bal_hist'] = cust_cur_bal_hist['cust_cur_bal_hist'].fillna(np.mean(cust_cur_bal_hist['cust_cur_bal_hist']))

cust_amt_due_hist['cust_amt_due_hist'] = cust_amt_due_hist['cust_amt_due_hist'].fillna(np.mean(cust_amt_due_hist['cust_amt_due_hist']))

cust_amt_paid_hist['cust_amt_paid_hist'] = cust_amt_paid_hist['cust_amt_paid_hist'].fillna(np.mean(cust_amt_paid_hist['cust_amt_paid_hist']))

cust_delayed_due_payments['cust_payment_delays'] = cust_delayed_due_payments['cust_payment_delays'].fillna(np.mean(cust_delayed_due_payments['cust_payment_delays']))



concat_data = pd.concat([df_e, cust_total_reports, cust_cur_bal_hist, cust_amt_due_hist, cust_amt_paid_hist,
                            cust_current_due, cust_delayed_due_payments], axis=1)




concat_data = concat_data.drop(['WRITE-OFF-AMT', 'TENURE', 'SELF-INDICATOR','DISBURSED-AMT/HIGH CREDIT', 'DISBURSED-DT',
              'DPD - HIST', 'INSTALLMENT-AMT', 'INSTALLMENT-FREQUENCY', 'LAST-PAYMENT-DATE',
              'CURRENT-BAL', 'CREDIT-LIMIT/SANC AMT', 'ASSET_CLASS', 'AMT PAID - HIST', 'ACCOUNT-STATUS',
              'REPORTED DATE - HIST', 'DATE-REPORTED', 'CLOSE-DT','CUR BAL - HIST',
              'OVERDUE-AMT', 'AMT OVERDUE - HIST', 'cust_current_due'], axis=1)

concat_data.dtypes

#Separating columns based on the data type
res = concat_data.dtypes.apply(lambda x: x.name).to_dict()

cat_cols, num_cols, bool_cols = [], [], []

for key in res.keys():
    if res[key] == 'object':
        cat_cols.append(key)
    else:
        num_cols.append(key)
        
# importing mode
from scipy.stats import skew, mode

# grouping based on ID        
id_grouping = concat_data.groupby('ID')

# taking sum of all numerical columns for ID group
num_grouping = id_grouping[num_cols].sum()

# taking mode of all categorical columns for ID group
cat_grouping = id_grouping[cat_cols].agg(mode)

# unpacking mode
for i in range(0, len(cat_grouping)):
    for j in range(0, len(cat_cols)):
        cat_grouping.iloc[i, j] = cat_grouping.iloc[i, j][0][0]


num_grouping = num_grouping.drop('ID', axis=1)


cat_grouping.reset_index(drop=True, inplace=True)




num_grouping.reset_index(drop=True, inplace=True)
total_accounts.reset_index(drop=True, inplace=True)
cust_cred_hist.reset_index(drop=True, inplace=True)





bureau_cleaned = pd.concat([cat_grouping, num_grouping, total_accounts, cust_cred_hist,
                            delinq_accounts, active_accounts, closed_accounts,
                            tractor_accounts, gold_accounts, agbs_accounts,
                            nab_accounts, cust_disb_amt, nbf_accounts, prb_accounts, other_accounts], axis=1)

bureau_cleaned['ID'] = range(1, 1+len(bureau_cleaned))

















################################______PART 3 MERGING L&T Original and Bureau L&T_________################_____________


#making bureau copy
df_b = bureau_combined.copy()

#getting all L&T accounts from bureau
df_lt = df_b[df_b['SELF-INDICATOR'] == True]

#merging L&T bureau data with L&T original data with inner join
combined_lt = pd.merge(full_data, df_lt, left_on=['ID', 'DisbursalDate'], right_on=['ID', 'DISBURSED-DT'] )

combined_lt['ACCT-TYPE'].isnull().sum(axis=0)

#diff in L&T cols and bureau L&T cols
cols_to_use = combined_lt.columns.difference(full_data.columns)

cols_to_merge = []

for i in cols_to_use:
    cols_to_merge.append(i)
   
    
cols_to_merge.append('ID')

# left join to get the remaining data where disbursal date is missing in bureau L&T
full_lt = pd.merge(full_data , combined_lt[cols_to_merge], how='left', on='ID')

# droping the duplicates that came with join
full_lt.drop_duplicates(subset = 'ID', inplace = True) 


#Creating new features from the existing L&T Original data
full_lt['disb_year'] = pd.DatetimeIndex(full_lt['DisbursalDate']).year

full_lt['mat_year'] = pd.DatetimeIndex(full_lt['MaturityDAte']).year

full_lt['PaymentMode'] = full_lt['PaymentMode'].replace(to_replace=['Direct Debit', 'Auto Debit'], value='ECS')

full_lt['PaymentMode'] = full_lt['PaymentMode'].replace(to_replace=['Cheque'], value='PDC')



#copying into a new dataframe
df = full_lt.copy()


df['Branch_status'] = pd.cut(df['BranchID'], bins=[1,106,212,424], labels=['oldest', 'old', 'new'])

df['ManufacturerID'] = pd.cut(df['ManufacturerID'], bins=[0,1500,4000], labels=['heavy', 'light'])



full_cols = df.columns

#dropping some obvious columns from L&T original data part
df = df.drop(['Area', 'DisbursalDate', 'MaturityDAte', 'AuthDate', 'AssetID',
              'SupplierID', 'City', 'ZiPCODE', 'ManufacturerID', 'BranchID'], axis=1)


#Creating new features from exisitng bureau L&T data part
lt_report_hist = pd.DataFrame(df['REPORTED DATE - HIST'].str.split(',')).rename(columns={'REPORTED DATE - HIST': 'lt_report_hist'})
lt_report_hist['lt_report_hist'] = lt_report_hist['lt_report_hist'].replace(to_replace=np.nan, value=0)

lt_reports = []

for i in range(0, len(lt_report_hist)):
    if lt_report_hist.iloc[i, 0] == 0:
        lt_reports.append(0)
    else:
        lt_reports.append(len(lt_report_hist.iloc[i, 0]))
              




cur_bal_col = df['CUR BAL - HIST'].str.split(',')

cur_bal_col = pd.DataFrame(cur_bal_col)

cur_bal_col['CUR BAL - HIST'] = cur_bal_col['CUR BAL - HIST'].replace(to_replace=np.nan, value=0)

for i in range(0, len(cur_bal_col)):
    if cur_bal_col.iloc[i, 0] != 0:
        cur_bal_col.iloc[i, 0] = [string for string in cur_bal_col.iloc[i, 0] if string != ""]
        
cur_bal_hist = []

# adding all the numerical values in each cell
for i in range(0, len(cur_bal_col)):
    if cur_bal_col.iloc[i, 0] == 0:
        cur_bal_hist.append(0)
    elif len(cur_bal_col.iloc[i, 0]) == 0:
        cur_bal_hist.append(0)
    else:
        cur_bal_col.iloc[i, 0] = [float(numeric_string) for numeric_string in cur_bal_col.iloc[i, 0]]
        cur_bal_hist.append(cur_bal_col.iloc[i, 0][0])
        
        
        
amt_due_col = df['AMT OVERDUE - HIST'].str.split(',')

amt_due_col = pd.DataFrame(amt_due_col)

amt_due_col['AMT OVERDUE - HIST'] = amt_due_col['AMT OVERDUE - HIST'].replace(to_replace=np.nan, value=0)

for i in range(0, len(amt_due_col)):
    if amt_due_col.iloc[i, 0] != 0:
        amt_due_col.iloc[i, 0] = [string for string in amt_due_col.iloc[i, 0] if string != ""]
        

amt_due_hist = []
delayed_due_payments = []
lt_due = []

# adding all the numerical values in each cell
for i in range(0, len(amt_due_col)):
    if amt_due_col.iloc[i, 0] == 0:
        amt_due_hist.append(0)
        delayed_due_payments.append(0)
    elif len(amt_due_col.iloc[i, 0]) == 0:
        amt_due_hist.append(0)
        delayed_due_payments.append(0)
    else:
        amt_due_col.iloc[i, 0] = [float(numeric_string) for numeric_string in amt_due_col.iloc[i, 0]]
        amt_due_hist.append(np.mean(amt_due_col.iloc[i, 0]))
        delayed_due_payments.append(np.count_nonzero(amt_due_col.iloc[i, 0]))
        lt_due.append(amt_due_col.iloc[i, 0][0])


df['lt_date_report'] = pd.DatetimeIndex(df['DATE-REPORTED']).year

df['lt_date_close'] = pd.DatetimeIndex(df['CLOSE-DT']).year







#similarly dropping some obvious columns from bureau L&T data part

df = df.drop(['WRITE-OFF-AMT', 'TENURE', 'SELF-INDICATOR','DISBURSED-AMT/HIGH CREDIT', 'DISBURSED-DT',
              'DPD - HIST', 'INSTALLMENT-AMT', 'INSTALLMENT-FREQUENCY', 'LAST-PAYMENT-DATE',
              'CURRENT-BAL', 'CREDIT-LIMIT/SANC AMT', 'ASSET_CLASS', 'AMT PAID - HIST', 'ACCOUNT-STATUS',
              'REPORTED DATE - HIST', 'DATE-REPORTED', 'CLOSE-DT','CUR BAL - HIST',
              'OVERDUE-AMT', 'AMT OVERDUE - HIST'], axis=1)




df['EMI'] = df['EMI'].fillna(np.mean(df['EMI']))


corrected_emi = []


for i in range(0, len(df)):
    if df.iloc[i, 1] == 'Quatrly':
        corrected_emi.append(df.iloc[i, 9]/4)
    elif df.iloc[i, 1] == 'Half Yearly':
        corrected_emi.append(df.iloc[i, 9]/6)
    elif df.iloc[i, 1] == 'BI-Monthly':
        corrected_emi.append(df.iloc[i, 9]/2)
    else:
        corrected_emi.append(df.iloc[i, 9])

corrected_emi = pd.DataFrame(corrected_emi).rename(columns={0: 'corrected_emi'})


df.reset_index(drop=True, inplace=True)
corrected_emi.reset_index(drop=True, inplace=True)



df = pd.concat([df, corrected_emi], axis=1)

total_payable = df['corrected_emi'] * df['Tenure']

interest_paid = total_payable - df['AmountFinance']

df['extra_paid_percent'] = interest_paid / df['AmountFinance']

df['extra_paid_percent'].describe()

df['corrected_interest'] = df['corrected_emi'] / df['AmountFinance']








# dropping the below columns as they have high correlation with other columns(LTV)
df = df.drop(['AmountFinance', 'DisbursalAmount'], axis=1)


#Concatenating all the additional columns
cur_bal_hist = pd.DataFrame(cur_bal_hist).rename(columns={0: 'cur_bal_hist'})

amt_due_hist = pd.DataFrame(amt_due_hist).rename(columns={0: 'amt_due_hist'})

delayed_due_payments = pd.DataFrame(delayed_due_payments).rename(columns={0: 'lt_payment_delays'})

lt_due = pd.DataFrame(lt_due).rename(columns={0: 'lt_due'})

lt_reports = pd.DataFrame(lt_reports).rename(columns={0: 'lt_reports'}).astype(int)


cur_bal_hist.reset_index(drop=True, inplace=True)
amt_due_hist.reset_index(drop=True, inplace=True)
delayed_due_payments.reset_index(drop=True, inplace=True)
lt_due.reset_index(drop=True, inplace=True)
total_accounts.reset_index(drop=True, inplace=True)
lt_reports.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)



combined_cleaned = pd.concat([df, cur_bal_hist, amt_due_hist, delayed_due_payments, lt_due, lt_reports], axis=1)

bureau_cleaned = pd.DataFrame(bureau_cleaned).rename(columns={'ACCT-TYPE': 'CUST_MODE_ACCT'})

bureau_cleaned = pd.DataFrame(bureau_cleaned).rename(columns={'CONTRIBUTOR-TYPE': 'CUST_CONTRIBUTOR_TYPE'})

bureau_cleaned = pd.DataFrame(bureau_cleaned).rename(columns={'MATCH-TYPE': 'CUST_MATCH_TYPE'})

bureau_cleaned = pd.DataFrame(bureau_cleaned).rename(columns={'OWNERSHIP-IND': 'CUST_OWNERSHIP_IND'})

df2 = pd.merge(combined_cleaned, bureau_cleaned, on='ID')

df2 = pd.merge(df2, cust_second_mode, on='ID', how='left')

df2 = pd.merge(df2, cust_third_mode, on='ID', how='left')

df2 = pd.merge(df2, cust_fourth_mode, on='ID', how='left')

df2 = pd.merge(df2, cust_fifth_mode, on='ID', how='left')

df2 = pd.merge(df2, contrib_second_mode, on='ID', how='left')

df2 = pd.merge(df2, contrib_third_mode, on='ID', how='left')

df2 = pd.merge(df2, contrib_fourth_mode, on='ID', how='left')








tenure_column = ['Tenure']
tenure_upper_lim = df2['Tenure'].quantile(.99)
df2.loc[(df2['Tenure'] > tenure_upper_lim), tenure_column] = tenure_upper_lim

AssetCost_column = ['AssetCost']
AssetCost_upper_lim = df2['AssetCost'].quantile(.99)
df2.loc[(df2['AssetCost'] > AssetCost_upper_lim), AssetCost_column] = AssetCost_upper_lim

emi_column = ['EMI']
emi_upper_lim = df2['EMI'].quantile(.99)
df2.loc[(df2['EMI'] > emi_upper_lim), emi_column] = emi_upper_lim

ltv_column = ['LTV']
ltv_upper_lim = df2['LTV'].quantile(.99)
df2.loc[(df2['LTV'] > ltv_upper_lim), ltv_column] = ltv_upper_lim

age_column = ['AGE']
age_upper_lim = df2['AGE'].quantile(.99)
df2.loc[(df2['AGE'] > age_upper_lim), age_column] = age_upper_lim

MonthlyIncome_column = ['MonthlyIncome']
MonthlyIncome_upper_lim = df2['MonthlyIncome'].quantile(.99)
df2.loc[(df2['MonthlyIncome'] > MonthlyIncome_upper_lim), MonthlyIncome_column] = MonthlyIncome_upper_lim




df2 = df2.drop('ID', axis=1)



# Checking for missing values in all the columns
miss_vals = pd.DataFrame(df.isnull().sum(axis=0))

miss_vals_cols = miss_vals[miss_vals[0] > 0]

# Filling the missing values
df2['SEX'] = df2['SEX'].fillna('M')

df2['AGE'] = df2['AGE'].fillna(np.mean(df2['AGE']))

df2['MonthlyIncome'] = df2['MonthlyIncome'].fillna(np.mean(df2['MonthlyIncome']))

df2['cust_cred_hist'] = df2['cust_cred_hist'].fillna(np.mean(df2['cust_cred_hist']))


df2['lt_date_report'].value_counts()
df2['lt_date_report'] = df2['lt_date_report'].fillna(2020)

df2['mat_year'].value_counts()
df2['mat_year'] = df2['mat_year'].fillna(2021)

df2['lt_date_close'] = df2['lt_date_close'].fillna(0)

# dropping few other irrelevant columns
df2 = df2.drop(['CONTRIBUTOR-TYPE', 'MATCH-TYPE', 'OWNERSHIP-IND'], axis=1)





df2['ACCT-TYPE'].value_counts()
df2['ACCT-TYPE'] = df2['ACCT-TYPE'].fillna('Tractor Loan')

df2['lt_due'].value_counts()
df2['lt_due'] = df2['lt_due'].fillna(0)






df2_cols = df2.columns

df2.isnull().sum(axis=0)



df2['lt_accounts'] = df2['total_accounts'] - df2['other_accounts']



# correlation drops
df2 = df2.drop(['cust_total_reports', 'mat_year', 'cur_bal_hist', 
                'lt_date_report', 'cust_payment_delays', 'closed_accounts', 'nab_accounts', 'other_accounts', 'EMI'], axis=1)





#temp fix
df2['delinq_accounts'] = df2['delinq_accounts'].fillna(np.mean(df2['delinq_accounts']))
df2['active_accounts'] = df2['active_accounts'].fillna(np.mean(df2['active_accounts']))
df2['tractor_accounts'] = df2['tractor_accounts'].fillna(np.mean(df2['tractor_accounts']))
df2['gold_accounts'] = df2['gold_accounts'].fillna(np.mean(df2['gold_accounts']))
df2['agbs_accounts'] = df2['agbs_accounts'].fillna(np.mean(df2['agbs_accounts']))
df2['cust_disb_amt'] = df2['cust_disb_amt'].fillna(np.mean(df2['cust_disb_amt']))
df2['nbf_accounts'] = df2['nbf_accounts'].fillna(np.mean(df2['nbf_accounts']))
df2['prb_accounts'] = df2['prb_accounts'].fillna(np.mean(df2['prb_accounts']))
df2['lt_accounts'] = df2['lt_accounts'].fillna(np.mean(df2['lt_accounts']))


df2['cust_second_mode'] = df2['cust_second_mode'].fillna('single_account')
df2['cust_third_mode'] = df2['cust_third_mode'].fillna('two_accounts')
df2['cust_fourth_mode'] = df2['cust_fourth_mode'].fillna('three_accounts')
df2['cust_fifth_mode'] = df2['cust_fifth_mode'].fillna('four_accounts')
df2['contrib_second_mode'] = df2['contrib_second_mode'].fillna('single_account')
df2['contrib_third_mode'] = df2['contrib_third_mode'].fillna('two_accounts')
df2['contrib_fourth_mode'] = df2['contrib_fourth_mode'].fillna('three_accounts')

df2.isnull().sum(axis=0)

df2['Branch_status'] = df2['Branch_status'].fillna('oldest')




























###################________PART 4 BULIDING MODEL FOR NORMAL CLASSIFICATION___########################_______________________

#Normal Classification
df_n = df2.copy()


# creating the features set for model building
features_set = df_n.drop(['Top-up Month'], axis=1)

features_set = pd.get_dummies(features_set, drop_first=True)

train_set = features_set.iloc[:len(train_data), :]

test_set = features_set.iloc[len(train_data):, :]

X_n = train_set.copy()   

X_n_miss = X_n.isnull().sum(axis=0)

Y_n = df_n[df_n['Top-up Month'].notnull()]

Y_n = Y_n['Top-up Month']




ncols = X_n.columns


#Scaling data
from sklearn.preprocessing import MinMaxScaler

mms_n = MinMaxScaler()

X_n = mms_n.fit_transform(X_n)

# Selecting the best features from the all
from sklearn.feature_selection import SelectKBest, chi2

kbest_n = SelectKBest(score_func=chi2, k='all')

kbest_n.fit_transform(X_n, Y_n)

npvalues = kbest_n.pvalues_

nf_scores = kbest_n.scores_

np_cols = pd.DataFrame(npvalues).rename(columns={0: 'np_cols'})

nf_cols = pd.DataFrame(nf_scores).rename(columns={0: 'nf_cols'})

nbest_features = nf_cols['nf_cols'].sort_values()

# taking columns with 95% confidence interval
nworking_cols = np_cols[(np_cols['np_cols'] < 0.05)]

nworking_cols_index = nworking_cols.index.values.tolist()

nselected_cols = ncols[nworking_cols_index]

#Finally selecting relevant features for ML model
X_m_n = train_set.copy()

X_k_n = X_m_n.loc[:, nselected_cols]







from sklearn.model_selection import train_test_split

nx_train, nx_test, ny_train, ny_test = train_test_split(X_k_n, Y_n, test_size = 0.3, random_state=1234, stratify=Y_n)

mms_n2 = MinMaxScaler()

nx_train = mms_n2.fit_transform(nx_train)

nx_test = mms_n2.transform(nx_test)

X_k_n = mms_n2.fit_transform(X_k_n)





# Best model 
from xgboost import XGBClassifier

rfc_n = XGBClassifier(silent=False, 
                      learning_rate=1,  
                      colsample_bytree = 0.9,
                      subsample = 1,
                      objective='multi:softprob', 
                      n_estimators=20, 
                      reg_alpha = 0.9,
                      max_depth=9, 
                      gamma=5)




#rfc_n.fit(nx_train, ny_train)

rfc_n.fit(X_k_n, Y_n)

ny_predict_rfc = rfc_n.predict(nx_test)



ny_predict_proba = rfc_n.predict_proba(nx_test)

from sklearn.metrics import classification_report, confusion_matrix

ncr_rfc = classification_report(ny_test, ny_predict_rfc)

ncm_rfc = confusion_matrix(ny_test, ny_predict_rfc)

nscore_rfc = rfc_n.score(nx_test, ny_test)


























#################____________PART 5 BULIDING MODEL FOR BINARY CLASSIFICATION___________#################

#Creating to sets for binary and multi classification

df_binary = df2.copy()

top_up = pd.DataFrame(df_binary['Top-up Month']).rename(columns={'Top-up Month': 'Predict_Label'})

df_binary = pd.concat([df_binary, top_up], axis=1)


df_binary['Top-up Month'] = df_binary['Top-up Month'].replace(to_replace=[' > 48 Months', '12-18 Months', '18-24 Months',
                                                                          '24-30 Months','30-36 Months', '36-48 Months'],
                                                              value='Top-up')



df_multi = df_binary.copy()

df_multi = df_binary[df_binary['Top-up Month'] == 'Top-up']














# creating the features set for model building
binary_features_set = df_binary.drop(['Top-up Month','Predict_Label'], axis=1)

binary_features_set = pd.get_dummies(binary_features_set, drop_first=True)

binary_train_set = binary_features_set.iloc[:len(train_data), :]

binary_test_set = binary_features_set.iloc[len(train_data):, :]

test_null = binary_test_set.isnull().sum(axis=0)

binary_X = binary_train_set.copy()   

binary_Y = df_binary[df_binary['Top-up Month'].notnull()]

binary_Y = binary_Y['Top-up Month']


binary_Y_ann = pd.get_dummies(binary_Y, drop_first=True)





cols = binary_X.columns


x_null = binary_X.isnull().sum(axis=0)

#Scaling data
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

binary_X = mms.fit_transform(binary_X)

# Selecting the best features from the all
from sklearn.feature_selection import SelectKBest, chi2

kbest_binary = SelectKBest(score_func=chi2, k='all')

kbest_binary.fit_transform(binary_X, binary_Y)

pvalues = kbest_binary.pvalues_

f_scores = kbest_binary.scores_

p_cols = pd.DataFrame(pvalues).rename(columns={0: 'p_cols'})

f_cols = pd.DataFrame(f_scores).rename(columns={0: 'f_cols'})

best_features = f_cols['f_cols'].sort_values()

# taking columns with 95% confidence interval
working_cols = p_cols[(p_cols['p_cols'] < 0.05)]

working_cols_index = working_cols.index.values.tolist()

selected_cols = cols[working_cols_index]

#Finally selecting relevant features for ML model
binary_X_m = binary_train_set.copy()

binary_X_k = binary_X_m.loc[:, selected_cols]


#from imblearn.over_sampling import SMOTE 

#sm = SMOTE()

#binary_X_k, binary_Y = sm.fit_resample(binary_X_k, binary_Y)





from sklearn.model_selection import train_test_split

bx_train, bx_test, by_train, by_test = train_test_split(binary_X_k, binary_Y, test_size = 0.3, random_state=5678, stratify=binary_Y)

ax_train, ax_test, ay_train, ay_test = train_test_split(binary_X_k, binary_Y_ann, test_size = 0.3, random_state=5678, stratify=binary_Y_ann)


mms2 = MinMaxScaler()

bx_train = mms2.fit_transform(bx_train)

bx_test = mms2.transform(bx_test)

ax_train = mms2.fit_transform(ax_train)

ax_test = mms2.transform(ax_test)

#binary_X_k = mms2.fit_transform(binary_X_k)





# Best model 
from xgboost import XGBClassifier


rfc = XGBClassifier(silent=False, 
                      learning_rate=0.05,  
                      colsample_bytree = 1,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=500, 
                      reg_alpha = 0.3,
                      max_depth=10, 
                      gamma=1)

#eval_set = [(bx_train, by_train), (bx_test, by_test)]
#eval_metric = ["auc","error"]
#rfc.fit(bx_train, by_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

#results = rfc.evals_result()
#epochs = len(results['validation_0']['error'])
#x_axis = range(0, epochs)

#fig, ax = plt.subplots()
#ax.plot(x_axis, results['validation_0']['auc'], label='Train')
#ax.plot(x_axis, results['validation_1']['auc'], label='Test')
#ax.legend()
#plt.ylabel('AUC')
#plt.title('XGBoost AUC')
#plt.show()

rfc.fit(bx_train, by_train)

#rfc.fit(binary_X_k, binary_Y)

by_predict_rfc = rfc.predict(bx_test)



by_predict_proba = rfc.predict_proba(bx_test)

from sklearn.metrics import classification_report, confusion_matrix

bcr_rfc = classification_report(by_test, by_predict_rfc)

bcm_rfc = confusion_matrix(by_test, by_predict_rfc)

bscore_rfc = rfc.score(bx_test, by_test)

by_predict_new = []


by_predict_proba[0][1]


for i in range(0, len(by_predict_proba)):
    if by_predict_proba[i][1] < 0.45:
        by_predict_new.append("No Top-up Service")
    else:
        by_predict_new.append('Top-up')
        
        
bcr_new = classification_report(by_test, by_predict_new)

bcm_new = confusion_matrix(by_test, by_predict_new)






# ANN MODEL
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=1000, activation='relu'))

ann.add(tf.keras.layers.Dropout(0.2))

ann.add(tf.keras.layers.Dense(units=1000,activation='relu'))

ann.add(tf.keras.layers.Dropout(0.2))

ann.add(tf.keras.layers.Dense(units=1000, activation='relu'))

ann.add(tf.keras.layers.Dropout(0.2))

ann.add(tf.keras.layers.Dense(units=1000,activation='relu'))

ann.add(tf.keras.layers.Dropout(0.2))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

ann.fit(ax_train, ay_train, validation_data=(ax_test, ay_test), epochs=10000, callbacks=[early_stopping], batch_size=1024)

y_predict_ann = ann.predict(ax_test)


# Extracting from the probabilities
#c = pd.DataFrame(np.argmax(y_predict_ann, axis=1)).rename(columns={0: 'Top-up Month'})

#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[0], value=' > 48 Months')
#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[1], value='12-18 Months')
#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[2], value='18-24 Months')
#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[3], value='30-36 Months')
#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[4], value='24-30 Months')
#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[5], value='36-48 Months')
#c['Top-up Month'] = c['Top-up Month'].replace(to_replace=[6], value='No Top-up Service')



#This is for classification report
#b = np.zeros_like(y_predict_ann)

#b[np.arange(len(y_predict_ann)), y_predict_ann.argmax(1)] = 1

# Metrics
#cr_ann = classification_report(ay_test, b)

#cm_ann = confusion_matrix(ay_test, c)



ay_predict_ann = []


y_predict_ann[0][0]


for i in range(0, len(y_predict_ann)):
    if y_predict_ann[i][0] < 0.45:
        ay_predict_ann.append("No Top-up Service")
    else:
        ay_predict_ann.append('Top-up')
        
        
        
acr_new = classification_report(by_test, ay_predict_ann)

acm_new = confusion_matrix(by_test, ay_predict_ann)








































##################_________________PART 6 MODEL BUILDING FOR MULTI CLASSIFICATION_________##############

# Multi Prediction
multi_features_set = df_multi.drop(['Top-up Month','Predict_Label'], axis=1)

multi_features_set = pd.get_dummies(multi_features_set, drop_first=True)

multi_X = multi_features_set.copy()


multi_Y = df_multi[df_multi['Predict_Label'].notnull()]

multi_Y = multi_Y['Predict_Label']



mcols = multi_X.columns


x_null = multi_X.isnull().sum(axis=0)

#Scaling data
from sklearn.preprocessing import MinMaxScaler

mms3 = MinMaxScaler()

multi_X = mms3.fit_transform(multi_X)

# Selecting the best features from the all
from sklearn.feature_selection import SelectKBest, chi2

mkbest = SelectKBest(score_func=chi2, k='all')

mkbest.fit_transform(multi_X, multi_Y)

mpvalues = mkbest.pvalues_

mf_scores = mkbest.scores_

mp_cols = pd.DataFrame(mpvalues).rename(columns={0: 'p_cols'})

mf_cols = pd.DataFrame(mf_scores).rename(columns={0: 'f_cols'})

mbest_features = mf_cols['f_cols'].sort_values()

# taking columns with 95% confidence interval
mworking_cols = mp_cols[(mp_cols['p_cols'] < 0.05)]

mworking_cols_index = mworking_cols.index.values.tolist()

mselected_cols = mcols[mworking_cols_index]

#Finally selecting relevant features for ML model
multi_X_m = multi_features_set.copy()

multi_X_k = multi_X_m.loc[:, mselected_cols]



from imblearn.over_sampling import SMOTE 

sm = SMOTE()

multi_X_k, multi_Y = sm.fit_resample(multi_X_k, multi_Y)





from sklearn.model_selection import train_test_split

mx_train, mx_test, my_train, my_test = train_test_split(multi_X_k, multi_Y, test_size = 0.3, random_state=5678, stratify=multi_Y)

mms4 = MinMaxScaler()

mx_train = mms4.fit_transform(mx_train)

mx_test = mms4.transform(mx_test)

#multi_X_k = mms4.fit_transform(multi_X_k)


# Best model 
from xgboost import XGBClassifier

mrfc = XGBClassifier(silent=False, 
                      learning_rate=0.05,  
                      colsample_bytree = 0.9,
                      subsample = 0.8,
                      objective='multi:softprob', 
                      n_estimators=100, 
                      reg_alpha = 0.3,
                      max_depth=13, 
                      gamma=0)







mrfc.fit(mx_train, my_train)

#mrfc.fit(multi_X_k, multi_Y)

my_predict_rfc = mrfc.predict(mx_test)



my_predict_proba = mrfc.predict_proba(mx_test)

from sklearn.metrics import classification_report, confusion_matrix

mcr_rfc = classification_report(my_test, my_predict_rfc)

mcm_rfc = confusion_matrix(my_test, my_predict_rfc)

mscore_rfc = mrfc.score(mx_test, my_test)


























########################____________PART 7 TEST SET PREDICTION FOR BINARY MODEL_____##################

binary_test_set = binary_features_set.iloc[len(train_data):, :]

sample_set = binary_test_set

binary_test_set = binary_test_set.loc[:, selected_cols]

binary_test_set = mms2.transform(binary_test_set)

test_predict = rfc.predict(binary_test_set)

test_predict_proba = rfc.predict_proba(binary_test_set)

test_predict_updated = []



for i in range(0, len(test_predict_proba)):
    if test_predict_proba[i][1] < 0.40:
        test_predict_updated.append("No Top-up Service")
    else:
        test_predict_updated.append('Top-up')


test_predict = pd.DataFrame(test_predict).rename(columns={0: 'Top-up Month'})

#test_predict_updated = pd.DataFrame(test_predict_updated).rename(columns={0: 'Top-up Month'})

sample_sub = pd.read_csv('sample_submission.csv')

sample_sub.reset_index(drop=True, inplace=True)

sample_set.reset_index(drop=True, inplace=True)

top_up_predict = pd.concat([sample_sub, sample_set, test_predict], axis=1)

#top_up_predict = pd.concat([sample_sub, sample_set, test_predict_updated], axis=1)

binary_top_up_predict = top_up_predict[top_up_predict['Top-up Month'] == 'No Top-up Service']





# multi prediction
multi_predict_set = top_up_predict[top_up_predict['Top-up Month'] == 'Top-up']

multi_predict_set = multi_predict_set.drop('Top-up Month', axis=1)

multi_sample_set = multi_predict_set

multi_predict_set = multi_predict_set.loc[:, mselected_cols]

multi_predict_set = mms4.transform(multi_predict_set)

multi_test_predict = mrfc.predict(multi_predict_set)

multi_test_predict = pd.DataFrame(multi_test_predict).rename(columns={0: 'Top-up Month'})

multi_sample_set.reset_index(drop=True, inplace=True)

multi_test_predict.reset_index(drop=True, inplace=True)

multi_top_up_predict = pd.concat([multi_sample_set, multi_test_predict], axis=1)





#final results
final_results = pd.concat([binary_top_up_predict, multi_top_up_predict], axis=0)

sort_results = final_results.sort_values('ID')

export_results = sort_results[['ID', 'Top-up Month']]

csv = export_results.to_csv('top_up_predict_34.csv')



export_results['Top-up Month'].value_counts()







###############______________PART 8 TEST SET PREDICTION FOR NORMAL CLASSIFICATION________#############################################

test_set = test_set.loc[:, nselected_cols]

test_set = mms_n2.transform(test_set)

test_predict = rfc_n.predict(test_set)

test_predict = pd.DataFrame(test_predict).rename(columns={0: 'Top-up Month'})

sample_sub = pd.read_csv('sample_submission.csv')

top_up_predict = pd.concat([sample_sub, test_predict], axis=1)

sns.countplot(top_up_predict['Top-up Month'])


top_up_predict_01 = top_up_predict.to_csv('top_up_predict_34.csv')



top_up_predict['Top-up Month'].value_counts()



























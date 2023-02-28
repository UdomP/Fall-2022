import numpy as np
import pandas as pd

student = pd.read_csv('student.csv')
gpa = pd.read_csv('gpa.csv')
balance = pd.read_csv('balance.csv')
hometown = pd.read_csv('hometown.csv')

student_gpa = pd.merge(student, gpa, on='CSU ID')
# print(student_gpa)
student_gpa_balance = pd.merge(student_gpa, balance, on='CSU ID')
# print(student_gpa_balance)
student_gpa_balance_hometown = pd.merge(student_gpa_balance, hometown, on=['CSU ID', 'Status'])
print(student_gpa_balance_hometown)
student_gpa_balance_hometown.to_csv('merged.csv', index=False)

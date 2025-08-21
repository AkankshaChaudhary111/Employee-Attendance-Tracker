import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================
# 1. Data Loading
# ==========================
df = pd.read_csv("C:/Users/akank/Downloads/employee_attendance.csv")

# Attendance columns start after first 6 columns
attendance_cols = df.columns[6:]

# ==========================
# 2. Data Cleaning & Transformation
# ==========================
# Standardize status values (strip spaces, lowercase)
df[attendance_cols] = df[attendance_cols].applymap(lambda x: str(x).strip().capitalize())

# Count Present, Absent, Late
df["Present_Count"] = (df[attendance_cols] == "Present").sum(axis=1)
df["Absent_Count"] = (df[attendance_cols] == "Absent").sum(axis=1)
df["Late_Count"] = (df[attendance_cols] == "Late").sum(axis=1)

total_days = len(attendance_cols)
df["Attendance_Percentage"] = round((df["Present_Count"] / total_days) * 100, 2)

# ==========================
# 3. Visualization (Matplotlib Only)
# ==========================
plt.figure(figsize=(10,6))
plt.hist(df["Attendance_Percentage"], bins=20, edgecolor="black", alpha=0.7)
plt.title("Distribution of Attendance Percentage")
plt.xlabel("Attendance %")
plt.ylabel("Number of Employees")
plt.show()

# Bar chart: Top 10 employees with lowest attendance
low_att = df.sort_values("Attendance_Percentage").head(10)
plt.figure(figsize=(10,6))
plt.barh(low_att["first_name"], low_att["Attendance_Percentage"], color="red")
plt.title("Employees with Lowest Attendance")
plt.xlabel("Attendance %")
plt.ylabel("Employee")
plt.show()

# Pie chart: Overall attendance status counts
status_counts = pd.Series({
    "Present": df["Present_Count"].sum(),
    "Absent": df["Absent_Count"].sum(),
    "Late": df["Late_Count"].sum()
})
plt.figure(figsize=(6,6))
plt.pie(status_counts, labels=status_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Overall Attendance Status Distribution")
plt.show()

# Line chart: Average attendance percentage trend over days
avg_trend = (df[attendance_cols] == "Present").mean()
plt.figure(figsize=(12,6))
plt.plot(avg_trend.index, avg_trend.values*100, marker="o", linestyle="-", color="blue")
plt.xticks(rotation=45)
plt.title("Average Daily Attendance Percentage")
plt.ylabel("% Present")
plt.xlabel("Date")
plt.show()

# ==========================
# 4. ML Model (Classification)
# ==========================
# Label: Good Attendance (>=85%) vs Poor Attendance (<85%)
df["Attendance_Label"] = np.where(df["Attendance_Percentage"] >= 85, 1, 0)

X = df[["Present_Count", "Absent_Count", "Late_Count"]]
y = df["Attendance_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ==========================
# 5. Export Final Report
# ==========================
summary = df[["employee_id", "first_name", "last_name", "Present_Count", "Absent_Count", "Late_Count", "Attendance_Percentage", "Attendance_Label"]]

# Export to Excel
try:
    summary.to_excel("Employee_Attendance_Summary.xlsx", index=False)
    print("✅ Summary saved as Excel file: Employee_Attendance_Summary.xlsx")
except ModuleNotFoundError:
    summary.to_csv("Employee_Attendance_Summary.csv", index=False)
    print("⚠️ openpyxl not found. Saved as CSV instead: Employee_Attendance_Summary.csv")


print("Final report saved as Employee_Attendance_Summary.xlsx")

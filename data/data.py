import json
import sqlite3

# File paths
json_file = "data.json"
db_file = "employee.db"

# Load JSON data
with open(json_file, "r") as f:
    employees = json.load(f)

# Connect to SQLite
conn = sqlite3.connect(db_file)
cur = conn.cursor()

# Create table (drop if exists to avoid duplicates when rerunning)
cur.execute("DROP TABLE IF EXISTS data")

# Create table with appropriate schema
cur.execute("""
CREATE TABLE data (
    EmployeeID TEXT PRIMARY KEY,
    MaritalStatus TEXT,
    Gender TEXT,
    EmploymentStatus TEXT,
    JobRole TEXT,
    CareerLevel TEXT,
    PerformanceRating TEXT,
    City TEXT,
    HiringPlatform TEXT,
    PhoneNumber TEXT,
    Email TEXT,
    EducationLevel TEXT,
    ReasonForResignation TEXT,
    DateOfBirth TEXT,
    DateOfJoining TEXT,
    LastAppraisalDate TEXT,
    ResignationDate TEXT,
    YearsOfExperience TEXT
)
""")

# Insert data
for emp in employees:
    cur.execute("""
    INSERT INTO data (
        EmployeeID, MaritalStatus, Gender, EmploymentStatus, JobRole,
        CareerLevel, PerformanceRating, City, HiringPlatform, PhoneNumber,
        Email, EducationLevel, ReasonForResignation, DateOfBirth, DateOfJoining,
        LastAppraisalDate, ResignationDate, YearsOfExperience
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        emp.get("EmployeeID"),
        emp.get("MaritalStatus"),
        emp.get("Gender"),
        emp.get("EmploymentStatus"),
        emp.get("JobRole"),
        emp.get("CareerLevel"),
        emp.get("PerformanceRating"),
        emp.get("City"),
        emp.get("HiringPlatform"),
        emp.get("PhoneNumber"),
        emp.get("Email"),
        emp.get("EducationLevel"),
        emp.get("ReasonForResignation"),
        emp.get("DateOfBirth"),
        emp.get("DateOfJoining"),
        emp.get("LastAppraisalDate"),
        emp.get("ResignationDate"),
        emp.get("YearsOfExperience")
    ))

# Commit and close
conn.commit()
conn.close()

print("Data inserted successfully into employee.db -> data table")

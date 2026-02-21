import pandas as pd
import numpy as np


file_path = "/Users/ahmedsaidgulsen/Desktop/lab/predicting_student/Predicting-Student-Grades-in-Courses-main/z_our_project/chinese_data/yiwen_data_translated.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")


# Apply semesters
for k in range(50):
    if(k < 8):
        col = df.columns[k]
        df.rename(columns={col: col + "/1"}, inplace=True)
    elif(k < 17):
        col = df.columns[k]
        df.rename(columns={col: col + "/2"}, inplace=True)
    elif(k < 25):
        col = df.columns[k]
        df.rename(columns={col: col + "/3"}, inplace=True)
    elif(k < 32):
        col = df.columns[k]
        df.rename(columns={col: col + "/4"}, inplace=True)
    elif(k < 38):
        col = df.columns[k]
        df.rename(columns={col: col + "/5"}, inplace=True)
    elif(k < 46):
        col = df.columns[k]
        df.rename(columns={col: col + "/6"}, inplace=True)
    elif(k < 50):
        col = df.columns[k]
        df.rename(columns={col: col + "/7"}, inplace=True)

# Apply student numbers
student_numbers = df.index

gpa = 0
gpa_per_semester = 0
completed_credits = 0
credit_per_semester = 0

# reshape the data and calculate gpa and total credit:
processed_data = []
semester_data = []
for student_id in student_numbers:
    for col in df.columns:
        course_info = col.rsplit("/", 3)  # rsplit "Course Title/Type/Credits/Semester"
        course_title = course_info[0]  # Extract course title
        course_credit = course_info[2]  # Extract course credit
        course_semester = course_info[3]  # Extract course semester
        grade = df.loc[student_id, col]  # Get student's grade

        gpa_per_semester += float(course_credit) * int(grade)  # Calculate semester gpa (later divide by credit)
        credit_per_semester += float(course_credit)  # Calculate semester credit

        if col == df.columns[7]:  # First semester
            gpa += gpa_per_semester  # Total gpa till this point (later divide by completed_credits)
            completed_credits += credit_per_semester  # Total credits till this point
            gpa_per_semester = gpa_per_semester / credit_per_semester  # Semester gpa

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[16]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[24]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[31]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[37]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[45]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[49]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0



        # Append data in required format
        processed_data.append([student_id, course_title, course_credit, grade, course_semester])
    gpa = 0
    completed_credits = 0

# total_credit = 139


# new data frame:
df_arranged = pd.DataFrame(processed_data, columns=["Student Number", "Course Title", "Course Credit",
                                                    "Grades", "Course Semester"])

df_to_be_added = pd.DataFrame(semester_data, columns=["Student Number", "Course Semester", "GPA", "Completed Credits",
                                                      "Semester GPA", "Semester Credit"])

print(df_to_be_added.iloc[0:14, 0:6])

df_final = pd.merge(df_arranged, df_to_be_added, on=["Student Number", "Course Semester"])

df_final["Department Code"] = "BLG"

# print(df_final["GPA"])


# df_final.to_csv("processed_data.csv", index=False) // output'u experiment klasorune aktardik
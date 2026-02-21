import pandas as pd
import numpy as np
import statistics


file_path = "./yiwen_data_translated.xlsx"
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

# Calculate grade rates and mean/std grade
grade_data = []
grades = []  # Store grades for each course
grade_1_counter = 0;  # Increment by 1 if grade is 1
grade_2_counter = 0;  # Increment by 1 if grade is 2
grade_3_counter = 0;  # Increment by 1 if grade is 3
grade_4_counter = 0;  # Increment by 1 if grade is 4
grade_5_counter = 0;  # Increment by 1 if grade is 5
for column in df:
    for i in range(382):
        grades.append(int(df[column].values[i]))
        match df[column].values[i]:
            case 1:
                grade_1_counter = grade_1_counter + 1
            case 2:
                grade_2_counter = grade_2_counter + 1
            case 3:
                grade_3_counter = grade_3_counter + 1
            case 4:
                grade_4_counter = grade_4_counter + 1
            case 5:
                grade_5_counter = grade_5_counter + 1

    mean_grade = statistics.mean(grades)
    std_dev_grade = statistics.stdev(grades)

    grade_1_rate = grade_1_counter / 382  # Amount of students
    grade_2_rate = grade_2_counter / 382
    grade_3_rate = grade_3_counter / 382
    grade_4_rate = grade_4_counter / 382
    grade_5_rate = grade_5_counter / 382
    grade_data.append([column.rsplit("/", 3)[0], grade_1_rate, grade_2_rate, grade_3_rate, grade_4_rate, grade_5_rate,
                       mean_grade, std_dev_grade])
    grade_1_counter = 0  # Reset it for each course
    grade_2_counter = 0
    grade_3_counter = 0
    grade_4_counter = 0
    grade_5_counter = 0
    grades.clear()

# Apply student numbers
student_numbers = df.index

gpa = 0
gpa_per_semester = 0
completed_credits = 0
credit_per_semester = 0

# reshape the data and calculate gpa and total credit:
gpas_semester1, gpas_semester2, gpas_semester3, gpas_semester4, gpas_semester5, gpas_semester6, gpas_semester7 = [], [], [], [], [], [], []
processed_data = []
semester_data = []
for student_id in student_numbers:
    for col in df.columns:
        course_info = col.rsplit("/", 3)  # rsplit "Course Title/Type/Credits/Semester"
        course_title = course_info[0]  # Extract course title
        course_credit = course_info[2]  # Extract course credit
        course_semester = int(course_info[3])  # Extract course semester
        grade = df.loc[student_id, col]  # Get student's grade

        gpa_per_semester += float(course_credit) * int(grade)  # Calculate semester gpa (later divide by credit)
        credit_per_semester += float(course_credit)  # Calculate semester credit

        if col == df.columns[7]:  # First semester
            gpa += gpa_per_semester  # Total gpa till this point (later divide by completed_credits)
            completed_credits += credit_per_semester  # Total credits till this point
            gpa_per_semester = gpa_per_semester / credit_per_semester  # Semester gpa
            gpas_semester1.append(gpa / completed_credits)  # List of GPAs of students in semester 1

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[16]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester
            gpas_semester2.append(gpa / completed_credits)  # List of GPAs of students in semester 2

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[24]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester
            gpas_semester3.append(gpa / completed_credits)  # List of GPAs of students in semester 3

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[31]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester
            gpas_semester4.append(gpa / completed_credits)  # List of GPAs of students in semester 4

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[37]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester
            gpas_semester5.append(gpa / completed_credits)  # List of GPAs of students in semester 5

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[45]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester
            gpas_semester6.append(gpa / completed_credits)  # List of GPAs of students in semester 6

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0
        elif col == df.columns[49]:
            gpa += gpa_per_semester
            completed_credits += credit_per_semester
            gpa_per_semester = gpa_per_semester / credit_per_semester
            gpas_semester7.append(gpa / completed_credits)  # List of GPAs of students in semester 7

            semester_data.append([student_id, course_semester, gpa / completed_credits, completed_credits, gpa_per_semester, credit_per_semester])
            gpa_per_semester = 0
            credit_per_semester = 0

        # Append data in required format
        processed_data.append([student_id, course_title, course_credit, grade, course_semester])
    gpa = 0
    completed_credits = 0

gpa_data = [[1, statistics.mean(gpas_semester1), statistics.stdev(gpas_semester1)],  # Semester Num, Mean GPA, Stdev GPA
            [2, statistics.mean(gpas_semester2), statistics.stdev(gpas_semester2)],
            [3, statistics.mean(gpas_semester3), statistics.stdev(gpas_semester3)],
            [4, statistics.mean(gpas_semester4), statistics.stdev(gpas_semester4)],
            [5, statistics.mean(gpas_semester5), statistics.stdev(gpas_semester5)],
            [6, statistics.mean(gpas_semester6), statistics.stdev(gpas_semester6)],
            [7, statistics.mean(gpas_semester7), statistics.stdev(gpas_semester7)]]

# new data frame:
df_arranged = pd.DataFrame(processed_data, columns=["Student Number", "Course Title", "Course Credit",
                                                    "Grades", "Course Semester"])

df_to_be_added = pd.DataFrame(semester_data, columns=["Student Number", "Course Semester", "GPA", "Completed Credits",
                                                      "Semester GPA", "Semester Credit"])

df_grade_rates = pd.DataFrame(grade_data, columns=["Course Title", "Grade 1 Rate", "Grade 2 Rate", "Grade 3 Rate",
                                                   "Grade 4 Rate", "Grade 5 Rate", "Mean Grade", "STDEV Grade"])

df_gpa_data = pd.DataFrame(gpa_data, columns=["Course Semester", "Mean GPA", "STDEV GPA"])


df_temp = pd.merge(df_arranged, df_to_be_added, on=["Student Number", "Course Semester"])

df_temp2 = pd.merge(df_temp, df_grade_rates, on=["Course Title"])

df_final = pd.merge(df_temp2, df_gpa_data, on=["Course Semester"])

df_final["Department Code"] = "BLG"


df_final.to_csv("processed_data.csv", index=False)
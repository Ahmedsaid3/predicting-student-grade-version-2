
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from math import sqrt
import recommendations

def run_sehir_analysis():
    print("Loading SEHIR data...")
    try:
        df = pd.read_csv('../../datasets/SEHIR/processed_dataset.csv')
        df = df[['Student Number', 'Course Code', 'Letter Grade', 'Semester', 'Course Credit', 'GPA', 'Completed Credits', 'Department Code']]
        df = pd.concat([df, pd.get_dummies(df['Department Code'], prefix='Department Code')], axis=1)
        df.drop(['Department Code'], axis=1, inplace=True)
    except FileNotFoundError:
        print("Error: Dataset not found. Please check path.")
        return

    numerical_grades = {'A+': 4.1, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7, 'C+': 2.3, 'C': 2.0,
                        'C-': 1.7, 'D+': 1.3, 'D': 1.0, 'D-': 0.5, 'F': 0.0}

    course_credits = {}
    for row_idx in df.index:
        course_code = df.iloc[row_idx, 1]
        credit = df.iloc[row_idx, 4]    
        course_credits[course_code] = credit

    def get_semester_data(semester_name):
        semester_data = {}
        dataset = df[df.iloc[:, 3] == semester_name]
        dataset.index = range(len(dataset))
        for row_idx in dataset.index:
            student_number = dataset.iloc[row_idx, 0]
            course_code = dataset.iloc[row_idx, 1]
            letter_grade = dataset.iloc[row_idx, 2]
            semester_data.setdefault(student_number, {})
            semester_data[student_number][course_code] = numerical_grades[letter_grade]
        return semester_data

    def get_avg_gpa(train_semester, student):
        courses = train_semester[student]
        total_credit = 0
        weights = 0
        for course in courses:
            if course in course_credits:
                total_credit += course_credits[course]
                weights += courses[course] * course_credits[course]
            else:
                total_credit += 1 # minimal fallback
                weights += courses[course]
        if total_credit == 0: return 0
        return weights / total_credit

    def get_grade_stats(semester_data, student):
        grade_list = []
        for course in semester_data[student]:
            numerical_grade = semester_data[student][course]
            grade_list.append(numerical_grade)
        if not grade_list: return 0, 0
        mean = np.mean(grade_list)
        std_dev = np.std(grade_list)
        return mean, std_dev

    def fit_cluster(train_sems, num_clusters, training_data, cluster_model):
        train_dataset = pd.DataFrame(columns=df.columns)
        for sem in train_sems:
            subset = df[df.iloc[:, 3] == sem]
            train_dataset = pd.concat([train_dataset, subset], ignore_index=True)

        cluster_features = train_dataset[['GPA', 'Completed Credits'] + list(train_dataset.columns[7:])]
        fitted_cluster_model = cluster_model(n_clusters=int(num_clusters), random_state=42, n_init='auto').fit(cluster_features)
        cluster_labels = fitted_cluster_model.labels_
        
        cluster_dataset = {}
        for i in range(len(cluster_labels)):
            cluster_dataset.setdefault(cluster_labels[i], {})
            student_number = train_dataset.iloc[i, 0]
            if student_number in training_data:
                cluster_dataset[cluster_labels[i]][student_number] = training_data[student_number]
        return cluster_dataset, fitted_cluster_model

    def cluster_test_data(cluster_model, semester_name):
        test_dataset = df[df.iloc[:, 3] == semester_name]
        test_dataset.index = range(len(test_dataset))
        cluster_features = test_dataset[['GPA', 'Completed Credits'] + list(test_dataset.columns[7:])]
        if len(cluster_features) == 0: return {}
        cluster_labels = cluster_model.predict(cluster_features)
        semester_data = get_semester_data(semester_name)
        cluster_dataset = {}
        for i in range(len(cluster_labels)):
            cluster_dataset.setdefault(cluster_labels[i], {})
            student_number = test_dataset.iloc[i, 0]
            if student_number in semester_data:
                cluster_dataset[cluster_labels[i]][student_number] = semester_data[student_number]
        return cluster_dataset    

    def get_errors_sehir(train_semester, test_semester, sim):
        average_gpa = {}
        gpa = {}
        
        # Coverage Analysis Lists
        cf_y_true = []
        cf_y_pred = []
        fallback_y_true = []
        fallback_y_pred = []

        for student in train_semester:
            gpa[student] = get_avg_gpa(train_semester, student)
            average_gpa[student] = gpa[student]
        
        # Using STANDARD CF logic as per notebook (History based)
        # Note: Sehir has overlap, so this works better than Anhui
        
        for student in train_semester:
            if student not in test_semester: continue
            
            recommended_courses = {}
            # Standard recommendation logic
            try:
                recs = recommendations.getRecommendations(train_semester, student, sim, dgpa=True, gpa=gpa, delta=0.7)
            except Exception as e:
                # print(f"Rec Error: {e}")
                recs = []

            for rec_grade, rec_course in recs:
                recommended_courses.setdefault(rec_course, rec_grade)
                
            mean, std_dev = get_grade_stats(train_semester, student)
            
            for course_code in test_semester[student]:
                is_cf = False
                rec_grade = 0
                
                if course_code in recommended_courses:
                    rec_grade = recommended_courses[course_code]
                    is_cf = True
                else:
                    if student in average_gpa:
                        rec_grade = average_gpa[student]
                        is_cf = False
                    else: continue

                if rec_grade < mean - (2 * std_dev) or rec_grade > mean + (2 * std_dev):
                    continue
                
                if is_cf:
                    cf_y_pred.append(rec_grade)
                    cf_y_true.append(test_semester[student][course_code])
                else:
                    fallback_y_pred.append(rec_grade)
                    fallback_y_true.append(test_semester[student][course_code])
                    
        return cf_y_true, cf_y_pred, fallback_y_true, fallback_y_pred

    # Run Prediction Loop
    sorted_semesters = sorted(set(df.iloc[:, 3]))
    print(f"Semesters: {sorted_semesters}")
    
    # Running for k=15 as a representative/best example
    num_clusters = 15 
    print(f"Processing Clusters: {num_clusters}")
    
    total_cf_true = []
    total_cf_pred = []
    total_fallback_true = []
    total_fallback_pred = []
    
    train_semester = {}
    
    for sem_idx in range(1, len(sorted_semesters)):
        print(f"  Processing Term {sem_idx+1}/{len(sorted_semesters)}")
        new_semester = get_semester_data(sorted_semesters[sem_idx-1])
        for student in new_semester:
            if student in train_semester:
                train_semester[student].update(new_semester[student])
            else:
                train_semester[student] = new_semester[student]
        
        training_semesters_name = sorted_semesters[:sem_idx]
        train_cluster_data, fitted_cluster_model = fit_cluster(training_semesters_name, num_clusters, train_semester, KMeans)
        test_cluster_data = cluster_test_data(fitted_cluster_model, sorted_semesters[sem_idx])
        
        for cluster_label in train_cluster_data:
            if cluster_label not in test_cluster_data: continue
            
            c_true, c_pred, f_true, f_pred = get_errors_sehir(
                train_cluster_data[cluster_label], 
                test_cluster_data[cluster_label], 
                recommendations.sim_pearson
            )
            total_cf_true.extend(c_true)
            total_cf_pred.extend(c_pred)
            total_fallback_true.extend(f_true)
            total_fallback_pred.extend(f_pred)

    # Final Stats
    total_preds = len(total_cf_true) + len(total_fallback_true)
    print("\n--- SEHIR DATASET RESULTS ---")
    if total_preds > 0:
        coverage = (len(total_cf_true) / total_preds) * 100
        print(f"Coverage Ratio: {coverage:.2f}% ({len(total_cf_true)}/{total_preds})")
        
        from sklearn.metrics import mean_squared_error
        if len(total_cf_true) > 0:
            cf_rmse = sqrt(mean_squared_error(total_cf_true, total_cf_pred))
            print(f"CF RMSE: {cf_rmse:.4f}")
        else:
            cf_rmse = 0
            print("CF RMSE: N/A")
            
        if len(total_fallback_true) > 0:
            fb_rmse = sqrt(mean_squared_error(total_fallback_true, total_fallback_pred))
            print(f"Fallback RMSE: {fb_rmse:.4f}")
        else:
            fb_rmse = 0
            print("Fallback RMSE: N/A")
            
        if cf_rmse and fb_rmse:
            print(f"RMSE Diff (FB - CF): {fb_rmse - cf_rmse:.4f}")

if __name__ == "__main__":
    run_sehir_analysis()

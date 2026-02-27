import json
import pandas as pd
import numpy as np
import recommendations
from sklearn.cluster import KMeans

# Load and preprocess data
print("Loading data...")
df = pd.read_csv("datasets/ANHUI/processed_data.csv")
df = pd.concat([df, pd.get_dummies(df['Department Code'], prefix='Department Code')], axis=1)
# df.drop(['Department Code'], axis=1, inplace=True) # Check if this drop is needed, notebook did it.
# Notebook line 276: df.drop(['Department Code'], axis=1, inplace=True)
# But get_dummies keeps original column usually? No, pandas concat.
# Let's verify if Department Code is used later. Notebook dropped it.
if 'Department Code' in df.columns:
    df.drop(['Department Code'], axis=1, inplace=True)

# Build course_credits dictionary
course_credits = {}
for row_idx in df.index:
    course_title = df.iloc[row_idx, 1]
    credit = df.iloc[row_idx, 2]    
    course_credits[course_title] = credit

def get_semester_data(semester_num):
    semester_data = {}   # semester data in shape {student_number: {course_title: grade, ...}, ...}
    
    # extracting the instances with the given semester_num from the main dataFrame
    dataset = df[df.iloc[:, 4] == semester_num]
    dataset.index = range(len(dataset)) 

    # filling the semester_data dictionary
    for row_idx in dataset.index:
        student_number = dataset.iloc[row_idx, 0]
        course_title = dataset.iloc[row_idx, 1]
        grade = dataset.iloc[row_idx, 3]
        
        semester_data.setdefault(student_number, {})
        semester_data[student_number][course_title] = grade
    
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
             # Fallback if course credit missing (shouldn't happen with correct logic)
             total_credit += 1 
             weights += courses[course]
    
    if total_credit == 0: return 0
    return weights / total_credit

def get_grade_stats(semester_data, student):
    grade_list = []
    
    for course in semester_data[student]:
        grade = semester_data[student][course]
        grade_list.append(grade)
    
    mean = np.mean(grade_list)
    std_dev = np.std(grade_list)
    
    return mean, std_dev

def fit_cluster(train_sems, num_clusters, training_data, cluster_model):

    train_dataset = pd.DataFrame(columns=df.columns)
    
    # extracting instances from the dataset which should be in training data
    for sem in train_sems:
        # Use simple concat loop
        subset = df[df.iloc[:, 4] == sem]
        train_dataset = pd.concat([train_dataset, subset], ignore_index=True)
 
    cluster_features = train_dataset[['GPA', 'Completed Credits'] + list(train_dataset.columns[9:])]
    
    # fitting a clustering model based on GPA, Completed Credits and Departments
    # Ensure n_clusters is int
    fitted_cluster_model = cluster_model(n_clusters=int(num_clusters)).fit(cluster_features)
    cluster_labels = fitted_cluster_model.labels_  
    
    cluster_dataset = {}   # splitting the train dataset into sub-dicts based on their predicted cluster label
    
    # assigning each students' data to their predicted clusters
    for i in range(len(cluster_labels)):
        cluster_dataset.setdefault(cluster_labels[i], {})
        student_number = train_dataset.iloc[i, 0]
        if student_number in training_data:
            cluster_dataset[cluster_labels[i]][student_number] = training_data[student_number]
    
    return cluster_dataset, fitted_cluster_model

def cluster_test_data(fitted_cluster_model, semester_num):
    # extracting all instances with the given semester_num from the main dataFrame
    test_dataset = df[df.iloc[:, 4] == semester_num]
    test_dataset.index = range(len(test_dataset))
    
    # predicting the cluster labels of test data using a cluster model fitted on the train data so far
    cluster_features = test_dataset[['GPA', 'Completed Credits'] + list(test_dataset.columns[9:])]
    if len(cluster_features) == 0:
        return {}
        
    cluster_labels = fitted_cluster_model.predict(cluster_features) # predict the test data 
    
    # getting the semester data of available students in test semester
    semester_data = get_semester_data(semester_num)
    
    cluster_dataset = {}   # splitting the test dataset into sub-dicts based on their predicted cluster label
    
    # assigning each students' data to their predicted clusters
    for i in range(len(cluster_labels)):
        cluster_dataset.setdefault(cluster_labels[i], {})
        student_number = test_dataset.iloc[i, 0]
        if student_number in semester_data:
            cluster_dataset[cluster_labels[i]][student_number] = semester_data[student_number]
        
    return cluster_dataset    

# --- NEW HELPER FOR CROSS-SEMESTER RECOMMENDATION ---
def get_recommendations_cross_semester(train_history, current_semester_data, person, similarity):
    totals = {}
    simSums = {}
    
    # We use train_history to calculate similarity (Context: History)
    # We use current_semester_data to get the grades to recommend (Context: Current/Peers)
    
    for other in train_history:
        if other == person: continue
        
        # Calculate similarity based on HISTORY
        sim = similarity(train_history, person, other)
        
        if sim <= 0: continue
        
        # Check if this 'other' person has data in the CURRENT semester (is a peer in current context)
        if other not in current_semester_data: continue
        
        for item in current_semester_data[other]:
            # Recommend items from CURRENT semester
            totals.setdefault(item, 0)
            totals[item] += current_semester_data[other][item] * sim
            simSums.setdefault(item, 0)
            simSums[item] += sim
            
    rankings = [(round(total/simSums[item], 2), item) for item, total in totals.items() if simSums[item] > 0]
    rankings.sort()
    rankings.reverse()
    return rankings

def get_errors(train_semester, test_semester, sim, item_based):
    average_gpa = {}
    y_true = []
    y_pred = []
    
    # Coverage Analysis Lists
    cf_y_true = []
    cf_y_pred = []
    fallback_y_true = []
    fallback_y_pred = []
    
    gpa = {}
    
    # Pre-calculate Average GPA from History for Fallback
    for student in train_semester:
        gpa[student] = get_avg_gpa(train_semester, student)
        average_gpa[student] = gpa[student]
    
    if item_based:
        print("Warning: Item-based CF not fully adapted for cross-semester disjoint data in this script.")
        pass 
    else:
        # User-Based Cross-Semester Recommendation
        relevant_students = [s for s in train_semester if s in test_semester]
        
        for student in relevant_students:
            recommended_courses = {}
            
            # --- CRITICAL FIX: Use Cross-Semester Recommendation ---
            recs = get_recommendations_cross_semester(train_semester, test_semester, student, sim)
            
            for rec_grade, rec_course in recs:
                recommended_courses.setdefault(rec_course, rec_grade)
                
            mean, std_dev = get_grade_stats(train_semester, student)

            # Check predictions against actuals
            for course_title in test_semester[student]:
                is_cf_prediction = False
                rec_grade = 0
                
                if course_title in recommended_courses:
                    rec_grade = recommended_courses[course_title]
                    is_cf_prediction = True
                else:
                    # Fallback to Average GPA
                    if student in average_gpa:
                        rec_grade = average_gpa[student]
                        is_cf_prediction = False
                    else:
                        continue # Should not happen if student in train_semester
                
                # Outlier Check - Applies to both
                if rec_grade < mean - (2 * std_dev) or rec_grade > mean + (2 * std_dev):
                    continue
                
                y_pred.append(rec_grade)
                y_true.append(test_semester[student][course_title])
                
                if is_cf_prediction:
                    cf_y_pred.append(rec_grade)
                    cf_y_true.append(test_semester[student][course_title])
                else:
                    fallback_y_pred.append(rec_grade)
                    fallback_y_true.append(test_semester[student][course_title])
                
    return y_true, y_pred, cf_y_true, cf_y_pred, fallback_y_true, fallback_y_pred

def predict(sim, cluster_model, item_based=False):
    predictions = {} 
    
    # Coverage Analysis Aggregators
    total_cf_true = []
    total_cf_pred = []
    total_fallback_true = []
    total_fallback_pred = []
    
    sorted_semesters = sorted(set(df.iloc[:, 4]))   # sorting semesters
    print(f"Semesters: {sorted_semesters}")
    
    # Range of clusters
    for num_clusters in range(10, 31, 5):
        print(f"Processing Clusters: {num_clusters}")
        predictions.setdefault(str(num_clusters), {})
        train_semester = {}   # {student_number: {course_title: grade, ...}, ...}
        
        # Reset aggregators for each cluster configuration (optional, currently aggregating globally or per K)
        # Let's track per K
        k_cf_true = []
        k_cf_pred = []
        k_fallback_true = []
        k_fallback_pred = []
        
        for sem_idx in range(1, len(sorted_semesters)): 
            predictions[str(num_clusters)].setdefault(str(sem_idx), {'y_true': [], 'y_pred': []})
            
            # Combining previous semesters as history
            new_semester = get_semester_data(sorted_semesters[sem_idx-1])
            for student in new_semester:
                if student in train_semester:
                    train_semester[student].update(new_semester[student])
                else:
                    train_semester[student] = new_semester[student]
            
            training_semesters_name = sorted_semesters[:sem_idx]
            
            # Cluster based on History
            train_cluster_data, fitted_cluster_model = fit_cluster(training_semesters_name, num_clusters, train_semester, cluster_model)
            
            # Cluster Current/Test Data
            test_semester_name = sorted_semesters[sem_idx]
            test_cluster_data = cluster_test_data(fitted_cluster_model, test_semester_name)
            
            for cluster_label in train_cluster_data:
                if cluster_label not in test_cluster_data:
                    continue
                
                # Pass History (train) and Current (test) to get_errors
                y_true, y_pred, cf_true, cf_pred, fb_true, fb_pred = get_errors(train_cluster_data[cluster_label], test_cluster_data[cluster_label], sim, item_based)
                
                predictions[str(num_clusters)][str(sem_idx)]['y_true'] += y_true
                predictions[str(num_clusters)][str(sem_idx)]['y_pred'] += y_pred
                
                k_cf_true.extend(cf_true)
                k_cf_pred.extend(cf_pred)
                k_fallback_true.extend(fb_true)
                k_fallback_pred.extend(fb_pred)
        
        # Calculate Stats for this K
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        
        total_predictions = len(k_cf_true) + len(k_fallback_true)
        if total_predictions > 0:
            coverage_ratio = (len(k_cf_true) / total_predictions) * 100
            print(f"--- K={num_clusters} Stats ---")
            print(f"Coverage Ratio: {coverage_ratio:.2f}% ({len(k_cf_true)}/{total_predictions})")
            
            if len(k_cf_true) > 0:
                cf_rmse = sqrt(mean_squared_error(k_cf_true, k_cf_pred))
                print(f"CF RMSE: {cf_rmse:.4f}")
            
            if len(k_fallback_true) > 0:
                fb_rmse = sqrt(mean_squared_error(k_fallback_true, k_fallback_pred))
                print(f"Fallback RMSE: {fb_rmse:.4f}")
                
            if len(k_cf_true) > 0 and len(k_fallback_true) > 0:
                print(f"RMSE Diff (FB - CF): {fb_rmse - cf_rmse:.4f}")
                
    return predictions

if __name__ == "__main__":
    model_predictions = {}
    
    print("Running Pearson Correlation CF...")
    predictions = predict(recommendations.sim_pearson, KMeans, item_based=False)
    model_predictions['Pearson Correlation'] = predictions
    
    output_file = 'student_base_clustering_and_userbased_collaborative_filtering_FIXED.json'
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as fw:
        json.dump(model_predictions, fw, default=str)
    
    print("Done!")

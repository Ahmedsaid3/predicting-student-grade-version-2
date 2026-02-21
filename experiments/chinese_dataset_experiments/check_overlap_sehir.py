
import pandas as pd

def check_sehir_overlap():
    try:
        # Load Sehir dataset
        df = pd.read_csv('../../datasets/SEHIR/processed_dataset.csv')
        
        # Column names might differ, checking based on notebook view
        # Notebook used: Student Number, Course Code, Semester
        # Let's verify column names first generically or assume standard
        # Based on notebook: 'Semester' is column 3 (index 3)
        
        print("Columns:", df.columns.tolist())
        
        # Assuming 'Semester' and 'Course Code' exist
        if 'Semester' not in df.columns or 'Course Code' not in df.columns:
            print("Required columns not found.")
            return

        sorted_semesters = sorted(set(df['Semester']))
        print("\nSorted Semesters:", sorted_semesters)
        
        for i in range(len(sorted_semesters) - 1):
            sem1 = sorted_semesters[i]
            sem2 = sorted_semesters[i+1]
            
            courses1 = set(df[df['Semester'] == sem1]['Course Code'])
            courses2 = set(df[df['Semester'] == sem2]['Course Code'])
            
            overlap = courses1.intersection(courses2)
            
            print(f"\nComparing {sem1} vs {sem2}:")
            print(f"  Unique courses in {sem1}: {len(courses1)}")
            print(f"  Unique courses in {sem2}: {len(courses2)}")
            print(f"  Overlapping courses: {len(overlap)}")
            if len(overlap) > 0:
                print(f"  Sample overlap: {list(overlap)[:5]}")
            else:
                print("  NO OVERLAP.")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_sehir_overlap()

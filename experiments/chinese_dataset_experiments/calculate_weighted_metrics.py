
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def calculate_metrics():
    json_path = '/Users/ahmedsaidgulsen/Desktop/predicting-student-grade-dev/experiments/chinese_dataset_experiments/student_base_clustering_and_userbased_collaborative_filtering_FIXED.json'
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    pearson_data = data.get('Pearson Correlation', {})
    
    # Weights from final_comparison.py
    # Term-2: 9, Term-3: 8, Term-4: 7, Term-5: 6, Term-6: 8, Term-7: 4
    # Mapping JSON semester keys to weights
    # Assuming JSON "1" -> Term-2, ..., "6" -> Term-7
    weights = {
        "1": 9,
        "2": 8,
        "3": 7,
        "4": 6,
        "5": 8,
        "6": 4
    }
    total_weight = sum(weights.values()) # Should be 42
    
    results = {}

    for cluster_size, semesters in pearson_data.items():
        weighted_rmse_sum = 0
        weighted_mae_sum = 0
        
        print(f"Processing Cluster Size: {cluster_size}")
        
        for sem_key, sem_data in semesters.items():
            if sem_key not in weights:
                print(f"  Warning: Semester {sem_key} not in weights map.")
                continue
                
            y_true = np.array(list(map(float, sem_data['y_true'])))
            y_pred = np.array(list(map(float, sem_data['y_pred'])))
            
            rmse = sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            w = weights[sem_key]
            weighted_rmse_sum += rmse * w
            weighted_mae_sum += mae * w
            
            print(f"  Sem {sem_key}: RMSE={rmse:.4f}, MAE={mae:.4f}, Weight={w}")
            
        avg_weighted_rmse = weighted_rmse_sum / total_weight
        avg_weighted_mae = weighted_mae_sum / total_weight
        
        results[cluster_size] = {
            "WeightedAvgRMSE": avg_weighted_rmse,
            "WeightedAvgMAE": avg_weighted_mae
        }
        
    print("\nFINAL RESULTS:")
    print(json.dumps(results, indent=4))

    # Find best cluster configuration (min RMSE)
    if results:
        best_cluster = min(results, key=lambda x: results[x]['WeightedAvgRMSE'])
        print(f"\nBest Cluster Size: {best_cluster}")
        print(json.dumps(results[best_cluster], indent=4))

if __name__ == "__main__":
    calculate_metrics()

import json
import time


def results_to_json(model_name,dataset_name,results,time_taken,filename):
    time_of_result = time.strftime("%Y-%m-%d %H:%M:%S")

    json_results = {
        'ModelName': model_name,
        'DatasetName': dataset_name,
        'Results': results,
        'Timestamp': time_of_result,
        'RunTime': time_taken,
    }

    with open(filename) as f:
        json.dump(json_results, f)
    print(f"Results for experiment{model_name}_{dataset_name} has been saved at {filename} at {time_of_result}")



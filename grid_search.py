# do hyperparameter grid search
import json
import os

task_log_path = "task_logs"
running_Log_path = "running_logs"


def grid_search():
    # load config from config/hyperparameter.json
    try:
        with open("config/hyperparameter.json", "r") as f:
            config = json.load(f)

    except Exception as e:
        print("Error loading hyperparameter.json")
        print(e)
        return

    # remove run.sh
    if os.path.exists("run.sh"):
        os.remove("run.sh")

    # import env if it's windows platform
    if os.name == "nt":
        with open("run.sh", "a") as f:
            f.write("source /e/miniconda/Scripts/activate\n")
            f.write("conda activate nlp_env\n")

    # create a dir to save logs
    if not os.path.exists(task_log_path):
        os.makedirs(task_log_path)

    if not os.path.exists(running_Log_path):
        os.makedirs(running_Log_path)

    # write config to task log named with timestamp
    import time
    import datetime
    timestamp = time.time()
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    task_log_name = os.path.join(task_log_path, dt_object.strftime("%Y-%m-%d-%H-%M-%S") + ".json")
    with open(task_log_name, "w") as f:
        json.dump(config, f, indent=4)

    # do grid search
    data_path_list = config["data_path"]
    model_list = config["model"]
    embedding_list = config["embeder"]
    prompt_version_list = config["prompt"]
    max_len_list = config["max_len"]
    N_list = config["N"]
    topK_list = config["top_K"]
    k_reverse_list = config["k_reverse"]

    for data_path in data_path_list:
        for model_path in model_list:
            for embedding in embedding_list:
                for prompt_version in prompt_version_list:
                    for max_len in max_len_list:
                        for N in N_list:
                            for topK in topK_list:
                                for k_reverse in k_reverse_list:
                                    # run the model
                                    # create output path named by all hyperparameters
                                    dataset_name = data_path.split("/")[-1].split(".")[0]
                                    output_path = "output/{}_{}_{}_{}_{}_{}".format(dataset_name, prompt_version,
                                                                                    max_len, N, topK, k_reverse)
                                    # create output path
                                    if not os.path.exists(output_path):
                                        os.makedirs(output_path)

                                    # running log named with parameter and timestamp
                                    running_log_name = os.path.join(running_Log_path, dt_object.strftime("%Y-%m-%d-%H"
                                                                                                         "-%M-%S") +
                                                                    "_{}_{}_{}_{}_{}_{}.log".format(dataset_name,
                                                                                                    prompt_version,
                                                                                                    max_len, N, topK,
                                                                                                    k_reverse))
                                    # replace \ with / in running log name
                                    running_log_name = running_log_name.replace("\\", "/")

                                    # append to a shell script to run model and redirect logs to running logs
                                    with open("run.sh", "a") as f:
                                        f.write("python eval_fewshot.py --data_path {} --model {} --embedder {} "
                                                "--prompt {} --max_len {} --N {} --top_k {} --top_k_reverse {} "
                                                "--output_path {} --device_id 0,1 --start_index 0 --end_index 9999 "
                                                "--overwrite False > {}\n".format(data_path, model_path, embedding,
                                                                                  prompt_version, max_len, N, topK,
                                                                                  k_reverse, output_path,
                                                                                  running_log_name))

                                    # save parameters to output dir with json format
                                    with open(os.path.join(output_path, "parameters.json"), "w") as f:
                                        json.dump({"data_path": data_path, "model": model_path, "embedder": embedding,
                                                   "prompt": prompt_version, "max_len": max_len, "N": N, "top_k": topK,
                                                   "top_k_reverse": k_reverse, "output_path": output_path}, f, indent=4)



if __name__ == "__main__":
    grid_search()

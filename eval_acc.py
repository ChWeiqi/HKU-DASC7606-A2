import os
import json


def eval_acc():
    # traverse all folder in output
    for folder in os.listdir("output"):
        if os.path.isdir(os.path.join("output", folder)):
            # do evaluation
            argsdict = {"prediction_path": os.path.join("output", folder)}
            from acc import main as acc_main
            accuracy = acc_main(argsdict)

            # load parameters
            with open(os.path.join("output", folder, "parameters.json"), "r") as f:
                params = json.load(f)

            # save parameters and accuracy in excel, every parameter in a column, append to output.xlsx
            import pandas as pd
            if not os.path.exists("output.xlsx"):
                df = pd.DataFrame(columns=["data_path", "model", "embedder", "prompt", "max_len", "N", "top_k",
                                           "top_k_reverse", "accuracy"])
            else:
                df = pd.read_excel("output.xlsx")

            df = pd.concat([df, pd.DataFrame({"data_path": [params["data_path"]], "model": [params["model"]],
                                              "embedder": [params["embedder"]], "prompt": [params["prompt"]],
                                              "max_len": [params["max_len"]], "N": [params["N"]],
                                              "top_k": [params["top_k"]], "top_k_reverse": [params["top_k_reverse"]],
                                              "accuracy": [accuracy]})], ignore_index=True)
            df.to_excel("output.xlsx", index=False)


if __name__ == "__main__":
    eval_acc()
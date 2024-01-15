import os


def save_to_csv(q, csv_output_path):
    """
    Helper function to save a pandas dataframe to a csv file from a queue.
    Used by AbstractChallenge.evaluate to dump results to csv in parallel.
    """
    while True:
        data_job = q.get()
        if data_job is None:
            break
        data_job.to_csv(csv_output_path, mode='a', index=False, header=not os.path.exists(csv_output_path))
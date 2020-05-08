import os
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.train.estimator import Estimator
from azureml.train.dnn import TensorFlow

compute_target_name = "gpu-nc24-standard-exp";


def create_experiment_from_aml(experiment_name):
    print("Azure ML SDK Version: ", azureml.core.VERSION)
    ws = Workspace.from_config()
    print(ws.name, ws.location, ws.resource_group, sep='\t')
    experiment_name = experiment_name
    experiment = Experiment(workspace=ws, name=experiment_name)
    compute_target = ws.compute_targets[compute_target_name]

    script_folder = os.getcwd();

    script_params = {
    # to mount files referenced by mnist dataset
    '--data-folder': os.path.join(os.getcwd(), "data"),
    }

    estimator = Estimator(source_directory=script_folder,
              script_params=script_params,
              compute_target=compute_target,
              entry_script='build_lstm_model.py',
              pip_packages=['tqdm', 'nltk'],
              use_gpu=True)

    return experiment, estimator;


if __name__ == '__main__':
        # set up to aml connection
    experiment_name = 'slot_tagging_tf_exp'
    experiment, estimator = create_experiment_from_aml(experiment_name)
    current_run = experiment.submit(config=estimator)
    current_run.wait_for_completion()

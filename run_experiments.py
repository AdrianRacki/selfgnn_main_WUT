import subprocess


TEMPLATE_CMD = "python src/main.py --run_nfolds={n_folds} --experiment={experiment} --run_name=\"{run_name}\" model={model} model.experts.0.num_layers={num_layers} model.gating={gating} {other_options}"
MULTI_TEMPLATE_CMD = "python src/main.py --run_nfolds={n_folds} --experiment={experiment} --run_name=\"{run_name}\" model={model} model.gating={gating} {other_options}"



def run_experiment(n_folds: int, experiment: str, run_name: str, model: str, num_layers: int, gating: bool, other_options: str = ""):
    cmd = TEMPLATE_CMD.format(
        n_folds=n_folds,
        experiment=experiment,
        run_name=run_name,
        model=model,
        num_layers=num_layers,
        gating=str(gating).lower(),
        other_options=other_options
    )
    print(f"Running command: {cmd}")
    
    subprocess.run(cmd, shell=True, check=True)
    
def run_multi_experiment(n_folds: int, experiment: str, run_name: str, model: str, gating: bool, other_options: str = ""):
    cmd = MULTI_TEMPLATE_CMD.format(
        n_folds=n_folds,
        experiment=experiment,
        run_name=run_name,
        model=model,
        gating=str(gating).lower(),
        other_options=other_options
    )
    print(f"Running command: {cmd}")
    
    subprocess.run(cmd, shell=True, check=True)
    
# Cross layers experiments

def run_cross_reg_experiments():
    RUN_NAME_TEMPLATE = "CROSS_EXP-GIN-{experiment}-{cross_layers}-"
    n_folds = 5
    gating = False
    experiments = [
        "lipo_base"]
    model = "GIN"
    other_options_template =  " model.experts.0.cross_layers={cross_layers} trainer.trainer.min_epochs=200 trainer.trainer.min_epochs=200"
    cross_layers_list = [0, 1, 2, 3, 4, 5, 6]

    for experiment in experiments:
        for cross_layers in cross_layers_list:
            run_name = RUN_NAME_TEMPLATE.format(experiment=experiment, cross_layers=cross_layers)
            other_options = other_options_template.format(cross_layers=cross_layers)
            run_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                num_layers=2,
                gating=gating,
                other_options=other_options
            )
            
def run_cross_classification_experiments():
    RUN_NAME_TEMPLATE = "CROSS_EXP-GIN-{experiment}-{cross_layers}"
    n_folds = 0
    gating = False
    experiments = [
        "bace_base",
        "bbbp_base",
        "tox_base"]
    model = "GIN"
    other_options_template =  " model.experts.0.cross_layers={cross_layers} trainer.trainer.min_epochs=200 trainer.trainer.max_epochs=200"
    cross_layers_list = [0, 1, 2, 3, 4, 5]

    for experiment in experiments:
        for cross_layers in cross_layers_list:
            run_name = RUN_NAME_TEMPLATE.format(experiment=experiment, cross_layers=cross_layers)
            other_options = other_options_template.format(cross_layers=cross_layers)
            run_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                num_layers=2,
                gating=gating,
                other_options=other_options
            )
          
def run_2_layer_reg_baselines():
    RUN_NAME_TEMPLATE = "BASELINE_EXP-{model}-{experiment}-2layer"
    # other_options = " trainer.trainer.min_epochs=200 trainer.trainer.max_epochs=200"
    # cross_other_options = " model.experts.0.cross_layers={cross_layers}"
    n_folds = 5
    gating = False
    experiments = [
        "mp_base",
        "esolv_base",
        "freesolv_base",
        "lipo_base"]
    models = ["GIN", "GAT", "GCN"]
    for experiment in experiments:
        for model in models:
            run_name = RUN_NAME_TEMPLATE.format(experiment=experiment, model=model)
            run_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                num_layers=2,
                gating=gating,
                other_options=""
            )

def run_2_layer_reg_cross():
    RUN_NAME_TEMPLATE = "BASELINE_CROSS_EXP-{model}-{experiment}-2layer"
    other_options = " model.experts.0.cross_layers=1"
    n_folds = 5
    gating = False
    experiments = [
        "mp_base",
        "esolv_base",
        "freesolv_base",
        "lipo_base"]
    models = ["GIN"]
    for experiment in experiments:
        for model in models:
            run_name = RUN_NAME_TEMPLATE.format(experiment=experiment, model=model)
            run_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                num_layers=2,
                gating=gating,
                other_options=other_options
            )

def run_2_layer_classification_baselines():
    RUN_NAME_TEMPLATE = "BASELINE_EXP-{model}-{experiment}-2layer"
    n_folds = 5
    gating = False
    experiments = [
        "bace_base",
        "bbbp_base",
        "tox_base",
        ]
    models = ["GIN", "GAT", "GCN"]
    for experiment in experiments:
        for model in models:
            run_name = RUN_NAME_TEMPLATE.format(experiment=experiment, model=model)
            run_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                num_layers=2,
                gating=gating,
                other_options=""
            )

def run_2_layer_classification_cross():
    RUN_NAME_TEMPLATE = "BASELINE_CROSS_EXP-{model}-{experiment}-2layer"
    other_options = " model.experts.0.cross_layers=2"
    n_folds = 5
    gating = False
    experiments = [
        "bace_base",
        "bbbp_base",
        # "tox_base",
        ]
    models = ["GIN"]
    for experiment in experiments:
        for model in models:
            run_name = RUN_NAME_TEMPLATE.format(experiment=experiment, model=model)
            run_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                num_layers=2,
                gating=gating,
                other_options=other_options
            )

def run_multi_layer_experiments():
    print("Starting parallel experiments")
    experiments = [
        # "lipo_base",
        # "freesolv_base",
        # "esolv_base",
        # "mp_base",
        # "bace_base",
        # "bbbp_base",
        "tox_base",
        ]
    n_folds = 0
    gating = True
    models = [
        "MoE_4_GIN_cross",
        "MoE_4_GIN",
        "MoE_6_GIN_cross",
        "MoE_6_GIN",
        "MoE_8_GIN_cross",
        "MoE_8_GIN",       
        "MoE_10_GIN_cross",
        "MoE_10_GIN",
        "MoE_12_GIN_cross",
        "MoE_12_GIN", 
        "MoE_20_GIN_cross",
        "MoE_20_GIN",         
    ]
    other_options = " trainer.trainer.min_epochs=200 trainer.trainer.max_epochs=200"
    for experiment in experiments:
        for model in models:
            is_cross = "cross" if "cross" in model else "baseline"
            num_layers = model.split("_")[1]
            run_name = f"MULTI_LAYER_EXP-{model}-{experiment}-{is_cross}-parallel-{num_layers}"
            run_multi_experiment(
                n_folds=n_folds,
                experiment=experiment,
                run_name=run_name,
                model=model,
                gating=gating,
                other_options=other_options
            )
    print("Starting stacked experiments")
    models_stacked = ["GIN"]
    gating = False
    n_layers = [2,4,6,8]
    cross_layers = [0,1]
    for experiment in experiments:
        for model in models_stacked:
            for n_layer in n_layers:
                for cross_layer in cross_layers:
                    other_options = f" model.experts.0.cross_layers={cross_layer} trainer.trainer.min_epochs=200 trainer.trainer.max_epochs=200"
                    is_cross = "cross" if cross_layer > 0 else "baseline"
                    run_name = f"MULTI_LAYER_EXP-{model}-{experiment}-{is_cross}-stacked-{n_layer}"
                    run_multi_experiment(
                        n_folds=n_folds,
                        experiment=experiment,
                        run_name=run_name,
                        model=model,
                        gating=gating,
                        other_options=f" model.experts.0.num_layers={n_layer} " + other_options
                    )
    
if __name__ == "__main__":
    run_cross_reg_experiments() # done
    run_cross_classification_experiments() # done
    run_2_layer_reg_baselines() # done
    run_2_layer_reg_cross() # done
    run_2_layer_classification_baselines() # done
    run_2_layer_classification_cross() # done
    run_multi_layer_experiments() # done
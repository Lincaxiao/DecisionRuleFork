import yaml


def get_default_args(dataset, _algo='default'):
    algo = 'LDRPM' if _algo == "default" else _algo

    defaults = {}

    # layers related parameters
    defaults["hidden_dims"] = [256, 256]
    defaults["model"] = "mlp"
    defaults["algo"] = algo  # LDRPM, DC3, POCS
    defaults["dc3_lr"] = 1e-4
    defaults["dc3_momentum"] = 0.5
    defaults["dc3_softweighteqfrac"] = 0.5
    defaults["dc3_softweight"] = 100
    defaults['ldr_temp'] = 10
    defaults["eq_tol"] = 1e-4
    defaults["ineq_tol"] = 1e-4
    defaults["max_iters"] = 300
    # dataset related parameters
    defaults["data_generator"] = True
    defaults["renew_freq"] = 20
    # layers related parameters
    defaults["hidden_dims"] = [256, 256]
    defaults["act_cls"] = "relu"
    defaults["batch_norm"] = True
    # training related parameters
    defaults["loss_type"] = "obj"
    defaults["optimizer"] = "adam"
    defaults["lr"] = 0.0001
    defaults["weight_decay"] = 1e-8
    defaults["batch_size"] = 64
    defaults["bsz_factor"] = 20
    defaults["epochs"] = 200
    defaults["pretrain_epochs"] = 100

    if algo == "DC3":
        defaults["max_iters"] = 10  # DC3 iterations
        if dataset == "case200_activ":
            defaults["dc3_softweight"] = 10000

    mapping = {'default': 0, 'POCS': 1, 'LDRPM': 2, 'DC3': 3}

    with open(f"./cfg/{dataset}_{mapping[_algo]}", "w") as yaml_file:
        yaml.dump(defaults, yaml_file, default_flow_style=False)

    print(f"Default Configuration file saved to ./cfg/{dataset}_{mapping[_algo]}")



def exps_args(dataset):
    defaults = {}

    for algo in ["POCS", "LDRPM", "DC3"]:
        defaults["algo"] = algo
        # layers related parameters
        defaults["hidden_dims"] = [256, 256]
        defaults["act_cls"] = "relu"
        defaults["batch_norm"] = True
        defaults["model"] = "mlp"
        defaults["dc3_softweighteqfrac"] = 0.5
        defaults['ldr_temp'] = 10
        defaults["eq_tol"] = 1e-4
        defaults["ineq_tol"] = 1e-4
        # dataset related parameters
        defaults["data_generator"] = True
        defaults["renew_freq"] = 20
        # training related parameters
        defaults["loss_type"] = "obj"
        defaults["optimizer"] = "adam"
        defaults["lr"] = 0.0001
        defaults["weight_decay"] = 1e-8
        defaults["batch_size"] = 64
        defaults["bsz_factor"] = 20

        defaults["dc3_lr"] = 1e-4
        defaults["dc3_momentum"] = 0.5

        if algo == 'POCS':
            # compare different softweight
            defaults["dc3_softweight"] = 100
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 100  # always 100
            defaults["epochs"] = 200  # always 200
            idx = 1
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 1000
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 100  # always 100
            defaults["epochs"] = 200  # always 200
            idx = 4
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 10000
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 100  # always 100
            defaults["epochs"] = 200  # always 200
            idx = 7
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

        if algo == 'LDRPM':
            # compare pretrain and no pretrain
            defaults["dc3_softweight"] = 100  # always 100
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 100
            defaults["epochs"] = 200
            idx = 2
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 100  # always 100
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 0
            defaults["epochs"] = 300
            idx = 5
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

        if algo == 'DC3':
            # compare different softweight
            defaults["dc3_softweight"] = 100
            defaults["max_iters"] = 300  # always 10
            defaults["pretrain_epochs"] = 100  # always 100
            defaults["epochs"] = 200  # always 200
            idx = 3
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 1000
            defaults["max_iters"] = 300  # always 10
            defaults["pretrain_epochs"] = 100  # always 100
            defaults["epochs"] = 200  # always 200
            idx = 6
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 10000
            defaults["max_iters"] = 300  # always 10
            defaults["pretrain_epochs"] = 100  # always 100
            defaults["epochs"] = 200  # always 200
            idx = 9
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)


def lhs_exps_args(dataset):
    defaults = {}

    for algo in ["LDRPM", "DC3", "POCS"]:
        defaults["algo"] = algo
        # layers related parameters
        defaults["hidden_dims"] = [256, 256]
        defaults["act_cls"] = "relu"
        defaults["batch_norm"] = True
        defaults["model"] = "mlp"
        defaults["dc3_softweighteqfrac"] = 0.5
        defaults['ldr_temp'] = 10
        defaults["eq_tol"] = 1e-4
        defaults["ineq_tol"] = 1e-4
        # dataset related parameters
        defaults["data_generator"] = True
        defaults["renew_freq"] = 20
        # training related parameters
        defaults["loss_type"] = "obj"
        defaults["optimizer"] = "adam"
        defaults["lr"] = 0.0001
        defaults["weight_decay"] = 1e-8
        defaults["batch_size"] = 64
        defaults["bsz_factor"] = 20

        defaults["dc3_lr"] = 1e-4
        defaults["dc3_momentum"] = 0.5

        if algo == 'LDRPM':
            # compare pretrain and no pretrain
            defaults["dc3_softweight"] = 1  # always 1
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 150
            defaults["epochs"] = 350
            idx = 1
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 1  # always 1
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 0
            defaults["epochs"] = 500
            idx = 3
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

        if algo == 'DC3':
            # compare different softweight
            defaults["dc3_softweight"] = 1
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 150  # always 100
            defaults["epochs"] = 350  # always 200
            idx = 2
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

            defaults["dc3_softweight"] = 10
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 150  # always 100
            defaults["epochs"] = 350  # always 200
            idx = 4
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)

        if algo == 'POCS':
            # compare different softweight
            defaults["dc3_softweight"] = 1
            defaults["max_iters"] = 300  # always 300
            defaults["pretrain_epochs"] = 150  # always 100
            defaults["epochs"] = 350  # always 200
            idx = 5
            with open(f"./cfg/{dataset}_{idx}", "w") as yaml_file:
                yaml.dump(defaults, yaml_file, default_flow_style=False)


# for dataset in ['case14_ieee', 'case30_ieee', 'case57_ieee', 'case118_ieee', 'case200_activ']:
#     for algo in ['default', 'POCS', 'LDRPM', 'DC3']:
#         get_default_args(dataset, algo)

for dataset in ['case14_ieee', 'case30_ieee', 'case57_ieee', 'case118_ieee', 'case200_activ']:
    exps_args(dataset)
    print(f"Configuration files of dataset {dataset} saved to ./cfg/")


for dataset in ['portfolio_optimization']:
    lhs_exps_args(dataset)
    print(f"Configuration files of dataset {dataset} saved to ./cfg/")
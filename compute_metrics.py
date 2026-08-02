import torch_fidelity


def main():
    # Compute metrics between two directories
    metrics_dict = torch_fidelity. calculate_metrics(
        input1='true_flowers_train',
        input2='true_flowers_test',
        cuda=True,
        isc=True, 
        fid=True, 
        kid=True, 
        prc=True,
        verbose=True,
    )

    print(metrics_dict)

    fid = metrics_dict['frechet_inception_distance']
    kid = metrics_dict['kernel_inception_distance_mean']
    isc = metrics_dict['inception_score_mean']


    print("-- Summary of computed metrics --")
    print(f"FID: {fid}")
    print(f"KID: {kid}")
    print(f"ISC: {isc}")



if __name__ == "__main__":
    main()
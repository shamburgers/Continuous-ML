import numpy as np
import json

def generate_anomalies(data, mask, LS_distribution, spread_distribution, anomaly_strength=2.0, seed=None, tol=1e-8):

    if seed is not None:
        np.random.seed(seed)

    # Create a copy of the original data to avoid modifying it
    data_copy = data.astype(np.float64)

    nonzero_entries = ~np.all(np.isclose(data_copy, 0.0), axis=(1, 2, 3, 4))

    data_copy = data_copy[nonzero_entries]
    
    N = data_copy.shape[0]  # Length of the dataset
    idx = 0  # Start index for processing

    # Store info for each anomaly
    anomalies_info = []

    #Save seed value
    anomalies_info.append({"seed": str(seed)})

    # Iterate over the entire dataset in time-window LS chunks
    while idx < N:
        L = LS_distribution()  # Generate a random chunk size L
        end_idx = min(idx + L, N)  # Ensure we don't go out of bounds

        # Check if LS are zero valued in DigiOccupancy
        for i in range(idx, end_idx):

            #Check if data is zero valued. This shouldn't occur as the data is already removed of zero valued entries.
            if np.all(np.isclose(data_copy[i, :, :, :, 0], 0.0, atol=tol)):
                print(f"Skipping index {i} (zero filled LS).")
                idx = i+1 # Move index along
                continue  # Skip this entry

        # Find random location for the anomaly within the HCAL mask
        valid_indices = np.argwhere(np.isclose(mask, 1.0, atol=tol))

        # Make sure valid location in mask doesnt contain zero valued DigiOccupancy
        anomaly_location = None
        while anomaly_location is None:
            candidate_location = valid_indices[np.random.choice(len(valid_indices))]
            eta, phi, depth = candidate_location

            # Check if the selected location is non-zero across the whole chunk
            if not np.any(np.isclose(data_copy[idx:end_idx, eta, phi, depth, 0], 0.0, atol=tol)):
                anomaly_location = (eta, phi, depth)

        # Sample spread independently for each dimension
        spread_eta = spread_distribution()
        spread_phi = spread_distribution()
        spread_depth = spread_distribution()

        if idx < 10:
            print(f"{idx} eta, phi, depth: {eta}, {phi}, {depth}")
            print(f"{idx} eta_spread, phi_spread, depth_spread, L: {spread_eta}, {spread_phi}, {spread_depth}, {L}")

        # Apply anomaly with mask-aware spread
        for ieta in range(max(0, eta - spread_eta), min(64, eta + spread_eta + 1)):
            for iphi in range(max(0, phi - spread_phi), min(72, phi + spread_phi + 1)):
                for idepth in range(max(0, depth - spread_depth), min(7, depth + spread_depth + 1)):
                    # Only modify locations where the HCAL mask value is 1.0
                    if np.isclose(mask[ieta, iphi, idepth], 1.0, atol=tol):
                        data_copy[idx:end_idx, ieta, iphi, idepth, 0] *= anomaly_strength

        # Store information about the anomaly
        anomalies_info.append({
            "start_index": str(idx),
            "L_chunk_size": str(L),
            "anomaly_location": (str(eta), str(phi), str(depth)),
            "x_spread": str(spread_eta),
            "y_spread": str(spread_phi),
            "z_spread": str(spread_depth)
        })

        idx = end_idx  # Move to the next chunk

    return data_copy, anomalies_info

if __name__ == "__main__":

    # HCAL mask
    mask = np.load('../data/HCAL_CONFIG/he_segmentation_config_mask.npy')  

    # Upper limit for how many LS the anomaly persists for
    time_window = 5

    # Upper limit for the spatial spread of the anomaly
    spatial_spread = 3

    LS_distribution = lambda: np.random.randint(1, time_window)
    spread_distribution = lambda: np.random.randint(0, spatial_spread)
    seed_selection = lambda: np.random.randint(0,10000)

    anomaly_strength = [2.0, 0.8, 0.6, 0.4, 0.2, 0.0]

    # for runid in ['Run323940', 'Run323997', 'Run324021', 'Run324022', 'Run325117', 'Run325170']:
    for runid in ['Run355456','Run355680','Run355769','Run356381','Run357081','Run357112','Run357442','Run357479','Run357612','Run357815','Run357899','Run359694','Run359764','Run360019','Run360459','Run360820','Run361240','Run361957','Run362091','Run362760']:

        dir_folder = f'../data/HE/he_master_npdataset_2022/{runid}/'
        file = dir_folder+'output__depth.npy'

        data = np.load(file, allow_pickle=True)

        seed = seed_selection()

        for anml_strg in anomaly_strength:
            # Generate anomalies
            modified_data, anomalies_info = generate_anomalies(
                data, mask, LS_distribution, spread_distribution, anomaly_strength=anml_strg, seed=seed
            )

            save_file = dir_folder + f"anomalous_data_{anml_strg}.npy"
            np.save(save_file, modified_data)
            print(f"Anomalous dataset saved to: {save_file}")

            json_file = dir_folder +f"anomalous_json_{anml_strg}.json"
            with open(json_file, 'w') as f:
                json.dump(anomalies_info, f, indent=4)
            print(f"Anomaly information saved to {json_file}")


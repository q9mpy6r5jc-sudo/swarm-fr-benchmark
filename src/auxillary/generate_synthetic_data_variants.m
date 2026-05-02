%% GeneSpider2 Simulation: Combinatorial Variants (Structure x Noise, Magnitude x Noise)

clear; clc; close all;
rng(42);

% --- Configuration ---
N = 1000;            % Number of genes
S = 3;               % Sparsity (Average degree)
cells_per_pert = 50; % Cells per perturbation
n_struct_levels = 5;

% Define SNR levels (Inf = No Noise)
snr_levels = [Inf, 1.0, 0.1, 0.01]; 
shrink_factors = [0.8, 0.6, 0.4, 0.2, 0.0];

% Main Output Directory
main_dir = 'synthetic_benchmark_dataset';
if ~exist(main_dir, 'dir')
    mkdir(main_dir);
end

% --- Helper Function Handle ---
save_data = @(folder, Y, SCC, P, A, Info) save_simulation_files(folder, Y, SCC, P, A, Info);

%% 1. Construct GRN_GT and Data_GT
fprintf('Constructing Ground Truth...\n');
% Create Network and Stabilize
A_GT = datastruct.scalefree2(N, S);
A_GT = datastruct.stabilize(A_GT, 'iaa', 'low');

% Create Perturbation Matrix (Perturb every gene once)
P_GT = -repmat(eye(N), 1, cells_per_pert);

% Generate Clean Data (SNR = Inf)
[Y_GT, ~, ~, ~, SCC_GT] = datastruct.scdata(A_GT, P_GT, 'SNR', Inf, 'raw_counts', true);
Y_GT = round(Y_GT); 
SCC_GT = round(SCC_GT);
Y_GT(Y_GT < 0) = 0;
SCC_GT(SCC_GT < 0) = 0;

gt_folder = fullfile(main_dir, 'variants', 'GT_Reference');
save_data(gt_folder, Y_GT, SCC_GT, P_GT, A_GT, 'Ground Truth: 0 Noise, Full Structure');
fprintf('   > GT Saved to: %s\n', gt_folder);

%% 2. Structure x Noise Sweep
fprintf('\nGenerating Structure x Noise Variants...\n');

% Calculate Gene Degrees (In-degree + Out-degree)
gene_degrees = sum(abs(A_GT), 1) + sum(abs(A_GT), 2)';
[~, ranked_gene_indices] = sort(gene_degrees, 'descend'); % Highest degrees first
total_genes = length(ranked_gene_indices);

for i = 1:n_struct_levels
    frac = i * 0.2;
    n_target = floor(total_genes * frac);
    target_genes = ranked_gene_indices(1:n_target);
    
    % Sever edges for the target hub genes
    current_A = A_GT;
    current_A(target_genes, :) = 0;
    current_A(:, target_genes) = 0;
    
    for j = 1:length(snr_levels)
        current_snr = snr_levels(j);
        
        % Generate Data with degraded structure and specific SNR
        [Y_struct, ~, ~, ~, SCC_struct] = datastruct.scdata(current_A, P_GT, 'SNR', current_snr, 'raw_counts', true);
        Y_struct = round(Y_struct);
        SCC_struct = round(SCC_struct);
        Y_struct(Y_struct < 0) = 0;
        SCC_struct(SCC_struct < 0) = 0;
        
        if isinf(current_snr)
            snr_str = 'Infinity';
        else
            snr_str = num2str(current_snr);
        end
        
        folder_name = sprintf('Structure_Level_%d_SNR_%s', i, snr_str);
        curr_folder = fullfile(main_dir, 'variants', folder_name);
        
        info_str = sprintf('Structure Level %d (Frac: %g) | SNR: %s', i, frac, snr_str);
        save_data(curr_folder, Y_struct, SCC_struct, P_GT, current_A, info_str);
        
        fprintf('   > Saved %s\n', folder_name);
    end
end

%% 3. Magnitude x Noise Sweep
fprintf('\nGenerating Magnitude x Noise Variants...\n');

mu_ctrl = mean(SCC_GT, 2);

for i = 1:length(shrink_factors)
    shrink_factor = shrink_factors(i);
    
    % Shrink the raw base signal
    effect = Y_GT - repmat(mu_ctrl, 1, size(Y_GT, 2));
    Y_mag_clean = round(repmat(mu_ctrl, 1, size(Y_GT, 2)) + (effect * shrink_factor));
    Y_mag_clean(Y_mag_clean < 0) = 0;
    
    for j = 1:length(snr_levels)
        current_snr = snr_levels(j);
        
        if isinf(current_snr)
            Y_final = Y_mag_clean;
            SCC_final = SCC_GT;
            snr_str = 'Infinity';
        else
            % Generate a standard noisy dataset from GT
            [Y_noisy, ~, ~, ~, SCC_noisy] = datastruct.scdata(A_GT, P_GT, 'SNR', current_snr, 'raw_counts', true);
            
            % Calculate the delta (noise matrix)
            Noise_Matrix = Y_noisy - Y_GT;
            Control_Noise_Matrix = SCC_noisy - SCC_GT;
            
            % Add the noise delta to the shrunk signal
            Y_final = round(Y_mag_clean + Noise_Matrix);
            SCC_final = round(SCC_GT + Control_Noise_Matrix);
            
            Y_final(Y_final < 0) = 0;
            SCC_final(SCC_final < 0) = 0;
            snr_str = num2str(current_snr);
        end
        
        folder_name = sprintf('Magnitude_Level_%d_SNR_%s', i, snr_str);
        curr_folder = fullfile(main_dir, 'variants', folder_name);
        
        info_str = sprintf('Magnitude Level %d (Shrink: %g) | SNR: %s', i, shrink_factor, snr_str);
        save_data(curr_folder, Y_final, SCC_final, P_GT, A_GT, info_str);
        
        fprintf('   > Saved %s\n', folder_name);
    end
end

fprintf('\nAll data saved in "%s".\n', main_dir);

function save_simulation_files(folder_path, Y, SCC, P, A, description)
    if ~exist(folder_path, 'dir')
        mkdir(folder_path);
    end

    writematrix(Y', fullfile(folder_path, 'Y_counts.csv'));              
    writematrix(SCC', fullfile(folder_path, 'Y_control_counts.csv'));    
    writematrix(P', fullfile(folder_path, 'P_perturbation.csv'));
    writematrix(A, fullfile(folder_path, 'A_network.csv'));
    
    % Info txt
    fid = fopen(fullfile(folder_path, 'info.txt'), 'w');
    fprintf(fid, '%s', description);
    fclose(fid);
end
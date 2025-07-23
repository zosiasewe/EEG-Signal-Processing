clc
clear all 
close all

closed_path = fullfile(pwd, 'closed_nose', 'AC.mat');
% lalala
tmp = load(closed_path);
data = struct2array(tmp);  % teraz: [160 × 1280 × 17]

data_perm = permute(data, [3, 2, 1]);  % → [channels × time × trials] = [17 × 1280 × 160]
data_2d = reshape(data_perm, 17, []);  % → [17 × (1280 * 160)] = [17 × 204800]
% 1280 punktów czasowych/trial × 160 trials = 204800 próbek na kanał
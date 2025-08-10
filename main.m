clear;
close all;

video = VideoReader('VIDEO.mp4');

filepath = "pc_section.las";
lasReader = lasFileReader(filepath);
pc_data = readPointCloud(lasReader);
pc  = pc_data.Location;

%Display original point cloud
%figure;
%pcshow(pc);

% Intrinsics parameters 
focalLength    = [4600 4500]; 
principalPoint = [975 590];
imageSize      = [2000 2000];
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

% Extrinsic parameters
translation = [0 0 3.5]; 
eul = deg2rad([259,16.5, 80]); % [roll, pitch, yaw]
R = eul2rotm(eul, 'ZYX');
tform = rigidtform3d(R, translation);
imgHeight = 1080;
imgWidth = 1920;

% Inicializar la nube de puntos combinada
combinedPC = pointCloud(zeros(0, 3));

t = 0;
inc_t = 1;
y_inc = 0.5;
y_init = 0;
y_end = 20;


for y = y_init:y_inc:y_end
    
    section_pts = pc(pc(:, 2) >= y & pc(:, 2) <= y+y_inc, :);
    pc_section = pointCloud(section_pts);
    [imPts, isValid] = projectLidarPointsOnImage(pc_section, intrinsics, tform);
    
    video.CurrentTime = t;
    img = readFrame(video);

    x_proj = round(imPts(:, 1)); 
    y_proj = round(imPts(:, 2)); 

    valid_indices = (x_proj > 0 & x_proj <= imgWidth) & (y_proj > 0 & y_proj <= imgHeight);
    
    x_proj_valid = x_proj(valid_indices);
    y_proj_valid = y_proj(valid_indices);
    
    pc_valid = section_pts(isValid, :);
    pc_valid = pc_valid(valid_indices, :);  
    
    %Extraer valores RGB
    rgb_values = zeros(length(x_proj_valid), 3);

    for j = 1:length(x_proj_valid)
        rgb_values(j, :) = double(img(y_proj_valid(j), x_proj_valid(j), :))/255;
    end

    %Asignar los colores a los puntos
    coloredPointCloud = pointCloud(pc_valid, 'Color', rgb_values);
    
    %Juntar las nubes de puntos
    combinedPC = pcmerge(combinedPC, coloredPointCloud,0.01);  % Ajustar resolución
    
    t = t + inc_t; %incremento del tiempo

end

%Nube de puntos completa
%figure(2);
%pcshow(combinedPC);

%Mesh
depth = 12;
mesh_ = pc2surfacemesh(combinedPC, "poisson",depth);

figure(3);
surfaceMeshShow(mesh_);


% % Estimación del diámetro reconstruido
% y_steps = 0:0.24:20; % Coincide con y_inc y y_end
% diameters = zeros(length(y_steps)-1, 1);
% reference_diameter = 0.9; % Diámetro original en unidades (ajústalos según datos reales)
% 
% for i = 1:length(y_steps)-1
%     % Filtrar puntos de la malla en el rango y
%     y_range = [y_steps(i) y_steps(i+1)];
%     idx = combinedPC.Location(:, 2) >= y_range(1) & combinedPC.Location(:, 2) <= y_range(2);
%     section_points = combinedPC.Location(idx, :);
% 
%     if ~isempty(section_points)
%         % Calcular el diámetro como la distancia máxima entre puntos en x-z
%         xz_points = section_points(:, [1 3]); % Proyección en plano x-z
%         if size(xz_points, 1) > 1
%             distances = pdist(xz_points, 'euclidean');
%             diameters(i) = max(distances); % Diámetro aproximado
%         else
%             diameters(i) = 0; % Si no hay suficientes puntos
%         end
%     end
% end
% 
% % Calcular error promedio
% valid_diameters = diameters(diameters > 0);
% if ~isempty(valid_diameters)
%     mean_reconstructed_diameter = mean(valid_diameters);
%     absolute_error = abs(mean_reconstructed_diameter - reference_diameter);
%     relative_error = (absolute_error / reference_diameter) * 100; % En porcentaje
% 
%     fprintf('Mean reconstructed diameter: %.3f m\n', mean_reconstructed_diameter);
%     fprintf('Absolute error: %.3f m\n', absolute_error);
%     fprintf('Relative error: %.2f%%\n', relative_error);
% else
%     fprintf('No valid diameters calculated.\n');
% end
% 
% 
% % Convertir malla a nube de puntos para comparación
% mesh_points = mesh_.Vertices; % Puntos de la malla
% original_points = pc; % Usar pc como matriz Nx3 de coordenadas originales
% 
% % Alineación aproximada: traslación basada en centroides
% mesh_centroid = mean(mesh_points, 1);
% original_centroid = mean(original_points, 1);
% translated_mesh = mesh_points - mesh_centroid + original_centroid;
% 
% % Submuestreo para reducir el tamaño (5% de los puntos, ajustable)
% sample_ratio = 0.05; % Reducir a 5% para manejar memoria
% num_samples_mesh = min(floor(size(translated_mesh, 1) * sample_ratio), 5000); % Límite a 5,000 puntos
% num_samples_orig = min(floor(size(original_points, 1) * sample_ratio), 5000);
% sampled_mesh = translated_mesh(randsample(size(translated_mesh, 1), num_samples_mesh), :);
% sampled_orig = original_points(randsample(size(original_points, 1), num_samples_orig), :);
% 
% % Calcular distancias bidireccionales con rangesearch
% radius = 0.45; % Radio en metros (mantenido según tu código)
% 
% % Mesh a Original (para RMSE y primera parte de Chamfer)
% [idx_m2o, dists_m2o] = rangesearch(sampled_orig, sampled_mesh, radius);
% dist_m2o = zeros(num_samples_mesh, 1);
% for i = 1:length(idx_m2o)
%     if ~isempty(idx_m2o{i})
%         dist_m2o(i) = min(dists_m2o{i}); % Distancia mínima
%     else
%         dist_m2o(i) = NaN; % Si no hay puntos dentro del radio
%     end
% end
% dist_m2o = dist_m2o(~isnan(dist_m2o)); % Eliminar NaN
% 
% % Original a Mesh (para segunda parte de Chamfer)
% [idx_o2m, dists_o2m] = rangesearch(sampled_mesh, sampled_orig, radius);
% dist_o2m = zeros(num_samples_orig, 1);
% for i = 1:length(idx_o2m)
%     if ~isempty(idx_o2m{i})
%         dist_o2m(i) = min(dists_o2m{i});
%     else
%         dist_o2m(i) = NaN;
%     end
% end
% dist_o2m = dist_o2m(~isnan(dist_o2m)); % Eliminar NaN
% 
% % Calcular RMSE (usando distancias de mesh a original)
% if ~isempty(dist_m2o)
%     rmse = sqrt(mean(dist_m2o.^2));
%     fprintf('Root Mean Square Error (RMSE) : %.3f m\n', rmse);
% else
%     fprintf('No points within the specified radius for RMSE calculation.\n');
% end
% 
% % Calcular Chamfer Distance (promedio bidireccional)
% if ~isempty(dist_m2o) && ~isempty(dist_o2m)
%     chamfer_dist = mean(dist_m2o) + mean(dist_o2m);
%     fprintf('Chamfer Distance: %.3f m\n', chamfer_dist);
% else
%     fprintf('No points within the specified radius for Chamfer Distance calculation.\n');
% end



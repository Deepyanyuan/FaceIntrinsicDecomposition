close all
clear all
clc


inputMatFile = 'D:\04_paper\Mat data for oe\metricsList_both-nodeF_nosourceF_test_results_checkpoint.mat';
% inputMatFile = 'D:\04_paper\Mat data for oe\metricsList_both-nodeF_nosourceT_test_results_checkpoint.mat';
% inputMatFile = 'D:\04_paper\Mat data for oe\metricsList_both-nodeT_nosourceF_test_results_checkpoint.mat';
% inputMatFile = 'D:\04_paper\Mat data for oe\metricsList_both-nodeT_nosourceT_test_results_checkpoint.mat';
% inputMatFile = 'D:\04_paper\Mat data for oe\metricsList_P-resize-nodeT_nosourceF_test_results_checkpoint.mat';
% inputMatFile = 'D:\04_paper\Mat data for oe\metricsList_P-resize-nodeT_nosourceT_test_results_checkpoint.mat';


load(inputMatFile);

init_M = zeros(6,7);
view1_1 = init_M;view1_2 = init_M;view1_3 = init_M;view1_4 = init_M;
view2_1 = init_M;view2_2 = init_M;view2_3 = init_M;view2_4 = init_M;

person_1 = metricsList(:,:,1:112);
person_2 = metricsList(:,:,113:224);

k = 28;

view1_1 = mean(metricsList(:,:,1:k), 3);view1_2 = mean(metricsList(:,:,(k+1):2*k), 3);view1_3 = mean(metricsList(:,:,(2*k+1):3*k), 3);view1_4 = mean(metricsList(:,:,(3*k+1):4*k), 3);
view2_1 = mean(metricsList(:,:,1:k), 3);view2_2 = mean(metricsList(:,:,(k+1):2*k), 3);view2_3 = mean(metricsList(:,:,(2*k+1):3*k), 3);view2_4 = mean(metricsList(:,:,(3*k+1):4*k), 3);

% view3_2 = (view1_1 + view1_4 + view2_1 + view2_4)/4;

view3_1 = (view1_1 + view2_1)/2;
view3_2 = (view1_2 + view2_2)/2;
view3_3 = (view1_3 + view2_3)/2;
view3_4 = (view1_4 + view2_4)/2;







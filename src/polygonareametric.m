function [Metrics]=polygonareametric(ActualLabel, PredictedLabel,isPlot)
%%%%%   WARNING   %%%%%%%
%NUMERICALLY LARGER CLASS WILL BE AUTOMATICALLY ASSIGNED AS PositiveClass
%THIS IS BECAUSE OF THE FUNCTION OF perfcurve. IT REQUIRES LARGER CLASS AS PositiveClass.

% INTRODUCTION:
%           This study proposes a stable and profound knowledge criterion that allows the performance of a classifier 
%           to be evaluated with only a single metric called as polygon area metric (PAM). This function is not only
%           calculates PAM value, but also gives Classification Accuracy (CA), Sensitivity (SE), Specificity (SP),
%           Kappa (K) and F  measure metrics. 
% 
% CITATION INFORMATION:
%           Please cite the following paper for the usage of PAM value:
%           Aydemir O., A New Performance Evaluation Metric for Classifiers: Polygon Area Metric, Journal of Classification, (2020). https://doi.org/10.1007/s00357-020-09362-5
% 
% USAGE OF THE FUNCTION:
% INPUTS;
%       -ActualLabel: Actual label of the trials (samples), 1xN dimension binary labels
%       -PredictedLabel: Predicted (estimated) label of the trials (samples), 1xN dimension binary labels
%       -isPlot: A logical value indicating whether the resultant figure will be drawn. Default is true
%
% OUTPUT;
%       -Metrics: This struct gives 7 evaluation metrics which are Polygon Area...
%       (PA), Classification_Accuracy (CA), Sensitivity (SE), Specificity...
%       (SP), AUC (AUC), Kappa (K), F_measure (F_M), respectively.
%        AUC: Area under curve value, which should be obtained by Receiver operating characteristic (ROC), 0<AUC<1
% 
% EXAMPLE;
%       -ActualLabel=[1 1 1 1 1 0 0 0 0];
%       -PredictedLabel=[1 1 1 0 0 0 0 0 1];
%       -[Metrics]=polygonareametric(ActualLabel,PredictedLabel)


%Code introduction
if nargin<2
    error('You have to supply all required input paremeters, which are ActualLabel, PredictedLabel')
end
if nargin < 3
    isPlot = true;
end

%plotting the widest polygon
A1=1;
A2=1;
A3=1;
A4=1;
A5=1;
A6=1;

a=[-A1 -A2/2 A3/2 A4 A5/2 -A6/2 -A1];
b=[0 -(A2*sqrt(3))/2 -(A3*sqrt(3))/2 0 (A5*sqrt(3))/2 (A6*sqrt(3))/2 0];

if isPlot
    figure
    plot(a, b, '--bo','LineWidth',1.3)
    axis([-1.5 1.5 -1.5 1.5]);
    hold on
    %grid
end


% Calculating the True positive (TP), False Negative (FN), False Positive...
% (FP),True Negative (TN), Classification Accuracy (CA), Sensitivity (SE), Specificity (SP),...
% Kappa (K) and F  measure (F_M) metrics
PositiveClass=max(ActualLabel);
NegativeClass=min(ActualLabel);
cp=classperf(ActualLabel,PredictedLabel,'Positive',PositiveClass,'Negative',NegativeClass);
 CM=cp.DiagnosticTable;
    TP=CM(1,1);
    FN=CM(2,1);
    FP=CM(1,2);
    TN=CM(2,2);
    CA=cp.CorrectRate;
    SE=cp.Sensitivity; %TP/(TP+FN)
    SP=cp.Specificity; %TN/(TN+FP)
    Pr=TP/(TP+FP);
    Re=TP/(TP+FN);
    F_M=2*Pr*Re/(Pr+Re);
    FPR=FP/(TN+FP);
    TPR=TP/(TP+FN);
    K=TP/(TP+FP+FN);
    [X1,Y1,T1,AUC] = perfcurve(ActualLabel,PredictedLabel,PositiveClass); 
    %ActualLabel(1) means that the first class is assigned as positive class
    %plotting the calculated CA, SE, SP, AUC, K and F_M on polygon
x=[-CA -SE/2 SP/2 AUC K/2 -F_M/2 -CA];
y=[0 -(SE*sqrt(3))/2 -(SP*sqrt(3))/2 0 (K*sqrt(3))/2 (F_M*sqrt(3))/2 0];

if isPlot
    plot(x, y, '-ko','LineWidth',1)
    fill(x, y,'r')
end

%calculating the PAM value
% Get the number of vertices
n = length(x);
% Initialize the area
p_area = 0;
% Apply the formula
for i = 1 : n-1
    p_area = p_area + (x(i) + x(i+1)) * (y(i) - y(i+1));
end
p_area = abs(p_area)/2;

%Normalization of the polygon area to one.
PA=p_area/2.59807;

if isPlot
    %Plotting the Polygon
    plot(0,0,'r+')
    plot([0 -A1],[0 0] ,'--ko')
    text(-A1-0.3, 0,'CA','FontWeight','bold')
    plot([0 -A2/2],[0 -(A2*sqrt(3))/2] ,'--ko')
    text(-0.59,-1.05,'SE','FontWeight','bold')
    plot([0 A3/2],[0 -(A3*sqrt(3))/2] ,'--ko')
    text(0.5, -1.05,'SP','FontWeight','bold')
    plot([0 A4],[0 0] ,'--ko')
    text(A4+0.08, 0,'AUC','FontWeight','bold')
    plot([0 A5/2],[0 (A5*sqrt(3))/2] ,'--ko')
    text(0.5, 1.05,'J','FontWeight','bold')
    plot([0 -A6/2],[0 (A6*sqrt(3))/2] ,'--ko')
    text(-0.65, 1.05,'FM','FontWeight','bold')
    grid
    daspect([1 1 1])
end
Metrics.PA=PA;
Metrics.CA=CA;
Metrics.SE=SE;
Metrics.SP=SP;
Metrics.AUC=AUC;
Metrics.K=K;
Metrics.F_M=F_M;

categories = {'Polygon Area';'Classification_Accuracy';'Sensitivity';'Specificity';'AUC';'Kappa'; 'F_measure'};
printVar = cell(7,2);
printVar(:,1)=categories;
printVar(:,2)={PA, CA, SE, SP, AUC, K, F_M};
disp('Results are:')
for i=1:length(categories)
    fprintf('%23s: %.2f \n', printVar{i,1}, printVar{i,2})
end
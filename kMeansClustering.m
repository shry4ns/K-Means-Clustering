%
%  k_means_clustering.m
%  Shryans Goyal, 
%  CAAM 335, 
%  August 29, 2018
%  

set(0, 'defaultaxesfontsize',18,'defaultaxeslinewidth',1.2,...
       'defaultlinelinewidth',1.0,'defaultpatchlinewidth',1.0,...
       'defaulttextfontsize',18);


%% Data matrix
%  Each cluster is obtained by generating vectors from a bivariate normal distribution 
%  with specified mean vector mu and covariance matrix Sigma.
Y = csvread('/Users/shryansgoyal/Desktop/K-Means-Clustering/breast-cancer-wisconsin.data');
X = Y(:, 2:10);

%% Initialization
% k = number of clusters
k     = 2;
% Mean = matrix of means; j-th column, Mean(:,j), is mean of cluster j
m = [ 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5 2.5  
         7.5 7.5 7.5 7.5 7.5 7.5 7.5 7.5 7.5];
Mean = transpose(m);

%% Iterations
nsamples  = size(X,1);
iter      = 0;
objective = [];
diff      = 1;
while iter < 11 && diff > 1.e-2
    % compute distances from currents means
    distance = zeros(2,nsamples);
    for i = 1:nsamples
        distance(1,i) = norm( Mean(:,1) - X(i,:) );
        distance(2,i) = norm( Mean(:,2) - X(i,:) );
    end
    
    % compute clusters (vectors that are clostest to mean)
    [~,cluster] = min( distance, [], 1);
    X1 = X(cluster==1, :);
    X2 = X(cluster==2, :);
    
%     % plot clusters
%     figure(2)
%     plot(Mean(1,1),Mean(2,1),'v','MarkerEdgeColor','k',...
%                          'MarkerFaceColor','r','MarkerSize',15); hold on
%     plot(Mean(1,2),Mean(2,2),'v','MarkerEdgeColor','k',...
%                          'MarkerFaceColor','b','MarkerSize',15);
%     plot(Mean(1,3),Mean(2,3),'v','MarkerEdgeColor','k',...
%                          'MarkerFaceColor','g','MarkerSize',15); 
%     plot(X1(1,:),X1(2,:),'o','MarkerEdgeColor','r',...
%                          'MarkerFaceColor','r','MarkerSize',5); hold on
%     plot(X2(1,:),X2(2,:),'o','MarkerEdgeColor','b',...
%                          'MarkerFaceColor','b','MarkerSize',5);
%     plot(X3(1,:),X3(2,:),'o','MarkerEdgeColor','g',...
%                          'MarkerFaceColor','g','MarkerSize',5); 
%     title([' Iteration ', num2str(iter)] )
%     axis([-8,8,-8,8]);
    
    % compute means of clusters
    Mean0     = Mean;
    Mean(:,1) = sum( X1, 1 ) / size(X1,1);
    Mean(:,2) = sum( X2, 1 ) / size(X2,1);
%     plot(Mean(1,1),Mean(2,1),'s','MarkerEdgeColor','k',...
%                          'MarkerFaceColor','r','MarkerSize',15); hold on
%     plot(Mean(1,2),Mean(2,2),'s','MarkerEdgeColor','k',...
%                          'MarkerFaceColor','b','MarkerSize',15);
%     plot(Mean(1,3),Mean(2,3),'s','MarkerEdgeColor','k',...
%                          'MarkerFaceColor','g','MarkerSize',15); hold off
  
    %eval(['print -depsc cluster_iteration',num2str(iter),'.eps'])
    
    iter = iter+1;
    
    % relative change in means
    diff = norm(Mean(:,1) - Mean0(:,1))/ norm(Mean(:,1)) ...
           + norm(Mean(:,2) - Mean0(:,2))/ norm(Mean(:,2));
       
    % objective function  value
    obj  = 0;
    for j = 1:size(X1,1)
        obj = obj + norm(X1(j,:)-Mean(:,1));
    end
    for j = 1:size(X2,1)
        obj = obj + norm(X2(j,:)-Mean(:,2));
    end
        
    objective = [objective, obj];
    
    fprintf( 'Iteration %3d,  relative change in means = %10.4e,  objective = %10.4e \n', iter, diff, obj )
    
end

% Compute accuracy of the prediction
cluster = transpose(cluster*2);
error = 0;
malignant = 0;
benign = 0;
for i=1:699
    if cluster(i) == Y(i,11) && cluster(i) == 2
        benign = benign + 1;
        error = error + 1/699;
    end
    if cluster(i) == Y(i,11) && cluster(i) == 4
        malignant = malignant + 1;
        error = error + 1/699;
    end
end
disp(["Overall accuracy: ", num2str(error*100)]);
disp(["Benign Error: ", num2str(476 - benign)]);
disp(["Malignant Error: ", num2str(223 - malignant)]);

c = categorical({'Benign',' Malignant'});
bar(c,[benign, 476 - benign; malignant, 223 - malignant]);
legend('Correct','Incorrect');
ylabel('Count');
% 
% figure(3)
% plot(objective, '*b')
% xlabel('Iteration')
% xlabel('Objective Function')


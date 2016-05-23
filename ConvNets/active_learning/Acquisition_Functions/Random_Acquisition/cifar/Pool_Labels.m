load('Score1.mat')
load('Score2.mat')

All_Std = zeros(size(score1,1), size(score1,2));
All_Mean = zeros(size(score1,1), size(score1,2));
Label_Prob = zeros(size(score1, 1),1);
Label_Class = zeros(size(score1, 1),1);
Std = zeros(size(score1, 1),1);

for i = 1:size(score1,1)
    for j = 1:size(score1,2)
        Y = [score1(i,j), score2(i,j) ];
        All_Mean(i,j) = mean(Y);
        All_Std(i,j) = std(Y);
    end
        [M,I] = max(All_Mean(i,:));        
        Label_Prob(i,:) = M;
        Label_Class(i,:) = I;
        Std(i,:) = All_Std(i, I);
end


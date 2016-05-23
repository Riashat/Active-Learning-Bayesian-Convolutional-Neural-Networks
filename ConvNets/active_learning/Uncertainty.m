load('Score0.mat')
load('Score1.mat')
load('Score2.mat')
load('Score3.mat')
load('Score4.mat')
load('Score5.mat')
% load('Score6.mat')
% load('Score7.mat')
% load('Score8.mat')
% load('Score9.mat')
% 
% All_Std = zeros(size(score1,1), size(score1,2));
% for i = 1:size(score1,1)
%     for j = 1:size(score1,2)
%         X = [score0(i,j), score1(i,j), score2(i,j),score3(i,j),score4(i,j),score5(i,j),score6(i,j),score7(i,j),score8(i,j),score9(i,j) ];
%         All_Std(i,j) = std(X);
%     end
% end


All_Std = zeros(size(score1,1), size(score1,2));
for i = 1:size(score1,1)
    for j = 1:size(score1,2)
        X = [score0(i,j), score1(i,j), score2(i,j),score3(i,j),score4(i,j),score5(i,j) ];
        All_Std(i,j) = std(X);
    end
end
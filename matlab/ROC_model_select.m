%% Load
model = {'ResNet-50';'FKNet';'FKNet+800'};
batch_size = 1;
for model_select = 1:3
    folder_path = ['../task/' 'model' num2str(model_select) '/'];
    No_subject = 190;
    No_set = 6;
    
    %load([folder_path 'scores-' num2str(batch_size) '.mat']) %for model_select = 1:4
    %outname = '';
    
    load([folder_path 'scores-align-' num2str(batch_size) '.mat']) %for model_select = 1:10
    outname = '-align';
    
    %load([folder_path 'scores-align-match-' num2str(batch_size) '.mat']) %for model_select = 5:10
    %outname = '-align-match';
    
    %load([folder_path 'scores-align-match-cos-' num2str(batch_size) '.mat']) %for model_select = 5:10
    %outname = '-align-match-cos';

    %% Genuine and Imposter Scores
    D_genuine = zeros(1, No_subject, 'single');
    D_imposter = zeros(1,No_subject*(No_set-1), 'single');

    counter_genuine = 1;
    counter_imposter = 1;
    for i = 1:size(Pred,1)
        for j = 1:size(Pred,2)
            D = -Pred(i,j);
            if(Actual(i) == j-1) %Label starts from 0
                D_genuine(counter_genuine) = D;
                counter_genuine = counter_genuine + 1;
            else
                D_imposter(counter_imposter) = D;
                counter_imposter = counter_imposter + 1;
            end
        end
    end

    %% Evaluation
    disp('Evaluation');

    v_length = 2000;
    TPR = zeros(1,v_length+1);
    FPR = zeros(1,v_length+1);
    db_count = 1;
    progress_count = 0;
    for decision_boundary = min(D_genuine):(max(D_imposter)-min(D_genuine))/v_length:max(D_imposter) %limit by imposter size: 1/size(D_imposter,2)
        TP = length(find(D_genuine <= decision_boundary));
        FN = length(D_genuine) - TP;
        FP = length(find(D_imposter <= decision_boundary));
        TN = length(D_imposter) - FP;

        TPR(db_count) = TP/(TP+FN);
        FPR(db_count) = FP/(FP+TN);
        db_count = db_count + 1;
    end

    samplespace = zeros(1,11);
    for logidx = -5:0.5:0
        [Y, I] = min(abs(FPR-10^logidx));
        samplespace(round(logidx*2+11)) = I(1);
    end

    %% EER
    Distance = ones(1, size(FPR,2));
    FNR = 1-TPR;
    for i=1:size(FPR,2)
        if(FPR(i)~=1 && FNR(i)~=1)
            Distance(i) = abs(FPR(i)-FNR(i));
        end
    end
    [temp idx] = min(Distance);
    EER = mean([FPR(idx) FNR(idx)])
    legend('Location','southeast')
    set(findall(gcf,'-property','FontSize'),'FontSize',12)
    threshold = min(D_genuine)+0.0001*(idx-1);

    plot(FPR(samplespace),TPR(samplespace),'x-','MarkerSize',6,'DisplayName',[model{model_select} ' (EER: ' num2str(EER*100,'%.1f') '%, Acc.: ' num2str(Accuracy*100,'%.1f') '%)']);
    %plot(FPR(samplespace),TPR(samplespace),'x-','MarkerSize',6,'DisplayName',[model{model_select1} '+' model{model_select2} ' (EER: ' num2str(EER*100,'%.1f') '%, Acc.: ' num2str(Accuracy*100,'%.1f') '%)']);
    legend

    set(gca,'Xscale','log');
    set(gca,'xlim',[10^-4.5, 10^0]); %min(nonzeros(FPR))+0.00001
    title('ROC')
    xlabel('False Acceptance Rate')
    ylabel('Genuine Acceptance Rate')

    %Extend
    %{
    hold on
    plot(FPR,TPR,'m-*','DisplayName',['Ours-NVD: EER = ' num2str(EER)]);
    legend('off')
    legend('show')
    %}

    %% print and save
    print([folder_path 'roc' outname],'-dtiffn')
    save([folder_path 'roc' outname '.mat'],'FPR','TPR','EER','Accuracy','threshold')
    hold all
    
    
    %% CMC
    score_matrix = -Pred';
    rpmf = zeros(1,No_subject); %initialize RPMF
    rpmf_total_mass = 0;
    for j=1:size(score_matrix,2) %for all images
        vector = score_matrix(:,j);
        [~,I] = sort(vector); %'descend'
        rank = find(I==ceil(j/No_set));
        rpmf(rank) = rpmf(rank) + 1;
        rpmf_total_mass = rpmf_total_mass + 1;
    end

    %RPMF normalization
    for r = 1:size(rpmf,2)
        rpmf(r) = rpmf(r)/rpmf_total_mass;
    end

    %CMC generation
    cmc = zeros(1,No_subject);
    for k = 1:size(rpmf,2)
        sum_rpmf = 0;
        for r = 1:k
            sum_rpmf = sum_rpmf + rpmf(r);
        end
        cmc(k) = sum_rpmf;
    end

    save([folder_path 'cmc' outname '.mat'],'cmc','rpmf')
    
end



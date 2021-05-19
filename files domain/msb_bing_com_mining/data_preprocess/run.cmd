REM ===================================================
REM QCSQueryLabelWithLES Aether Wrapper module (-dl option)
REM   Run QCSQueryLabelWithLES.exe with -dl (multiple domains)
REM ===================================================

echo %1
echo %2



REM FOR /L %%G IN (1,1,2708) DO (
REM FOR /L %%G IN (1,1,2) DO (
FOR /L %%G IN (%1,1,%2) DO (
    E:\qasquerylevel\app\queryclassificationprocessing\QCSQueryLabelWithLES.exe -c E:\CoreScienceDataStaging\b2_models\files\ --variant files_enus_mv5 --encoding utf-8 --legacyAllowUnusedParameters -i E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output\MSB_combine_%%G.tsv -dl files --queryInColumn 1 -o E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine.txt
    copy E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine.txt.files.maxent.output.txt E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine_%%G.maxent.output.txt
    copy E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine.txt.files.multiclass.output.txt E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine_%%G.multiclass.output.txt
)
del E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine.txt.files*.txt
del E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine.txt.files*.xml

echo ======== [Run] =========
REM echo QCSQueryLabel\QCSQueryLabelWithLES.exe -c Model --variant %TargetQCSConfig% %ClientIdOption% -dl %DomainList% -i query.txt --verbose
REM QCSQueryLabel\QCSQueryLabelWithLES.exe -c Model --variant %TargetQCSConfig% %ClientIdOption% -i query.txt --verbose --legacyAllowUnusedParameters --dumpFormat %DumpFormat% %QueryInColumnOption% %ExternalInputInColumnOption% %HeaderInInputOption% --outputFullLine --avoidProcessingTermination


REM E:\qasquerylevel\app\queryclassificationprocessing\QCSQueryLabelWithLES.exe -c E:\CoreScienceDataStaging\b2_models\files\ --variant files_enus_mv5 --encoding utf-8 --legacyAllowUnusedParameters -i E:\CoreScienceDataStaging\datasets\TeamsVoiceSkills\Test\Mustpass_TVS_Calendar_Golden_singleturn_for_dnn_output.tsv -dl files --queryInColumn 1 -o E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\output.txt
REM E:\qasquerylevel\app\queryclassificationprocessing\QCSQueryLabelWithLES.exe -c E:\CoreScienceDataStaging\b2_models\files\ --variant files_enus_mv5 --encoding utf-8 --legacyAllowUnusedParameters -i E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output\MSB_combine_0.tsv -dl files --queryInColumn 1 -o E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine_0.txt


REM copy query.txt.analyzedqueries.output.txt %OutputFile%


    REM E:\qasquerylevel\app\queryclassificationprocessing\QCSQueryLabelWithLES.exe -c E:\CoreScienceDataStaging\b2_models\files\ --variant files_enus_mv5 --encoding utf-8 --loadingThreadsNum 1--legacyAllowUnusedParameters -i E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output\MSB_combine_%%G.tsv -dl files --queryInColumn 1 -o E:\fileAnswer_data_synthesis\CMF_training\msb_bing_com_mining\data_preprocess\output_with_qas_result\MSB_combine.txt

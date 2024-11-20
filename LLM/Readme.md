notes: <br>
- all_reviews.csv not included due to file size
- 1_df.pkl not included due to file size
- finetuned weights and tokenizers not included in repo.<br> Onedrive link to trained models accessible till mid 2025.
---
Go about using this LLM section in the following order:
1. Use 1_summarizer.py to create output folder of firm review summaries.<br> Config file for the summarizer is 1_config.ini<br><br>
2. Use 1_LLM_create_firm_summaries.ipynb to<br>
&nbsp; - create 1_LLM_finetuning_training_dataset.pkl file (section 2)<br>
&nbsp; - create 1_summary_reviews.csv (section 4)<br><br>
3. Use 2_finetuning_review_sentiment_analyzer.ipynb to<br>
&nbsp; - create 2_data_dict.pkl (section 3)<br>
&nbsp; - create 2_base_eval.json (section 4)<br>
&nbsp; - create 2_summ_text.csv (section 8)<br><br>
4. use finetune/3_run_finetune.bat to create finetuned weights and tokenizers in trained_weights folder<br>
&nbsp; - max steps in the 3_finetune.py file has been set to 1500 to limit training time, resulting in training loss of about 1.1+ on average.<br>
&nbsp; - if max steps is set to -1 (unlimited), training loss can be reduced to about 0.8+ on average.<br>
&nbsp; - if running in a Unix-based environment, create a shell script with function of the batch file<br><br>
5. use finetune/3_run_test_eval.bat to create finetune/3_<category>_test_predictions.csv evaluation metrics<br>
&nbsp; - if running in a Unix-based environment, create a shell script with function of the batch file<br><br>
6. use finetune/4_run_summ_eval.bat to create finetune/4_summ_ratings_<category>.csv evaluation metrics<br>
&nbsp; - this script predicts ratings for all firm summary reviews and stores predictions to the csv files<br>
&nbsp; - if running in a Unix-based environment, create a shell script with function of the batch file<br>

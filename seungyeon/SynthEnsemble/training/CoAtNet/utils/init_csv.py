from datetime import datetime
import pandas as pd
import os

class CSV_Record():
    def __init__(self, model_name):
        columns = [	"Time",
                    "Accuracy",
                    "F1_macro",
                    "Roc_Auc_macro"]
        
        self.columns = columns

        result_save_path = 'save'
        
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)

        init_time = datetime.now()
        init_time = init_time.strftime('%m%d_%H%M')

        init_df = pd.DataFrame(columns=columns)
        self.csv_name = f'{result_save_path}/{init_time}_{model_name}_output.csv'
        init_df.to_csv(self.csv_name, index=False)


    def record_csv(self, accuracy, f1_macro, roc_auc_macro):
        now = datetime.now() 
        csv_record_time = now.strftime('%Y%m%d_%H%M%S')
        csv_accuracy = f"{accuracy:.4f}"
        csv_f1_macro = f"{f1_macro:.4f}"
        csv_roc_auc_macro = f"{roc_auc_macro:.4f}"

        csv_data = [csv_record_time,
                    csv_accuracy,
                    csv_f1_macro,
                    csv_roc_auc_macro]

        df = pd.DataFrame([csv_data], columns=self.columns)
        df.to_csv(self.csv_name, mode='a', header=False, index=False)
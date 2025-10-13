import sys
import os

import pandas as pd
import numpy as np
import subprocess

import json
from utils.parser import parse_json_block_safely
from utils.eda import analyze_missing_values, analyze_data_distribution


def run_streamlit_matrix():
    """Streamlit matrix 앱을 실행합니다."""
    try:
        # 현재 디렉토리를 프로젝트 루트로 설정
        project_root = os.path.dirname(os.path.abspath(__file__))
        os.chdir(project_root)
        
        # streamlit run matrix/matrix.py 명령어 실행
        cmd = [sys.executable, "streamlit run matrix/matrix.py"]
        
        print("Streamlit Matrix 앱을 실행합니다...")
        print(f"실행 명령어: {' '.join(cmd)}")
        print("브라우저에서 http://localhost:8501 을 열어 앱을 확인하세요.")
        print("앱을 종료하려면 Ctrl+C를 누르세요.")
        print("-" * 50)
        
        # subprocess로 streamlit 실행
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n앱이 사용자에 의해 종료되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit 실행 중 오류가 발생했습니다: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)

def eval_result_analysis():
    """평가 결과 분석"""
    try:
        df_eval_result = pd.read_csv("dataset/4.eval_result_data/702-samples-7-target-llms-merged-2_result_fin.csv")
        #print(df_eval_result.head(1).to_dict())

        df_eval_result_matrix = pd.DataFrame(
            {
                #"id": df_eval_result['id'].tolist(),
                "id": "",
                #"answerer_llm_alias": df_eval_result['answerer_llm_alias'].tolist(),
                "answerer_llm_alias": df_eval_result['model_name'].tolist(),
                "label_task": df_eval_result['label_task'].tolist(),
                "label_domain": df_eval_result['label_domain'].tolist(),
                "label_level": df_eval_result['label_level'].tolist(),
                "eval_result": df_eval_result['eval_result'].tolist(),
                #"sp_max_tokens": df_eval_result['sp_max_tokens'].tolist()
                "sp_max_tokens": ""
            }
        )
        print(len(df_eval_result_matrix))
        print(df_eval_result_matrix.describe)
        # # 결측값 분석
        missing_values_df = analyze_missing_values(df_eval_result_matrix)
        print(missing_values_df)
        print(missing_values_df[missing_values_df['Missing Count'] > 0])

        # 데이터 분포 상세 정보
        df_info = analyze_data_distribution(df_eval_result_matrix, column_name="label_task")
        print(df_info)

        j=0
        y=0
        for i, row in df_eval_result_matrix.iterrows():
            eval_result = row.eval_result
            if pd.isna(eval_result):
                dict_evaluation = {}
            else:
                try:
                    #if i >= 2012 and i<=2015: print(i, row.eval_result)
                    dict_evaluation = parse_json_block_safely(eval_result)
                    #if i >= 621 and i <= 624 or i == 2012: print(i, dict_evaluation)
                    #if i == 622: print(i, dict_evaluation)
                except Exception as e:
                    #list_error_index.append(index)
                    # j += 1
                    # print(j)
                    # print(eval_result)
                    # print(e)

                    #print(dict_evaluation)       
                    #dict_evaluation = {}     
                    continue


    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    eval_result_analysis()
    #run_streamlit_matrix()

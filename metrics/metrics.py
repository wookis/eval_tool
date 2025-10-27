
import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eda import ChartView
from utils.parser import load_dataset, dataset_clean
from utils.parser import parse_json_block_safely
from utils.eda import EDA
import itertools
from tqdm import tqdm
from utils.eda import filter_efficient_llm, get_efficient_tld_by_step_by_step



# 페이지 설정
st.set_page_config(layout="wide")



class DataLoader:
    """데이터 로딩을 담당하는 클래스"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.result_files = self._get_result_files()
    
    def _get_result_files(self) -> List[str]:
        """결과 파일 목록을 가져옵니다."""
        if not os.path.exists(self.dataset_dir):
            return []
        
        files = [f for f in os.listdir(self.dataset_dir) 
                if f.endswith('.json') or f.endswith('.csv')]
                #if f.startswith('702-') and f.endswith('.json')]
        
        #file 최근 생성된 순으로 정렬
        #files.sort(key=lambda x: os.path.getmtime(os.path.join(self.dataset_dir, x)), reverse=True)
        
        return files

def main():
    """메인 함수"""
    # 데이터 로더 초기화
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset/5.matrix_data')
    data_loader = DataLoader(dataset_dir)
    
    if not data_loader.result_files:
        st.error('평가 결과 파일이 없습니다.')
        st.stop()
    
    # 파일 선택
    result_file = st.selectbox('결과 파일 선택', data_loader.result_files)
    result_file = "702-sample-evaluaution-9-target-llms-cleansed.pkl"
    result_file = "250911-102500-answered_df_result9_fin.csv"
    
    
    # 데이터 로드
    try:
        df = load_dataset('dataset/5.matrix_data/'+result_file)
        #print(df.columns)
    except Exception as e:
        st.error(f'데이터 로드 중 오류가 발생했습니다: {e}')
        st.stop()
        exit()
###########################################################################################

    #평가대상 데이터 분포 확인
    ChartView.DataEDA(df)
    
    #평가결과 데이터 클렌징    
    df_filtered = dataset_clean(df)
    
    #모델별 평가 결과 분포 차트 그리기
    ChartView.ModelEvalDist(df_filtered)

    #모델별 스코어 분포 차트 그리기
    ChartView.ModelScoreDist(df_filtered)

    #모델별 Cost 분포 차트 그리기
    ChartView.ModelCostDist(df_filtered)

    #tldc 맵핑 데이터 만들기
    efficient_tld, pdf_merged_sample_llm_name_as_tld_table, df_efficient_tld_num_llm_as_tld_table, dict_llm_name2df = EDA.TLDMapData(df_filtered)

    #tldc 맵핑 히트맵 그리기
    ChartView.TLDCMappingHeatmap(efficient_tld, pdf_merged_sample_llm_name_as_tld_table, df_efficient_tld_num_llm_as_tld_table)

        #tldc 맵핑 히트맵 그리기
    ChartView.QualityCostGraph(efficient_tld, dict_llm_name2df)


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# 페이지 설정
st.set_page_config(layout="wide")

@dataclass
class MatrixConfig:
    """매트릭스 설정을 위한 데이터 클래스"""
    domains: List[str]
    tasks: List[str]
    levels: List[str]
    tasks_with_all_levels: List[str]
    model_colors: Dict[str, str]
    
    @classmethod
    def create_default(cls) -> 'MatrixConfig':
        """기본 설정으로 MatrixConfig 생성"""
        domains = [f'D{i}' for i in range(1, 11)]
        tasks = [f'T{i}' for i in range(1, 12)]
        levels = [f'L{i}' for i in range(1, 3)]
        
        # 각 task별로 L1, L2 세 개의 level을 모두 포함하는 세로축 생성
        tasks_with_all_levels = []
        for task in tasks:
            for level in levels:
                if level == 'L1':
                    tasks_with_all_levels.append(f"{task} {level}")
                tasks_with_all_levels.append(f"{level}")
        
        # 모델별 컬러맵 배열 정의
        model_colors = ["Reds", "Blues", "Greens", "Oranges"]
        
        return cls(domains, tasks, levels, tasks_with_all_levels, {})

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
        return sorted(files)
    
    def load_data(self, filename: str) -> List[Dict[str, Any]]:
        """JSON 파일에서 데이터를 로드합니다."""
        filepath = os.path.join(self.dataset_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
        
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    
    def get_model_names(self, data: List[Dict[str, Any]]) -> List[str]:
        """데이터에서 모델명 목록을 추출합니다."""
        return sorted(list(set(r['model'] for r in data)))

class MatrixProcessor:
    """매트릭스 데이터 처리를 담당하는 클래스"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
    
    def create_matrix(self, data: List[Dict[str, Any]], model: str) -> pd.DataFrame:
        """특정 모델의 매트릭스를 생성합니다."""
        matrix = np.full((len(self.config.tasks_with_all_levels), 
                         len(self.config.domains)), np.nan)
        
        for r in data:
            if r['model'] != model:
                continue
            
            d = r.get('domain', '')
            t = r.get('task', '')
            l = r.get('level', '')
            
            try:
                d_idx = self.config.domains.index(d)
                task_level_key = f"{t} {l}"
                t_idx = self.config.tasks_with_all_levels.index(task_level_key)
                score = r['score'] / 5.3
                
                matrix[t_idx, d_idx] = score
            except ValueError:
                continue
        
        index = pd.Index(self.config.tasks_with_all_levels)
        columns = pd.Index(self.config.domains)
        return pd.DataFrame(matrix, index=index, columns=columns)
    
    def create_model_matrices(self, data: List[Dict[str, Any]], 
                            model_names: List[str]) -> Tuple[Dict[str, np.ndarray], 
                                                           Dict[str, np.ndarray]]:
        """모든 모델의 점수와 토큰 매트릭스를 생성합니다."""
        model_matrices = {m: np.full((len(self.config.tasks_with_all_levels), 
                                    len(self.config.domains)), np.nan) 
                         for m in model_names}
        token_matrices = {m: np.full((len(self.config.tasks_with_all_levels), 
                                    len(self.config.domains)), np.nan) 
                         for m in model_names}
        
        for r in data:
            m = r['model']
            d = r.get('domain', '')
            t = r.get('task', '')
            l = r.get('level', '')
            tokens = r.get('token', r.get('tokens', 0))
            
            try:
                d_idx = self.config.domains.index(d)
                task_level_key = f"{t} {l}"
                t_idx = self.config.tasks_with_all_levels.index(task_level_key)
                model_matrices[m][t_idx, d_idx] = r['score'] / 5.3
                token_matrices[m][t_idx, d_idx] = tokens
            except ValueError:
                continue
        
        return model_matrices, token_matrices
    
    def calculate_best_values(self, model_matrices: Dict[str, np.ndarray], 
                            token_matrices: Dict[str, np.ndarray], 
                            model_names: List[str], 
                            weight_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """가중치 타입에 따라 최고 값을 계산합니다."""
        best_values = np.full((len(self.config.tasks_with_all_levels), 
                             len(self.config.domains)), np.nan)
        best_models = np.full((len(self.config.tasks_with_all_levels), 
                             len(self.config.domains)), '', dtype=object)
        
        for i in range(len(self.config.tasks_with_all_levels)):
            for j in range(len(self.config.domains)):
                cell_scores = []
                cell_tokens = []
                
                for m in model_names:
                    score = model_matrices[m][i, j]
                    token = token_matrices[m][i, j]
                    cell_scores.append(score)
                    cell_tokens.append(token)
                
                cell_scores = np.array(cell_scores)
                cell_tokens = np.array(cell_tokens)
                
                if weight_type == "품질":
                    if np.all(np.isnan(cell_scores)):
                        best_values[i, j] = np.nan
                        best_models[i, j] = ''
                    else:
                        idx = np.nanargmax(cell_scores)
                        best_values[i, j] = cell_scores[idx]
                        best_models[i, j] = model_names[idx]
                
                elif weight_type == "가격":
                    valid = (cell_tokens > 0)
                    if not np.any(valid):
                        best_values[i, j] = np.nan
                        best_models[i, j] = ''
                    else:
                        idx = np.argmin(cell_tokens[valid])
                        best_values[i, j] = cell_tokens[valid][idx]
                        best_models[i, j] = np.array(model_names)[valid][idx]
                
                elif weight_type == "가성비":
                    valid = (cell_tokens > 0)
                    ratio = np.zeros_like(cell_scores)
                    ratio[valid] = cell_scores[valid] / cell_tokens[valid] * 100
                    ratio[~valid] = np.nan
                    
                    if np.all(np.isnan(ratio)):
                        best_values[i, j] = np.nan
                        best_models[i, j] = ''
                    else:
                        idx = np.nanargmax(ratio)
                        best_values[i, j] = ratio[idx]
                        best_models[i, j] = model_names[idx]
        
        return best_values, best_models

class ChartRenderer:
    """차트 렌더링을 담당하는 클래스"""
    
    def __init__(self, config: MatrixConfig):
        self.config = config
    
    def render_single_model_heatmap(self, df: pd.DataFrame, model: str) -> None:
        """단일 모델의 히트맵을 렌더링합니다."""
        st.title(f"모델별 평가 결과: {model}")
        st.write("각 셀은 평균 점수(0~1, 소수점 3자리)입니다.")
        
        cmap = self.config.model_colors.get(model, "gray")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            df, 
            annot=True, 
            fmt=".3f", 
            cmap=cmap, 
            linewidths=1.5, 
            linecolor="#888888", 
            ax=ax, 
            vmin=0, 
            vmax=1,
            cbar_kws={"shrink": 0.8, "orientation": "vertical"},
            yticklabels=True,
            xticklabels=True,
            annot_kws={"size": 8}  # 셀 내 텍스트 폰트 사이즈
        )
        
        self._style_heatmap(ax, title_size=14, label_size=10, tick_size=9)
        st.pyplot(fig)
    
    def render_model_comparison(self, data: List[Dict[str, Any]], 
                              model_names: List[str], 
                              processor: MatrixProcessor) -> None:
        """모든 모델의 비교 히트맵을 렌더링합니다."""
        st.title("모델별 평가 결과 비교")
        
        cols = st.columns(len(model_names))
        for idx, model in enumerate(model_names):
            df = processor.create_matrix(data, model)
            
            with cols[idx]:
                st.subheader(model)
                fig, ax = plt.subplots(figsize=(6, 6))
                cmap = self.config.model_colors.get(model, "YlGnBu")
                
                sns.heatmap(
                    df,
                    annot=True,
                    fmt=".3f",
                    cmap=cmap,
                    linewidths=1.5,
                    linecolor="#888888",
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cbar_kws={"shrink": 0.8, "orientation": "vertical"},
                    yticklabels=True,
                    xticklabels=True,
                    annot_kws={"size": 6}  # 작은 차트의 셀 내 텍스트 폰트 사이즈
                )
                
                self._style_heatmap(ax, title_size=10, label_size=8, tick_size=7)
                st.pyplot(fig)
    
    def render_best_model_heatmap(self, best_values: np.ndarray, 
                                best_models: np.ndarray, 
                                weight_type: str) -> None:
        """최고 모델 히트맵을 렌더링합니다."""
        st.title(f"라우팅 품질 Matrix (최고 {weight_type} 모델)")
        
        if weight_type == "품질":
            st.write("각 셀에는 최고 점수를 받은 모델명과 점수가 표시됩니다.")
        elif weight_type == "가격":
            st.write("각 셀에는 비용이 가장 적은 모델명과 토큰 수가 표시됩니다.")
        elif weight_type == "가성비":
            st.write("각 셀에는 가성비(점수/비용)가 가장 높은 모델명과 그 값을 표시합니다.")
        
        # annotation 생성
        annot = np.empty(best_values.shape, dtype=object)
        for i in range(len(self.config.tasks_with_all_levels)):
            for j in range(len(self.config.domains)):
                if best_models[i, j]:
                    annot[i, j] = f"{best_models[i, j]}\n{best_values[i, j]:.3f}"
                else:
                    annot[i, j] = ""
        
        df_best = pd.DataFrame(best_values, 
                             index=pd.Index(self.config.tasks_with_all_levels), 
                             columns=pd.Index(self.config.domains))
        
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            df_best,
            annot=annot,
            fmt="",
            cmap="Reds",
            linewidths=1.5,
            linecolor="#888888",
            ax=ax2,
            vmin=0,
            vmax=1 if weight_type != "가격" else None,
            cbar_kws={"shrink": 0.8, "orientation": "vertical"},
            yticklabels=True,
            xticklabels=True,
            annot_kws={"size": 7}  # 모델명과 점수가 함께 표시되는 셀의 폰트 사이즈
        )
        
        self._style_heatmap(ax2, title_size=14, label_size=10, tick_size=9)
        st.pyplot(fig2)
    
    def _style_heatmap(self, ax: plt.Axes, title_size: int = 12, label_size: int = 10, tick_size: int = 9) -> None:
        """히트맵 스타일을 적용합니다."""
        # 외곽선 추가
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('#888888')
            spine.set_linewidth(2)
        
        # 가로 범주(컬럼명)를 위에 표시
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        # Set tick positions first, then rotate
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # 폰트 사이즈 설정
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        
        # 축 라벨 폰트 사이즈 설정
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)
        
        # 제목 폰트 사이즈 설정
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=title_size)

def main():
    """메인 함수"""
    # 설정 초기화
    config = MatrixConfig.create_default()
    
    # 데이터 로더 초기화
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset/5.matrix_data')
    data_loader = DataLoader(dataset_dir)
    
    if not data_loader.result_files:
        st.error('평가 결과 파일이 없습니다.')
        st.stop()
    
    # 파일 선택
    result_file = st.selectbox('결과 파일 선택', data_loader.result_files)
    
    # 데이터 로드
    try:
        data = data_loader.load_data(result_file)
        model_names = data_loader.get_model_names(data)
    except Exception as e:
        st.error(f'데이터 로드 중 오류가 발생했습니다: {e}')
        st.stop()
    
    # 모델별 컬러맵 설정
    model_colors = ["Reds", "Blues", "Greens", "Oranges"]
    for i, model in enumerate(model_names):
        if i < len(model_colors):
            config.model_colors[model] = model_colors[i]
        else:
            config.model_colors[model] = "gray"
    
    # 프로세서와 렌더러 초기화
    processor = MatrixProcessor(config)
    renderer = ChartRenderer(config)
    
    # 모델 선택
    model = st.selectbox("모델 선택", model_names)
    
    # 단일 모델 히트맵 렌더링
    df = processor.create_matrix(data, model)
    renderer.render_single_model_heatmap(df, model)
    
    # 모델 비교 히트맵 렌더링
    renderer.render_model_comparison(data, model_names, processor)
    
    # 평가 비중 선택
    weight_type = st.selectbox("평가 비중", ["품질", "가격", "가성비"])
    
    # 최고 모델 히트맵 렌더링
    model_matrices, token_matrices = processor.create_model_matrices(data, model_names)
    best_values, best_models = processor.calculate_best_values(
        model_matrices, token_matrices, model_names, weight_type
    )
    renderer.render_best_model_heatmap(best_values, best_models, weight_type)

if __name__ == "__main__":
    main()

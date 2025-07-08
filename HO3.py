# ==============================================================================
# IMPORT LIBRARIES (การนำเข้าไลบรารีที่จำเป็น)
# ==============================================================================
import streamlit as st
import io
import urllib.parse
import json
import re
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import google.generativeai as genai


# ==============================================================================
# PAGE CONFIGURATION (การตั้งค่าหน้าเว็บเบื้องต้น)
# ==============================================================================
LOGO_URL = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/refs/heads/main/logo1.png"
st.set_page_config(page_title="HOIA-RR", page_icon=LOGO_URL, layout="wide")
st.markdown("""
<style>
/* CSS to style the text area inside the chat input */
[data-testid="stChatInput"] textarea {
    min-height: 80px;
    height: 100px;
    resize: vertical;
    background-color: transparent;
    border: none;
}
</style>
""", unsafe_allow_html=True)
# --- START: CSS Styles ---
st.markdown("""
<style>
    /* CSS Styles for Print View Only */
    @media print {
        /* === FIX FOR st.columns TO STAY IN A ROW === */
        div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: repeat(5, 1fr) !important;
            gap: 1.2rem !important;
        }
        /* === FIX FOR TABLES & DATAFRAMES SPANNING PAGES === */
        .stDataFrame, .stTable { break-inside: avoid; page-break-inside: avoid; }
        thead, tr, th, td { break-inside: avoid !important; page-break-inside: avoid !important; }
        h1, h2, h3, h4, h5 { page-break-after: avoid; }
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* CSS ทั่วไปสำหรับ Header */
.custom-header { font-size: 20px; font-weight: bold; margin-top: 0px !important; padding-top: 0px !important; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] {
    border: 1px solid #ddd; padding: 0.75rem; border-radius: 0.5rem; height: 100%;
    display: flex; flex-direction: column; justify-content: center;}
div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] { background-color: #e6fffa; border-color: #b2f5ea; }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] { background-color: #fff3e0; border-color: #ffe0b2; }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] { background-color: #fce4ec; border-color: #f8bbd0; }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] { background-color: #e3f2fd; border-color: #bbdefb; }
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div,
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stHorizontalBlock"] > div div[data-testid="stMetric"] [data-testid="stMetricDelta"]
{ color: #262730 !important; }
div[data-testid="stMetric"] [data-testid="stMetricLabel"] > div { font-size: 0.8rem !important; line-height: 1.2 !important; white-space: normal !important; overflow-wrap: break-word !important; word-break: break-word; display: block !important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
div[data-testid="stHorizontalBlock"] > div .stExpander { border: none !important; box-shadow: none !important; padding: 0 !important; margin-top: 0.5rem;}
div[data-testid="stHorizontalBlock"] > div .stExpander header { padding: 0.25rem 0.5rem !important; font-size: 0.75rem !important; border-radius: 0.25rem;}
div[data-testid="stHorizontalBlock"] > div .stExpander div[data-testid="stExpanderDetails"] { max-height: 200px; overflow-y: auto;}
.stDataFrame table td, .stDataFrame table th { color: black !important; font-size: 0.9rem !important; }
.stDataFrame table th { font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)


# --- END: CSS Styles ---
# ==============================================================================
# ANALYSIS FUNCTIONS (ฟังก์ชันสำหรับการวิเคราะห์)
# ==============================================================================
def image_to_base64(img_path_str):
    img_path = Path(img_path_str)
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None
def extract_severity_level(prompt_text):
    prompt_upper = prompt_text.upper()
    match1 = re.search(r'(?:ระดับ|LEVEL)\s*([A-I])\b', prompt_upper)
    if match1:
        return (match1.group(1), 'clinical')
    match2 = re.search(r'(?:ระดับ|LEVEL)\s*([1-5])\b', prompt_upper)
    if match2:
        return (match2.group(1), 'general')
    match3 = re.search(r'\b([A-I])\b', prompt_upper)
    if match3:
        return (match3.group(1), 'clinical')
    match4 = re.search(r'\b([1-5])\b', prompt_upper)
    if match4:
        return (match4.group(1), 'general')
    return (None, None)


def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl', keep_default_na=False)
        return df
    except Exception as e:
        st.session_state.upload_error_message = f"เกิดข้อผิดพลาดในการอ่านไฟล์ Excel: {e}"
        return pd.DataFrame()


def calculate_autocorrelation(series: pd.Series):
    if len(series) < 2: return 0.0
    if series.std() == 0: return 1.0
    try:
        series_mean = series.mean()
        c1 = np.sum((series.iloc[1:].values - series_mean) * (series.iloc[:-1].values - series_mean))
        c0 = np.sum((series.values - series_mean) ** 2)
        return c1 / c0 if c0 != 0 else 0.0
    except Exception:
        return 0.0


@st.cache_data
def calculate_risk_level_trend(_df: pd.DataFrame):
    risk_level_map_to_score = {
        "51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16, "42": 17, "43": 18, "44": 19, "45": 20,
        "31": 11, "32": 12, "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8, "24": 9, "25": 10,
        "11": 1, "12": 2, "13": 3, "14": 4, "15": 5
    }
    if _df.empty or 'รหัส' not in _df.columns or 'Occurrence Date' not in _df.columns or 'Risk Level' not in _df.columns:
        return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Occurrence Date', 'Risk Level']].copy()
    analysis_df['Ordinal_Risk_Score'] = analysis_df['Risk Level'].astype(str).map(risk_level_map_to_score)
    analysis_df.dropna(subset=['Ordinal_Risk_Score'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    analysis_df['YearMonth'] = pd.to_datetime(analysis_df['Occurrence Date'], errors='coerce').dt.to_period('M')
    results = []
    all_incident_codes = analysis_df['รหัส'].unique()
    for code in all_incident_codes:
        incident_subset = analysis_df[analysis_df['รหัส'] == code]
        monthly_risk_score = incident_subset.groupby('YearMonth')['Ordinal_Risk_Score'].mean().reset_index()
        slope = 0
        if len(monthly_risk_score) >= 3:
            X = np.arange(len(monthly_risk_score)).reshape(-1, 1)
            y = monthly_risk_score['Ordinal_Risk_Score']
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
        average_risk_score = incident_subset['Ordinal_Risk_Score'].mean()
        results.append({
            'รหัส': code,
            'Average_Risk_Score': average_risk_score,
            'Data_Points_Months': len(monthly_risk_score),
            'Risk_Level_Trend_Slope': slope
        })
    if not results: return pd.DataFrame()
    final_df = pd.DataFrame(results)
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(final_df, incident_names, on='รหัส', how='left')
    final_df = final_df.sort_values(by='Risk_Level_Trend_Slope', ascending=False)
    return final_df


@st.cache_data
def calculate_persistence_risk_score(_df: pd.DataFrame, total_months: int):
    risk_level_map_to_score = {
        "51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16, "42": 17, "43": 18, "44": 19, "45": 20,
        "31": 11, "32": 12, "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8, "24": 9, "25": 10,
        "11": 1, "12": 2, "13": 3, "14": 4, "15": 5
    }
    if _df.empty or 'รหัส' not in _df.columns or 'Risk Level' not in _df.columns:
        return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Risk Level']].copy()
    analysis_df['Ordinal_Risk_Score'] = analysis_df['Risk Level'].astype(str).map(risk_level_map_to_score)
    analysis_df.dropna(subset=['Ordinal_Risk_Score'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    persistence_metrics = analysis_df.groupby('รหัส').agg(
        Average_Ordinal_Risk_Score=('Ordinal_Risk_Score', 'mean'),
        Total_Occurrences=('รหัส', 'size')
    ).reset_index()
    if total_months == 0: total_months = 1
    persistence_metrics['Incident_Rate_Per_Month'] = persistence_metrics['Total_Occurrences'] / total_months
    max_rate = persistence_metrics['Incident_Rate_Per_Month'].max()
    if max_rate == 0: max_rate = 1
    persistence_metrics['Frequency_Score'] = persistence_metrics['Incident_Rate_Per_Month'] / max_rate
    persistence_metrics['Avg_Severity_Score'] = persistence_metrics['Average_Ordinal_Risk_Score'] / 25.0
    persistence_metrics['Persistence_Risk_Score'] = persistence_metrics['Frequency_Score'] + persistence_metrics[
        'Avg_Severity_Score']
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(persistence_metrics, incident_names, on='รหัส', how='left')
    final_df = final_df.sort_values(by='Persistence_Risk_Score', ascending=False)
    return final_df


@st.cache_data
def calculate_frequency_trend_poisson(_df: pd.DataFrame):
    if _df.empty or 'รหัส' not in _df.columns or 'Occurrence Date' not in _df.columns:
        return pd.DataFrame()
    analysis_df = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Occurrence Date']].copy()
    analysis_df.dropna(subset=['Occurrence Date'], inplace=True)
    if analysis_df.empty: return pd.DataFrame()
    analysis_df['YearMonth'] = pd.to_datetime(analysis_df['Occurrence Date']).dt.to_period('M')
    full_date_range = pd.period_range(start=analysis_df['YearMonth'].min(), end=analysis_df['YearMonth'].max(),
                                      freq='M')
    results = []
    all_incident_codes = analysis_df['รหัส'].unique()
    for code in all_incident_codes:
        incident_subset = analysis_df[analysis_df['รหัส'] == code]
        MIN_OCCURRENCES = 3
        if len(incident_subset) < MIN_OCCURRENCES:
            continue
        monthly_counts = incident_subset.groupby('YearMonth').size().reindex(full_date_range, fill_value=0)
        if len(monthly_counts) < 2:
            continue
        y = monthly_counts.values
        X = np.arange(len(monthly_counts))
        X = sm.add_constant(X)
        try:
            model = sm.Poisson(y, X).fit(disp=0)
            time_coefficient = model.params[1]
            results.append({
                'รหัส': code,
                'Poisson_Trend_Slope': time_coefficient,
                'Total_Occurrences': y.sum(),
                'Months_Observed': len(y)
            })
        except Exception:
            continue
    if not results: return pd.DataFrame()
    final_df = pd.DataFrame(results)
    incident_names = _df[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง']].drop_duplicates()
    final_df = pd.merge(final_df, incident_names, on='รหัส', how='left')
    final_df = final_df.sort_values(by='Poisson_Trend_Slope', ascending=False)
    return final_df


def create_poisson_trend_plot(df, selected_code_for_plot, display_df):
    full_date_range_for_plot = pd.period_range(
        start=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').min(),
        end=pd.to_datetime(df['Occurrence Date']).dt.to_period('M').max(),
        freq='M'
    )
    subset_for_plot = df[df['รหัส'] == selected_code_for_plot].copy()
    subset_for_plot['YearMonth'] = pd.to_datetime(subset_for_plot['Occurrence Date']).dt.to_period('M')
    counts_for_plot = subset_for_plot.groupby('YearMonth').size().reindex(full_date_range_for_plot, fill_value=0)
    y_plot = counts_for_plot.values
    trend_line_for_plot = []
    try:
        X_plot_raw = np.arange(len(counts_for_plot))
        X_lin_reg = X_plot_raw.reshape(-1, 1)
        lin_reg_model = LinearRegression().fit(X_lin_reg, y_plot)
        trend_line_for_plot = lin_reg_model.predict(X_lin_reg)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณเส้นแนวโน้ม (Linear): {e}")
    fig_plot = go.Figure()
    fig_plot.add_trace(go.Bar(
        x=counts_for_plot.index.strftime('%Y-%m'),
        y=y_plot,
        name='จำนวนครั้งที่เกิดจริง',
        marker=dict(color='#AED6F1', cornerradius=8)
    ))
    if len(trend_line_for_plot) > 0:
        fig_plot.add_trace(go.Scatter(
            x=counts_for_plot.index.strftime('%Y-%m'),
            y=trend_line_for_plot,
            mode='lines',
            name='เส้นแนวโน้ม (Linear)',
            line=dict(color='red', width=2, dash='dash')
        ))
    fig_plot.update_layout(
        title=f'การกระจายตัวของอุบัติการณ์: {selected_code_for_plot}',
        xaxis_title='เดือน-ปี',
        yaxis_title='จำนวนครั้งที่เกิด',
        barmode='overlay',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)')
    )
    slope_val_for_annot = \
        display_df.loc[display_df['รหัส'] == selected_code_for_plot, 'ดัชนีแนวโน้มความถี่ (Slope)'].iloc[0]
    factor_val_for_annot = \
        display_df.loc[display_df['รหัส'] == selected_code_for_plot, 'อัตราเปลี่ยนแปลง (เท่า/เดือน)'].iloc[0]
    annot_text = (f"<b>Poisson Slope: {slope_val_for_annot:.4f}</b><br>"
                  f"อัตราเปลี่ยนแปลง: x{factor_val_for_annot:.2f} ต่อเดือน")
    fig_plot.add_annotation(
        x=0.5, y=0.98,
        xref="paper", yref="paper",
        text=annot_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        align="center",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="rgba(255, 255, 224, 0.7)"
    )
    return fig_plot


def create_goal_summary_table(data_df_goal, goal_category_name_param,
                              e_up_non_numeric_levels_param, e_up_numeric_levels_param=None,
                              is_org_safety_table=False):
    goal_category_name_param = str(goal_category_name_param).strip()
    if 'หมวด' not in data_df_goal.columns:
        return pd.DataFrame()
    df_filtered_by_goal_cat = data_df_goal[
        data_df_goal['หมวด'].astype(str).str.strip() == goal_category_name_param].copy()
    if df_filtered_by_goal_cat.empty: return pd.DataFrame()
    if 'Incident Type' not in df_filtered_by_goal_cat.columns or 'Impact' not in df_filtered_by_goal_cat.columns: return pd.DataFrame()
    try:
        pvt_table_goal = pd.crosstab(df_filtered_by_goal_cat['Incident Type'],
                                     df_filtered_by_goal_cat['Impact'].astype(str).str.strip(), margins=True,
                                     margins_name='รวมทั้งหมด')
    except Exception:
        return pd.DataFrame()
    if 'รวมทั้งหมด' in pvt_table_goal.index: pvt_table_goal = pvt_table_goal.drop(index='รวมทั้งหมด')
    if pvt_table_goal.empty: return pd.DataFrame()
    if 'รวมทั้งหมด' not in pvt_table_goal.columns: pvt_table_goal['รวมทั้งหมด'] = pvt_table_goal.sum(axis=1)
    all_impact_columns_goal = [str(col).strip() for col in pvt_table_goal.columns if col != 'รวมทั้งหมด']
    e_up_non_numeric_levels_param_stripped = [str(level).strip() for level in e_up_non_numeric_levels_param]
    e_up_numeric_levels_param_stripped = [str(level).strip() for level in
                                          e_up_numeric_levels_param] if e_up_numeric_levels_param else []
    e_up_columns_goal = [col for col in all_impact_columns_goal if
                         col not in e_up_non_numeric_levels_param_stripped and (
                                 not e_up_numeric_levels_param_stripped or col not in e_up_numeric_levels_param_stripped)]
    report_data_goal = []
    for incident_type_goal, row_data_goal in pvt_table_goal.iterrows():
        total_e_up_count_goal = sum(row_data_goal[col] for col in e_up_columns_goal if
                                    col in row_data_goal.index and pd.notna(row_data_goal[col]))
        total_all_impacts_goal = row_data_goal['รวมทั้งหมด'] if 'รวมทั้งหมด' in row_data_goal and pd.notna(
            row_data_goal['รวมทั้งหมด']) else 0
        percent_e_up_goal = (total_e_up_count_goal / total_all_impacts_goal * 100) if total_all_impacts_goal > 0 else 0
        report_data_goal.append(
            {'Incident Type': incident_type_goal, 'รวม E-up': total_e_up_count_goal, 'ร้อยละ E-up': percent_e_up_goal})
    report_df_goal = pd.DataFrame(report_data_goal)
    if report_df_goal.empty:
        merged_report_table_goal = pvt_table_goal.reset_index()
        merged_report_table_goal['รวม E-up'] = 0
        merged_report_table_goal['ร้อยละ E-up'] = 0.0
    else:
        merged_report_table_goal = pd.merge(pvt_table_goal.reset_index(), report_df_goal, on='Incident Type',
                                            how='outer')
    if 'รวม E-up' not in merged_report_table_goal.columns:
        merged_report_table_goal['รวม E-up'] = 0
    else:
        merged_report_table_goal['รวม E-up'].fillna(0, inplace=True)
    if 'ร้อยละ E-up' not in merged_report_table_goal.columns:
        merged_report_table_goal['ร้อยละ E-up'] = 0.0
    else:
        merged_report_table_goal['ร้อยละ E-up'].fillna(0.0, inplace=True)
    cols_to_drop_from_display_goal = [col for col in e_up_non_numeric_levels_param_stripped if
                                      col in merged_report_table_goal.columns]
    if e_up_numeric_levels_param_stripped: cols_to_drop_from_display_goal.extend(
        [col for col in e_up_numeric_levels_param_stripped if col in merged_report_table_goal.columns])
    merged_report_table_goal = merged_report_table_goal.drop(columns=cols_to_drop_from_display_goal, errors='ignore')
    total_col_original_name, e_up_col_name, percent_e_up_col_name = 'รวมทั้งหมด', 'รวม E-up', 'ร้อยละ E-up'
    if is_org_safety_table:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'รวม 1-5', 'รวม 3-5', 'ร้อยละ 3-5'
        merged_report_table_goal.rename(
            columns={total_col_original_name: total_col_display_name, e_up_col_name: e_up_col_display_name,
                     percent_e_up_col_name: percent_e_up_display_name}, inplace=True, errors='ignore')
    else:
        total_col_display_name, e_up_col_display_name, percent_e_up_display_name = 'รวม A-I', e_up_col_name, percent_e_up_col_name
        merged_report_table_goal.rename(columns={total_col_original_name: total_col_display_name}, inplace=True,
                                        errors='ignore')
    merged_report_table_goal['Incident Type Name'] = merged_report_table_goal['Incident Type'].map(type_name).fillna(
        merged_report_table_goal['Incident Type'])
    final_columns_goal_order = ['Incident Type Name'] + [col for col in e_up_columns_goal if
                                                         col in merged_report_table_goal.columns] + [
                                   e_up_col_display_name, total_col_display_name, percent_e_up_display_name]
    final_columns_present_goal = [col for col in final_columns_goal_order if col in merged_report_table_goal.columns]
    merged_report_table_goal = merged_report_table_goal[final_columns_present_goal]
    if percent_e_up_display_name in merged_report_table_goal.columns and pd.api.types.is_numeric_dtype(
            merged_report_table_goal[percent_e_up_display_name]):
        try:
            merged_report_table_goal[percent_e_up_display_name] = merged_report_table_goal[
                percent_e_up_display_name].astype(float).map('{:.2f}%'.format)
        except ValueError:
            pass
    return merged_report_table_goal.set_index('Incident Type Name')


def create_severity_table(input_df, row_column_name, table_title, specific_row_order=None):
    if not isinstance(input_df,
                      pd.DataFrame) or input_df.empty or row_column_name not in input_df.columns or 'Impact Level' not in input_df.columns: return None
    temp_df = input_df.copy()
    temp_df['Impact Level'] = temp_df['Impact Level'].astype(str).str.strip().replace('N/A', 'ไม่ระบุ')
    if temp_df[row_column_name].dropna().empty: return None
    try:
        severity_crosstab = pd.crosstab(temp_df[row_column_name].astype(str).str.strip(), temp_df['Impact Level'])
    except Exception:
        return None
    impact_level_map_cols = {'1': 'A-B (1)', '2': 'C-D (2)', '3': 'E-F (3)', '4': 'G-H (4)', '5': 'I (5)',
                             'ไม่ระบุ': 'ไม่ระบุ LV'}
    desired_cols_ordered_keys = ['1', '2', '3', '4', '5', 'ไม่ระบุ']
    for col_key in desired_cols_ordered_keys:
        if col_key not in severity_crosstab.columns: severity_crosstab[col_key] = 0
    present_ordered_keys = [key for key in desired_cols_ordered_keys if key in severity_crosstab.columns]
    if not present_ordered_keys: return None
    severity_crosstab = severity_crosstab[present_ordered_keys].rename(columns=impact_level_map_cols)
    final_display_cols_renamed = [impact_level_map_cols[key] for key in present_ordered_keys if
                                  key in impact_level_map_cols]
    if not final_display_cols_renamed: return None
    severity_crosstab['รวมทุกระดับ'] = severity_crosstab[
        [col for col in final_display_cols_renamed if col in severity_crosstab.columns]].sum(axis=1)
    if specific_row_order:
        severity_crosstab = severity_crosstab.reindex([str(i) for i in specific_row_order]).fillna(0).astype(int)
    else:
        severity_crosstab = severity_crosstab[severity_crosstab['รวมทุกระดับ'] > 0]
    if severity_crosstab.empty: return None
    st.markdown(f"##### {table_title}")
    display_column_order_from_map = [impact_level_map_cols.get(key) for key in desired_cols_ordered_keys]
    display_column_order_present = [col for col in display_column_order_from_map if
                                    col in severity_crosstab.columns] + (
                                       ['รวมทุกระดับ'] if 'รวมทุกระดับ' in severity_crosstab.columns else [])
    st.dataframe(
        severity_crosstab[[col for col in display_column_order_present if col in severity_crosstab.columns]].astype(
            int), use_container_width=True)
    return severity_crosstab


def create_psg9_summary_table(input_df):
    if not isinstance(input_df,
                      pd.DataFrame) or 'หมวดหมู่มาตรฐานสำคัญ' not in input_df.columns or 'Impact' not in input_df.columns: return None
    psg9_placeholders = ["ไม่จัดอยู่ใน PSG9 Catalog", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)",
                         "ไม่สามารถระบุ (เช็คคอลัมน์ใน PSG9code.xlsx)",
                         "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ว่างเปล่า)",
                         "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว - rename)", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว - no col)",
                         "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด/ข้อมูลไม่ครบถ้วน)"]
    df_filtered = input_df[
        ~input_df['หมวดหมู่มาตรฐานสำคัญ'].isin(psg9_placeholders) & input_df['หมวดหมู่มาตรฐานสำคัญ'].notna()].copy()
    if df_filtered.empty: return pd.DataFrame()
    try:
        summary_table = pd.crosstab(df_filtered['หมวดหมู่มาตรฐานสำคัญ'], df_filtered['Impact'], margins=True,
                                    margins_name='รวม A-I')
    except Exception:
        return pd.DataFrame()
    if 'รวม A-I' in summary_table.index: summary_table = summary_table.drop(index='รวม A-I')
    if summary_table.empty: return pd.DataFrame()
    all_impacts, e_up_impacts = list('ABCDEFGHI'), list('EFGHI')
    for impact_col in all_impacts:
        if impact_col not in summary_table.columns: summary_table[impact_col] = 0
    if 'รวม A-I' not in summary_table.columns: summary_table['รวม A-I'] = summary_table[
        [col for col in all_impacts if col in summary_table.columns]].sum(axis=1)
    summary_table['รวม E-up'] = summary_table[[col for col in e_up_impacts if col in summary_table.columns]].sum(axis=1)
    summary_table['ร้อยละ E-up'] = (summary_table['รวม E-up'] / summary_table['รวม A-I'] * 100).fillna(0)
    psg_order = [PSG9_label_dict[i] for i in sorted(PSG9_label_dict.keys())]
    summary_table = summary_table.reindex(psg_order).fillna(0)
    display_cols_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'รวม E-up', 'รวม A-I', 'ร้อยละ E-up']
    final_table = summary_table[[col for col in display_cols_order if col in summary_table.columns]].copy()
    for col in final_table.columns:
        if col != 'ร้อยละ E-up': final_table[col] = final_table[col].astype(int)
    final_table['ร้อยละ E-up'] = final_table['ร้อยละ E-up'].map('{:.2f}%'.format)
    return final_table


def get_text_color_for_bg(hex_color):
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6: return '#000000'
        rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        return '#FFFFFF' if luminance < 0.5 else '#000000'
    except ValueError:
        return '#000000'


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = {}
if 'df_freq_for_display' not in st.session_state:
    st.session_state.df_freq_for_display = pd.DataFrame()
if 'upload_error_message' not in st.session_state:
    st.session_state.upload_error_message = None

analysis_options_list = [
    "แดชบอร์ดสรุปภาพรวม",
    "รายการ Sentinel Events",
    "Risk Matrix (Interactive)",
    "การแก้ไขอุบัติการณ์",
    "กราฟสรุปอุบัติการณ์ (รายมิติ)",
    "Sankey: ภาพรวม",
    "Sankey: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ",
    "Scatter Plot & Top 10",
    "สรุปอุบัติการณ์ตาม Safety Goals",
    "วิเคราะห์ตามหมวดหมู่และสถานะ",
    "Persistence Risk Index",
    "แนวโน้มความถี่ (Poisson Trend)",
    "บทสรุปสำหรับผู้บริหาร",
    "คุยกับ AI Assistant",
]

if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = analysis_options_list[0]
if 'show_incident_table' not in st.session_state:
    st.session_state.show_incident_table = False
if 'clicked_risk_impact' not in st.session_state:
    st.session_state.clicked_risk_impact = None
if 'clicked_risk_freq' not in st.session_state:
    st.session_state.clicked_risk_freq = None
# ==============================================================================
# STATIC DATA DEFINITIONS
# ==============================================================================
PSG9_FILE_PATH = "PSG9code.xlsx"
SENTINEL_FILE_PATH = "Sentinel2024.xlsx"
ALLCODE_FILE_PATH = "Code2024.xlsx"
GEMINI_PERSONA_INSTRUCTION = "ในฐานะ AI assistant, โปรดใช้สรรพนาม 'ผม' แทนตัวเอง และลงท้ายประโยคด้วย 'ครับ' เสมอ. ตอบคำถามนี้:"
psg9_r_codes_for_counting = set()
sentinel_composite_keys = set()
df2 = pd.DataFrame()
PSG9code_df_master = pd.DataFrame()
Sentinel2024_df = pd.DataFrame()
allcode2024_df = pd.DataFrame()

try:
    if Path(PSG9_FILE_PATH).is_file():
        PSG9code_df_master = pd.read_excel(PSG9_FILE_PATH)
        if 'รหัส' in PSG9code_df_master.columns:
            psg9_r_codes_for_counting = set(PSG9code_df_master['รหัส'].astype(str).str.strip().unique())
    if Path(SENTINEL_FILE_PATH).is_file():
        Sentinel2024_df = pd.read_excel(SENTINEL_FILE_PATH)
        if 'รหัส' in Sentinel2024_df.columns and 'Impact' in Sentinel2024_df.columns:
            Sentinel2024_df['รหัส'] = Sentinel2024_df['รหัส'].astype(str).str.strip()
            Sentinel2024_df['Impact'] = Sentinel2024_df['Impact'].astype(str).str.strip()
            Sentinel2024_df.dropna(subset=['รหัส', 'Impact'], inplace=True)
            sentinel_composite_keys = set((Sentinel2024_df['รหัส'] + '-' + Sentinel2024_df['Impact']).unique())
    if Path(ALLCODE_FILE_PATH).is_file():
        allcode2024_df = pd.read_excel(ALLCODE_FILE_PATH)
        if 'รหัส' in allcode2024_df.columns and all(
                col in allcode2024_df.columns for col in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]):
            df2 = allcode2024_df[["รหัส", "ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]].drop_duplicates().copy()
            df2['รหัส'] = df2['รหัส'].astype(str).str.strip()
except FileNotFoundError as e:
    st.session_state.upload_error_message = f"ไม่พบไฟล์นิยาม: {e}. กรุณาวางไฟล์ในโฟลเดอร์เดียวกับโปรแกรม"
except Exception as e:
    st.session_state.upload_error_message = f"เกิดปัญหาในการโหลดไฟล์นิยาม: {e}"
color_discrete_map = {'Critical': 'red', 'High': 'orange', 'Medium': 'yellow', 'Low': 'green', 'Undefined': '#D3D3D3'}
month_label = {1: '01 มกราคม', 2: '02 กุมภาพันธ์', 3: '03 มีนาคม', 4: '04 เมษายน', 5: '05 พฤษภาคม', 6: '06 มิถุนายน',
               7: '07 กรกฎาคม', 8: '08 สิงหาคม', 9: '09 กันยายน', 10: '10 ตุลาคม', 11: '11 พฤศจิกายน', 12: '12 ธันวาคม'}

severity_definitions_clinical = {
    'A': "A (เกิดที่นี่): เกิดเหตุการณ์ขึ้นแล้วจากตัวเองและค้นพบได้ด้วยตัวเองสามารถปรับแก้ไขได้ไม่ส่งผลกระทบถึงผู้อื่นและผู้ป่วยหรือบุคลากร",
    'B': "B (เกิดที่ไกล): เกิดเหตุการณ์/ความผิดพลาดขึ้นแล้วโดยส่งต่อเหตุการณ์/ความผิดพลาดนั้นไปที่ผู้อื่นแต่สามารถตรวจพบและแก้ไขได้ โดยยังไม่มีผลกระทบใดๆ ถึงผู้ป่วยหรือบุคลากร",
    'C': "C (เกิดกับใคร): เกิดเหตุการณ์/ความผิดพลาดขึ้นและมีผลกระทบถึงผู้ป่วยหรือบุคลากร แต่ไม่เกิดอันตรายหรือเสียหาย",
    'D': "D (ให้ระวัง): เกิดความผิดพลาดขึ้น มีผลกระทบถึงผู้ป่วยหรือบุคลากร ต้องให้การดูแลเฝ้าระวังเป็นพิเศษว่าจะไม่เป็นอันตราย",
    'E': "E (ต้องรักษา): เกิดความผิดพลาดขึ้น มีผลกระทบถึงผู้ป่วยหรือบุคลากร เกิดอันตรายชั่วคราวที่ต้องแก้ไข/รักษาเพิ่มมากขึ้น",
    'F': "F (เยียวยานาน): เกิดความผิดพลาดขึ้น มีผลกระทบที่ต้องใช้เวลาแก้ไขนานกว่าปกติหรือเกินกำหนด ผู้ป่วยหรือบุคลากร ต้องรักษา/นอนโรงพยาบาลนานขึ้น",
    'G': "G (ต้องพิการ): เกิดความผิดพลาดถึงผู้ป่วยหรือบุคลากร ทำให้เกิดความพิการถาวร หรือมีผลกระทบทำให้เสียชื่อเสียง/ความเชื่อถือและ/หรือมีการร้องเรียน",
    'H': "H (ต้องการปั๊ม): เกิดความผิดพลาดถึงผู้ป่วยหรือบุคลากร มีผลทำให้ต้องทำการช่วยชีวิต หรือกรณีทำให้เสียชื่อเสียงและ/หรือมีการเรียกร้องค่าเสียหายจากโรงพยาบาล",
    'I': "I (จำใจลา): เกิดความผิดพลาดถึงผู้ป่วยหรือบุคลากร เป็นสาเหตุทำให้เสียชีวิต เสียชื่อเสียงโดยมีการฟ้องร้องทางศาล/สื่อ"
}
severity_definitions_general = {
    '1': "ระดับ 1: เกิดความผิดพลาดขึ้นแต่ไม่มีผลกระทบต่อผลสำเร็จหรือวัตถุประสงค์ของการดำเนินงาน (*เกิดผลกระทบที่มีมูลค่าความเสียหาย 0 - 10,000 บาท)",
    '2': "ระดับ 2: เกิดความผิดพลาดขึ้นแล้ว โดยมีผลกระทบ (ที่ควบคุมได้) ต่อผลสำเร็จหรือวัตถุประสงค์ของการดำเนินงาน (*เกิดผลกระทบที่มีมูลค่าความเสียหาย 10,001 - 50,000 บาท)",
    '3': "ระดับ 3: เกิดความผิดพลาดขึ้นแล้ว และมีผลกระทบ (ที่ต้องทำการแก้ไข) ต่อผลสำเร็จหรือวัตถุประสงค์ของการดำเนินงาน (*เกิดผลกระทบที่มีมูลค่าความเสียหาย 50,001 - 250,000 บาท)",
    '4': "ระดับ 4: เกิดความผิดพลาดขึ้นแล้ว และทำให้การดำเนินงานไม่บรรลุผลสำเร็จตามเป้าหมาย (*เกิดผลกระทบที่มีมูลค่าความเสียหาย 250,001 – 10,000,000 บาท)",
    '5': "ระดับ 5: เกิดความผิดพลาดขึ้นแล้ว และมีผลให้การดำเนินงานไม่บรรลุผลสำเร็จตามเป้าหมาย ทำให้ภารกิจขององค์กรเสียหายอย่างร้ายแรง (*เกิดผลกระทบที่มีมูลค่าความเสียหายมากกว่า 10 ล้านบาท)"
}
PSG9_label_dict = {
    1: '01 ผ่าตัดผิดคน ผิดข้าง ผิดตำแหน่ง ผิดหัตถการ',
    2: '02 บุคลากรติดเชื้อจากการปฏิบัติหน้าที่',
    3: '03 การติดเชื้อสำคัญ (SSI, VAP,CAUTI, CLABSI)',
    4: '04 การเกิด Medication Error และ Adverse Drug Event',
    5: '05 การให้เลือดผิดคน ผิดหมู่ ผิดชนิด',
    6: '06 การระบุตัวผู้ป่วยผิดพลาด',
    7: '07 ความคลาดเคลื่อนในการวินิจฉัยโรค',
    8: '08 การรายงานผลการตรวจทางห้องปฏิบัติการ/พยาธิวิทยา คลาดเคลื่อน',
    9: '09 การคัดกรองที่ห้องฉุกเฉินคลาดเคลื่อน'
}
type_name = {'CPS': 'Safe Surgery', 'CPI': 'Infection Prevention and Control', 'CPM': 'Medication & Blood Safety',
             'CPP': 'Patient Care Process', 'CPL': 'Line, Tube & Catheter and Laboratory', 'CPE': 'Emergency Response',
             'CSG': 'Gynecology & Obstetrics diseases and procedure', 'CSS': 'Surgical diseases and procedure',
             'CSM': 'Medical diseases and procedure', 'CSP': 'Pediatric diseases and procedure',
             'CSO': 'Orthopedic diseases and procedure', 'CSD': 'Dental diseases and procedure',
             'GPS': 'Social Media and Communication', 'GPI': 'Infection and Exposure',
             'GPM': 'Mental Health and Mediation', 'GPP': 'Process of work', 'GPL': 'Lane (Traffic) and Legal Issues',
             'GPE': 'Environment and Working Conditions', 'GOS': 'Strategy, Structure, Security',
             'GOI': 'Information Technology & Communication, Internal control & Inventory',
             'GOM': 'Manpower, Management', 'GOP': 'Policy, Process of work & Operation',
             'GOL': 'Licensed & Professional certificate', 'GOE': 'Economy'}
colors2 = np.array([["#e1f5fe", "#f6c8b6", "#dd191d", "#dd191d", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ff8f00", "#ff8f00", "#dd191d", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#ffee58", "#ffee58", "#ff8f00", "#dd191d", "#dd191d"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#ffee58", "#ffee58", "#ff8f00", "#ff8f00"],
                    ["#e1f5fe", "#f6c8b6", "#42db41", "#42db41", "#42db41", "#ffee58", "#ffee58"],
                    ["#e1f5fe", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6", "#f6c8b6"],
                    ["#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe", "#e1f5fe"]])
risk_color_data = {
    'Category Color': ["Critical", "Critical", "Critical", "Critical", "Critical", "High", "High", "Critical",
                       "Critical", "Critical", "Medium", "Medium", "High", "Critical", "Critical", "Low", "Medium",
                       "Medium", "High", "High", "Low", "Low", "Low", "Medium", "Medium"],
    'Risk Level': ["51", "52", "53", "54", "55", "41", "42", "43", "44", "45", "31", "32", "33", "34", "35", "21", "22",
                   "23", "24", "25", "11", "12", "13", "14", "15"]}
risk_color_df = pd.DataFrame(risk_color_data)
display_cols_common = ['Occurrence Date', 'Incident', 'รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact', 'Impact Level',
                       'รายละเอียดการเกิด', 'Resulting Actions']
# ==============================================================================
# PAGE DISPLAY LOGIC
# ==============================================================================
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

if not st.session_state.data_ready:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.container():
            LOGO_URL_HEADER = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/refs/heads/main/logo1.png"
            LOGO_HEIGHT_HEADER = 28
            header_text = "เครื่องมือวิเคราะห์ข้อมูลและทะเบียนความเสี่ยงจากการรายงานอุบัติการณ์ในโรงพยาบาล (HOSPITAL OCCURRENCE/INCIDENT ANALYSIS & RISK REGISTER: HOIA-RR tool) "
            logo_html_tag = f'<img src="{LOGO_URL_HEADER}" style="height: {LOGO_HEIGHT_HEADER}px; margin-right: 10px; vertical-align: middle;">'
            st.markdown(
                f'<div class="custom-header" style="display: flex; align-items: center;">{logo_html_tag}<span>{header_text}</span></div>',
                unsafe_allow_html=True)
        st.title("ยินดีต้อนรับ")
        st.markdown("กรุณาอัปโหลดไฟล์รายงานอุบัติการณ์ (.xlsx) เพื่อเริ่มการวิเคราะห์ข้อมูล")
        if st.session_state.get('upload_error_message'):
            st.error(st.session_state.upload_error_message)
            st.session_state.upload_error_message = None
        uploaded_file_main = st.file_uploader("เลือกไฟล์ Excel ของคุณที่นี่:", type=".xlsx", key="main_page_uploader",
                                              label_visibility="collapsed")

        st.markdown(
            '<p style="font-size:12px; color:gray;">*เครื่องมือนี้เป็นส่วนหนึ่งของวิทยานิพนธ์ IMPLEMENTING THE  HOSPITAL OCCURRENCE/INCIDENT ANALYSIS & RISK REGISTER (HOIA-RR TOOL) IN THAI HOSPITALS: A STUDY ON EFFECTIVE ADOPTION โดย นางสาววิลาศินี  เขื่อนแก้ว นักศึกษาปริญญาโท สำนักวิชาวิทยาศาสตร์สุขภาพ มหาวิทยาลัยแม่ฟ้าหลวง</p>',
            unsafe_allow_html=True)

    if uploaded_file_main is not None:
        st.session_state.upload_error_message = None
        with st.spinner("กำลังประมวลผลไฟล์ กรุณารอสักครู่..."):
            df = load_data(uploaded_file_main)
            if not df.empty:
                df.replace('', 'None', inplace=True)
                df = df.fillna('None')
                df.rename(columns={'วดป.ที่เกิด': 'Occurrence Date', 'ความรุนแรง': 'Impact'}, inplace=True)
                required_cols_in_upload = ['Incident', 'Occurrence Date', 'Impact']
                missing_cols_check = [col for col in required_cols_in_upload if col not in df.columns]
                if missing_cols_check:
                    st.error(f"ไฟล์อัปโหลดขาดคอลัมน์ที่จำเป็น: {', '.join(missing_cols_check)}.")
                    st.stop()
                df['Impact_original_value'] = df['Impact']
                df['Impact'] = df['Impact'].astype(str).str.strip()
                df['รหัส'] = df['Incident'].astype(str).str.slice(0, 6).str.strip()
                if not df2.empty:
                    df = pd.merge(df, df2, on='รหัส', how='left')
                for col_name in ["ชื่ออุบัติการณ์ความเสี่ยง", "กลุ่ม", "หมวด"]:
                    if col_name not in df.columns:
                        df[col_name] = 'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)'
                    else:
                        df[col_name].fillna('N/A (ไม่พบรหัสใน AllCode)', inplace=True)
                try:
                    df['Occurrence Date'] = pd.to_datetime(df['Occurrence Date'], errors='coerce')
                    df.dropna(subset=['Occurrence Date'], inplace=True)
                    if df.empty:
                        st.error("ไม่พบข้อมูลวันที่ (Occurrence Date) ที่ถูกต้องหลังจากการแปลงค่า")
                        st.stop()
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการแปลง 'Occurrence Date': {e}")
                    st.stop()

                metrics_data_dict = {}
                metrics_data_dict['df_original_rows'] = df.shape[0]
                metrics_data_dict['total_psg9_incidents_for_metric1'] = \
                df[df['รหัส'].isin(psg9_r_codes_for_counting)].shape[0] if psg9_r_codes_for_counting else 0
                if sentinel_composite_keys:
                    df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                        str).str.strip()
                    metrics_data_dict['total_sentinel_incidents_for_metric1'] = \
                    df[df['Sentinel code for check'].isin(sentinel_composite_keys)].shape[0]
                else:
                    metrics_data_dict['total_sentinel_incidents_for_metric1'] = 0

                impact_level_map = {('A', 'B', '1'): '1', ('C', 'D', '2'): '2', ('E', 'F', '3'): '3',
                                    ('G', 'H', '4'): '4', ('I', '5'): '5'}


                def map_impact_level_func(impact_val):
                    impact_val_str = str(impact_val)
                    for k, v_level in impact_level_map.items():
                        if impact_val_str in k: return v_level
                    return 'N/A'


                df['Impact Level'] = df['Impact'].apply(map_impact_level_func)
                severe_impact_levels_list = ['3', '4', '5']
                df_severe_incidents_calc = df[df['Impact Level'].isin(severe_impact_levels_list)].copy()
                metrics_data_dict['total_severe_incidents'] = df_severe_incidents_calc.shape[0]
                metrics_data_dict['total_severe_psg9_incidents'] = \
                df_severe_incidents_calc[df_severe_incidents_calc['รหัส'].isin(psg9_r_codes_for_counting)].shape[
                    0] if psg9_r_codes_for_counting else 0

                if 'Resulting Actions' in df.columns:
                    unresolved_conditions = df_severe_incidents_calc['Resulting Actions'].astype(str).isin(['None', ''])
                    df_severe_unresolved_calc = df_severe_incidents_calc[unresolved_conditions].copy()
                    metrics_data_dict['total_severe_unresolved_incidents_val'] = df_severe_unresolved_calc.shape[0]
                    metrics_data_dict['total_severe_unresolved_psg9_incidents_val'] = \
                    df_severe_unresolved_calc[df_severe_unresolved_calc['รหัส'].isin(psg9_r_codes_for_counting)].shape[
                        0] if psg9_r_codes_for_counting else 0
                else:
                    df_severe_unresolved_calc = pd.DataFrame()
                    metrics_data_dict['total_severe_unresolved_incidents_val'] = "N/A"
                    metrics_data_dict['total_severe_unresolved_psg9_incidents_val'] = "N/A"

                metrics_data_dict['df_severe_unresolved_for_expander'] = df_severe_unresolved_calc
                df.drop(columns=[col for col in
                                 ['In.HCode', 'วดป.ที่ Import การเกิด', 'รหัสรายงาน', 'Sentinel code for check'] if
                                 col in df.columns], inplace=True, errors='ignore')

                total_month_calc = 1
                if not df.empty:
                    max_date_period = df['Occurrence Date'].max().to_period('M')
                    min_date_period = df['Occurrence Date'].min().to_period('M')
                    total_month_calc = (max_date_period.year - min_date_period.year) * 12 + (
                                max_date_period.month - min_date_period.month) + 1
                metrics_data_dict['total_month'] = max(1, total_month_calc)

                df['Incident Type'] = df['Incident'].astype(str).str[:3]
                df['Month'] = df['Occurrence Date'].dt.month
                df['เดือน'] = df['Month'].map(month_label)
                df['Year'] = df['Occurrence Date'].dt.year.astype(str)

                PSG9_ID_COL = 'PSG_ID'
                if not PSG9code_df_master.empty and PSG9_ID_COL in PSG9code_df_master.columns:
                    standards_to_merge = PSG9code_df_master[['รหัส', PSG9_ID_COL]].copy().drop_duplicates(
                        subset=['รหัส'])
                    standards_to_merge['รหัส'] = standards_to_merge['รหัส'].astype(str).str.strip()
                    df = pd.merge(df, standards_to_merge, on='รหัส', how='left')
                    df['หมวดหมู่มาตรฐานสำคัญ'] = df[PSG9_ID_COL].map(PSG9_label_dict).fillna(
                        "ไม่จัดอยู่ใน PSG9 Catalog")
                else:
                    df['หมวดหมู่มาตรฐานสำคัญ'] = "ไม่สามารถระบุ (PSG9code.xlsx ไม่ได้โหลด)"

                df_freq_temp = df['Incident'].value_counts().reset_index()
                df_freq_temp.columns = ['Incident', 'count']
                df_freq_temp['Incident Rate/mth'] = (df_freq_temp['count'] / metrics_data_dict['total_month']).round(1)
                df = pd.merge(df, df_freq_temp, on="Incident", how='left')
                st.session_state.df_freq_for_display = df_freq_temp.copy()

                conditions_freq = [(df['Incident Rate/mth'] < 2.0), (df['Incident Rate/mth'] < 3.9),
                                   (df['Incident Rate/mth'] < 6.9), (df['Incident Rate/mth'] < 29.9)]
                choices_freq = ['1', '2', '3', '4']
                df['Frequency Level'] = np.select(conditions_freq, choices_freq, default='5')
                df['Risk Level'] = df.apply(
                    lambda row: f"{row['Impact Level']}{row['Frequency Level']}" if pd.notna(row['Impact Level']) and
                                                                                    row[
                                                                                        'Impact Level'] != 'N/A' else 'N/A',
                    axis=1)
                df = pd.merge(df, risk_color_df, on='Risk Level', how='left')
                df['Category Color'].fillna('Undefined', inplace=True)

                metrics_data_dict['total_processed_incidents'] = df.shape[0]
                st.session_state.processed_df = df.copy()
                st.session_state.metrics_data = metrics_data_dict
                st.session_state.data_ready = True
                st.rerun()
else:
    # ==============================================================================
    # โครงสร้างหลักของแอปพลิเคชัน
    # ==============================================================================
    df = st.session_state.get('processed_df', pd.DataFrame())
    metrics_data = st.session_state.get('metrics_data', {})
    df_freq = st.session_state.get('df_freq_for_display', pd.DataFrame())

    if df.empty:
        st.error("ข้อมูลไม่พร้อมใช้งาน กรุณากลับไปหน้าอัปโหลด")
        if st.button("กลับไปหน้าอัปโหลด"):
            st.session_state.clear()
            st.rerun()
        st.stop()

    total_month = metrics_data.get("total_month", 1)

    st.sidebar.markdown(
        f"""<div style="display: flex; align-items: center; margin-bottom: 1rem;"><img src="{LOGO_URL}" style="height: 32px; margin-right: 10px;"><h2 style="margin: 0; font-size: 1.7rem; color: #001f3f; font-weight: bold;">HOIA-RR Menu</h2></div>""",
        unsafe_allow_html=True)
    if st.sidebar.button("อัปโหลดไฟล์ใหม่ / Upload New File", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.sidebar.markdown("---")
    if 'Occurrence Date' in df.columns and not df.empty and pd.api.types.is_datetime64_any_dtype(df['Occurrence Date']):
        min_date_str = df['Occurrence Date'].min().strftime('%m/%Y')
        max_date_str = df['Occurrence Date'].max().strftime('%m/%Y')
        st.sidebar.markdown(f"**ช่วงข้อมูลที่วิเคราะห์:** {min_date_str} ถึง {max_date_str}")
    st.sidebar.markdown(f"**จำนวนเดือนทั้งหมด:** {total_month} เดือน")
    st.sidebar.markdown(f"**จำนวนอุบัติการณ์ในไฟล์:** {df.shape[0]:,} รายการ")
    st.sidebar.markdown("เลือกส่วนที่ต้องการแสดงผล:")

    selected_analysis = st.session_state.get('selected_analysis', analysis_options_list[0])
    for option in analysis_options_list:
        is_selected = (selected_analysis == option)
        button_type = "primary" if is_selected else "secondary"
        if st.sidebar.button(option, key=f"btn_{option}", type=button_type, use_container_width=True):
            st.session_state.selected_analysis = option
            st.rerun()

    # === วางโค้ดสำหรับปุ่ม LINE ของคุณตรงนี้ค่ะ ===
    st.sidebar.markdown("---") # เพิ่มเส้นคั่นเพื่อความสวยงาม

    line_oa_id = "@144pdywm"
    line_url = f"https://page.line.me/144pdywm"
    logo_line_url = "https://raw.githubusercontent.com/HOIARRTool/hoiarr/main/irchatbot.png"
    # สร้างปุ่มที่เป็นลิงก์ไปยัง LINE OA
    st.sidebar.link_button("คุยกับ IR-Chatbot", line_url, use_container_width=True, type="secondary")
    st.sidebar.markdown(f"""  
    
    (สำหรับวิเคราะห์ระดับความรุนแรงและแนะนำ contributing factor รายอุบัติการณ์)
    

    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        **กิตติกรรมประกาศ:** ขอขอบพระคุณ 
        - Prof. Shin Ushiro
        - นพ.อนุวัฒน์ ศุภชุติกุล 
        - นพ.ก้องเกียรติ เกษเพ็ชร์ 
        - พญ.ปิยวรรณ ลิ้มปัญญาเลิศ 
        - ภก.ปรมินทร์ วีระอนันตวัฒน์    
        - ผศ.ดร.นิเวศน์ กุลวงค์ (อ.ที่ปรึกษา)

        เป็นอย่างสูง สำหรับการริเริ่ม เติมเต็ม สนับสนุน และสร้างแรงบันดาลใจ อันเป็นรากฐานสำคัญในการพัฒนาเครื่องมือนี้
        """)
    # =========================================================================
    # PAGE CONTENT ROUTING (โค้ดฉบับสมบูรณ์ทั้งหมด)
    # =========================================================================
    if selected_analysis == "แดชบอร์ดสรุปภาพรวม":
        st.markdown("<h4 style='color: #001f3f;'>สรุปภาพรวมอุบัติการณ์:</h4>", unsafe_allow_html=True)

        with st.expander("แสดง/ซ่อน ตารางข้อมูลอุบัติการณ์ทั้งหมด (Full Data Table)"):
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")
            })

        dashboard_expander_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
        date_format_config = {
            "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")
        }

        total_processed_incidents = metrics_data.get("total_processed_incidents", 0)
        total_psg9_incidents_for_metric1 = metrics_data.get("total_psg9_incidents_for_metric1", 0)
        total_sentinel_incidents_for_metric1 = metrics_data.get("total_sentinel_incidents_for_metric1", 0)
        total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
        total_severe_psg9_incidents = metrics_data.get("total_severe_psg9_incidents", 0)
        total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", "N/A")
        total_severe_unresolved_psg9_incidents_val = metrics_data.get("total_severe_unresolved_psg9_incidents_val",
                                                                      "N/A")
        df_severe_incidents = df[
            df['Impact Level'].isin(['3', '4', '5'])].copy() if 'Impact Level' in df.columns else pd.DataFrame()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", f"{total_processed_incidents:,}")
        with col2:
            st.metric("PSG9", f"{total_psg9_incidents_for_metric1:,}")
            with st.expander(f"ดูรายละเอียด ({total_psg9_incidents_for_metric1} รายการ)"):
                psg9_df = df[df['รหัส'].isin(
                    psg9_r_codes_for_counting)] if 'psg9_r_codes_for_counting' in globals() and psg9_r_codes_for_counting else pd.DataFrame()
                st.dataframe(psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col3:
            st.metric("Sentinel", f"{total_sentinel_incidents_for_metric1:,}")
            with st.expander(f"ดูรายละเอียด ({total_sentinel_incidents_for_metric1} รายการ)"):
                if 'sentinel_composite_keys' in globals() and sentinel_composite_keys:
                    df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                        str).str.strip()
                    sentinel_df = df[df['Sentinel code for check'].isin(sentinel_composite_keys)]
                    st.dataframe(sentinel_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)
                else:
                    st.write("ไม่สามารถแสดงข้อมูลได้")

        col4, col5, col6, col7 = st.columns(4)
        with col4:
            st.metric("E-I & 3-5 [all]", f"{total_severe_incidents:,}")
            with st.expander(f"ดูรายละเอียด ({total_severe_incidents} รายการ)"):
                st.dataframe(df_severe_incidents[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col5:
            st.metric("E-I & 3-5 [PSG9]", f"{total_severe_psg9_incidents:,}")
            with st.expander(f"ดูรายละเอียด ({total_severe_psg9_incidents} รายการ)"):
                severe_psg9_df = df_severe_incidents[df_severe_incidents['รหัส'].isin(
                    psg9_r_codes_for_counting)] if 'psg9_r_codes_for_counting' in globals() and psg9_r_codes_for_counting else pd.DataFrame()
                st.dataframe(severe_psg9_df[dashboard_expander_cols], use_container_width=True, hide_index=True,
                             column_config=date_format_config)
        with col6:
            val_unresolved_all = f"{total_severe_unresolved_incidents_val:,}" if isinstance(
                total_severe_unresolved_incidents_val, int) else "N/A"
            st.metric(f"E-I & 3-5 [all] ที่ยังไม่ถูกแก้ไข", val_unresolved_all)
            if isinstance(total_severe_unresolved_incidents_val, int) and total_severe_unresolved_incidents_val > 0:
                with st.expander(f"ดูรายละเอียด ({total_severe_unresolved_incidents_val} รายการ)"):
                    unresolved_df_all = df[
                        df['Impact Level'].isin(['3', '4', '5']) & df['Resulting Actions'].astype(str).isin(
                            ['None', ''])]
                    st.dataframe(unresolved_df_all[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)
        with col7:
            val_unresolved_psg9 = f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(
                total_severe_unresolved_psg9_incidents_val, int) else "N/A"
            st.metric(f"E-I & 3-5 [PSG9] ที่ยังไม่ถูกแก้ไข", val_unresolved_psg9)
            if isinstance(total_severe_unresolved_psg9_incidents_val,
                          int) and total_severe_unresolved_psg9_incidents_val > 0:
                with st.expander(f"ดูรายละเอียด ({total_severe_unresolved_psg9_incidents_val} รายการ)"):
                    unresolved_df_all = df[
                        df['Impact Level'].isin(['3', '4', '5']) & df['Resulting Actions'].astype(str).isin(
                            ['None', ''])]
                    unresolved_df_psg9 = unresolved_df_all[unresolved_df_all['รหัส'].isin(
                        psg9_r_codes_for_counting)] if 'psg9_r_codes_for_counting' in globals() and psg9_r_codes_for_counting else pd.DataFrame()
                    st.dataframe(unresolved_df_psg9[dashboard_expander_cols], use_container_width=True, hide_index=True,
                                 column_config=date_format_config)

    elif selected_analysis == "รายการ Sentinel Events":
        st.markdown("<h4 style='color: #001f3f;'>รายการ Sentinel Events ที่ตรวจพบ</h4>", unsafe_allow_html=True)
        if 'sentinel_composite_keys' in globals() and sentinel_composite_keys:
            df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                str).str.strip()
            sentinel_events = df[df['Sentinel code for check'].isin(sentinel_composite_keys)].copy()
            if not Sentinel2024_df.empty and 'ชื่ออุบัติการณ์ความเสี่ยง' in Sentinel2024_df.columns:
                sentinel_events = pd.merge(sentinel_events,
                                           Sentinel2024_df[['รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง']].rename(
                                               columns={'ชื่ออุบัติการณ์ความเสี่ยง': 'Sentinel Event Name'}),
                                           on=['รหัส', 'Impact'], how='left')
            display_sentinel_cols = ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
            if 'Sentinel Event Name' in sentinel_events.columns:
                display_sentinel_cols.insert(2, 'Sentinel Event Name')
            final_display_cols = [col for col in display_sentinel_cols if col in sentinel_events.columns]
            st.dataframe(sentinel_events[final_display_cols], use_container_width=True, hide_index=True,
                         column_config={
                             "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
        else:
            st.warning("ไม่สามารถตรวจสอบ Sentinel Events ได้ (ไฟล์ Sentinel2024.xlsx อาจไม่มีข้อมูล)")


    elif selected_analysis == "Risk Matrix (Interactive)":

        st.subheader("Risk Matrix (Interactive)")

        matrix_data_counts = np.zeros((5, 5), dtype=int)

        impact_level_keys = ['5', '4', '3', '2', '1']

        freq_level_keys = ['1', '2', '3', '4', '5']

        if 'Risk Level' in df.columns and 'Impact Level' in df.columns and 'Frequency Level' in df.columns and not df[

            df['Risk Level'] != 'N/A'].empty:

            risk_counts_df = df.groupby(['Impact Level', 'Frequency Level']).size().reset_index(name='counts')

            for _, row in risk_counts_df.iterrows():

                il_key = str(row['Impact Level'])

                fl_key = str(row['Frequency Level'])

                count_val = row['counts']

                if il_key in impact_level_keys and fl_key in freq_level_keys:
                    row_idx = impact_level_keys.index(il_key)

                    col_idx = freq_level_keys.index(fl_key)

                    matrix_data_counts[row_idx, col_idx] = count_val

        impact_labels_display = {

            '5': "I / 5<br>Extreme / Death", '4': "G-H / 4<br>Major / Severe",

            '3': "E-F / 3<br>Moderate", '2': "C-D / 2<br>Minor / Low", '1': "A-B / 1<br>Insignificant / No Harm"

        }

        freq_labels_display_short = {"1": "F1", "2": "F2", "3": "F3", "4": "F4", "5": "F5"}

        freq_labels_display_long = {

            "1": "Remote<br>(<2/mth)", "2": "Uncommon<br>(2-3/mth)", "3": "Occasional<br>(4-6/mth)",

            "4": "Probable<br>(7-29/mth)", "5": "Frequent<br>(>=30/mth)"

        }

        impact_to_color_row = {'5': 0, '4': 1, '3': 2, '2': 3, '1': 4}

        freq_to_color_col = {'1': 2, '2': 3, '3': 4, '4': 5, '5': 6}

        cols_header = st.columns([2.2, 1, 1, 1, 1, 1])

        with cols_header[0]:

            st.markdown(

                f"<div style='background-color:{colors2[6, 0]}; color:{get_text_color_for_bg(colors2[6, 0])}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; align-items:center; justify-content:center;'>Impact / Frequency</div>",

                unsafe_allow_html=True)

        for i, fl_key in enumerate(freq_level_keys):
            with cols_header[i + 1]:
                header_freq_bg_color = colors2[5, freq_to_color_col.get(fl_key, 2) - 1]

                header_freq_text_color = get_text_color_for_bg(header_freq_bg_color)

                st.markdown(

                    f"<div style='background-color:{header_freq_bg_color}; color:{header_freq_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:60px; display:flex; flex-direction: column; align-items:center; justify-content:center;'><div>{freq_labels_display_short.get(fl_key, '')}</div><div style='font-size:0.7em;'>{freq_labels_display_long.get(fl_key, '')}</div></div>",

                    unsafe_allow_html=True)

        for il_key in impact_level_keys:

            cols_data_row = st.columns([2.2, 1, 1, 1, 1, 1])

            row_idx_color = impact_to_color_row[il_key]

            with cols_data_row[0]:

                header_impact_bg_color = colors2[row_idx_color, 1]

                header_impact_text_color = get_text_color_for_bg(header_impact_bg_color)

                st.markdown(

                    f"<div style='background-color:{header_impact_bg_color}; color:{header_impact_text_color}; padding:8px; text-align:center; font-weight:bold; border-radius:3px; margin:1px; height:70px; display:flex; align-items:center; justify-content:center;'>{impact_labels_display[il_key]}</div>",

                    unsafe_allow_html=True)

            for i, fl_key in enumerate(freq_level_keys):

                with cols_data_row[i + 1]:

                    count = matrix_data_counts[impact_level_keys.index(il_key), freq_level_keys.index(fl_key)]

                    cell_bg_color = colors2[row_idx_color, freq_to_color_col[fl_key]]

                    text_color = get_text_color_for_bg(cell_bg_color)

                    st.markdown(

                        f"<div style='background-color:{cell_bg_color}; color:{text_color}; padding:5px; margin:1px; border-radius:3px; text-align:center; font-weight:bold; min-height:40px; display:flex; align-items:center; justify-content:center;'>{count}</div>",

                        unsafe_allow_html=True)

                    if count > 0:

                        button_key = f"view_risk_{il_key}_{fl_key}"

                        if st.button("👁️", key=button_key, help=f"ดูรายการ - {count} รายการ", use_container_width=True):
                            st.session_state.clicked_risk_impact = il_key

                            st.session_state.clicked_risk_freq = fl_key

                            st.session_state.show_incident_table = True

                            st.rerun()

                    else:

                        st.markdown("<div style='height:38px; margin-top:5px;'></div>", unsafe_allow_html=True)

        if st.session_state.show_incident_table and st.session_state.clicked_risk_impact is not None:

            il_selected = st.session_state.clicked_risk_impact

            fl_selected = st.session_state.clicked_risk_freq

            df_filtered_incidents = df[(df['Impact Level'].astype(str) == il_selected) & (

                    df['Frequency Level'].astype(str) == fl_selected)].copy()

            expander_title = f"รายการอุบัติการณ์: Impact Level {il_selected}, Frequency Level {fl_selected} - พบ {len(df_filtered_incidents)} รายการ"

            with st.expander(expander_title, expanded=True):

                st.dataframe(df_filtered_incidents[display_cols_common], use_container_width=True, hide_index=True)

                if st.button("ปิดรายการ", key="clear_risk_selection_button"):
                    st.session_state.show_incident_table = False

                    st.session_state.clicked_risk_impact = None

                    st.session_state.clicked_risk_freq = None

                    st.rerun()
    elif selected_analysis == "การแก้ไขอุบัติการณ์":
        st.subheader("✅ สรุปอุบัติการณ์ที่ได้รับการแก้ไขแล้ว (ตามหมวด)")
        all_categories = sorted([cat for cat in df['หมวด'].unique() if cat and pd.notna(cat)])
        resolved_df_total_count = df[~df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])].copy()
        if resolved_df_total_count.empty:
            st.info("ยังไม่มีรายการอุบัติการณ์ที่ถูกบันทึกการแก้ไข")
        else:
            st.metric("จำนวนรายการที่ได้รับการแก้ไขทั้งหมด", f"{len(resolved_df_total_count):,} รายการ")
            for category in all_categories:
                total_in_category_df = df[df['หมวด'] == category]
                total_count = len(total_in_category_df)
                resolved_in_category_df = total_in_category_df[
                    ~total_in_category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                resolved_count = len(resolved_in_category_df)
                if resolved_count > 0:
                    expander_title = f"หมวด: {category} (แก้ไขแล้ว {resolved_count}/{total_count} รายการ)"
                    with st.expander(expander_title):
                        display_cols = ['Occurrence Date', 'Incident', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact',
                                        'Resulting Actions']
                        displayable_cols = [col for col in display_cols if col in resolved_in_category_df.columns]
                        st.dataframe(resolved_in_category_df[displayable_cols], use_container_width=True,
                                     hide_index=True)
        st.markdown("---")
        st.subheader("⏱️ รายการอุบัติการณ์ที่รอการแก้ไข (ตามความรุนแรง)")
        unresolved_df = df[df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])].copy()
        if unresolved_df.empty:
            st.success("🎉 ไม่พบรายการที่รอการแก้ไขในขณะนี้ ยอดเยี่ยมมากครับ!")
        else:
            st.metric("จำนวนรายการที่รอการแก้ไขทั้งหมด", f"{len(unresolved_df):,} รายการ")
            severity_order = ['Critical', 'High', 'Medium', 'Low', 'Undefined']
            for severity in severity_order:
                severity_df = unresolved_df[unresolved_df['Category Color'] == severity]
                if not severity_df.empty:
                    with st.expander(f"ระดับความรุนแรง: {severity} ({len(severity_df)} รายการ)"):
                        display_cols = ['Occurrence Date', 'Incident', 'ชื่ออุบัติการณ์ความเสี่ยง', 'Impact',
                                        'Impact Level', 'รายละเอียดการเกิด']
                        displayable_cols = [col for col in display_cols if col in severity_df.columns]
                        st.dataframe(severity_df[displayable_cols], use_container_width=True, hide_index=True)

    elif selected_analysis == "กราฟสรุปอุบัติการณ์ (รายมิติ)":
        st.markdown("<h4 style='color: #001f3f;'>กราฟสรุปอุบัติการณ์ (แบ่งตามมิติต่างๆ)</h4>", unsafe_allow_html=True)
        pastel_color_discrete_map_dimensions = {
            'Critical': '#FF9999', 'High': '#FFCC99', 'Medium': '#FFFF99',
            'Low': '#99FF99', 'Undefined': '#D3D3D3'
        }
        tab1_v, tab2_v, tab3_v, tab4_v = st.tabs(
            ["👁️By Goals (หมวด)", "👁️By Group (กลุ่ม)", "👁️By Shift (เวร)", "👁️By Place (สถานที่)"])
        if isinstance(df, pd.DataFrame):
            df_charts = df.copy()
            if 'Count' not in df_charts.columns: df_charts['Count'] = 1

            with tab1_v:
                st.markdown(f"Incidents by Safety Goals ({total_month}m)")
                if 'หมวด' in df_charts.columns:
                    df_c1 = df_charts[~df_charts['หมวด'].isin(
                        ['N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)', 'N/A (ไม่พบรหัสใน AllCode)'])]
                    if not df_c1.empty:
                        fig_c1 = px.bar(df_c1, x='หมวด', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c1, use_container_width=True)
            with tab2_v:
                st.markdown(f"Incidents by Group ({total_month}m)")
                if 'กลุ่ม' in df_charts.columns:
                    df_c2 = df_charts[~df_charts['กลุ่ม'].isin(
                        ['N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)', 'N/A (ไม่พบรหัสใน AllCode)'])]
                    if not df_c2.empty:
                        fig_c2 = px.bar(df_c2, x='กลุ่ม', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c2, use_container_width=True)
            with tab3_v:
                st.markdown(f"Incidents by Shift ({total_month}m)")
                if 'ช่วงเวลา/เวร' in df_charts.columns:
                    df_c3 = df_charts[df_charts['ช่วงเวลา/เวร'] != 'N/A (ข้อมูลจากไฟล์อัปโหลด)']
                    if not df_c3.empty:
                        fig_c3 = px.bar(df_c3, x='ช่วงเวลา/เวร', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c3, use_container_width=True)
            with tab4_v:
                st.markdown(f"Incidents by Place ({total_month}m)")
                if 'ชนิดสถานที่' in df_charts.columns:
                    df_c4 = df_charts[df_charts['ชนิดสถานที่'] != 'N/A (ข้อมูลจากไฟล์อัปโหลด)']
                    if not df_c4.empty:
                        fig_c4 = px.bar(df_c4, x='ชนิดสถานที่', y='Count', color='Category Color',
                                        color_discrete_map=pastel_color_discrete_map_dimensions)
                        st.plotly_chart(fig_c4, use_container_width=True)

    # =========================================================================
    # โค้ดสำหรับหน้า "Sankey: ภาพรวม" (แก้ไขตัวอักษรหนา/มีเงา)
    # =========================================================================
    elif selected_analysis == "Sankey: ภาพรวม":
        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: ภาพรวม</h4>", unsafe_allow_html=True)

        # --- เพิ่ม CSS พิเศษสำหรับแก้ไขการแสดงผลตัวอักษรในกราฟ ---
        st.markdown("""
        <style>
            /* เจาะจงไปที่ตัวหนังสือในกราฟ Plotly */
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important;      /* เอาเส้นขอบตัวอักษรออก */
                text-shadow: none !important;    /* เอาเงาของตัวอักษรออก */
                paint-order: stroke fill;        /* สั่งให้วาดสีพื้นก่อนเส้นขอบ */
            }
        </style>
        """, unsafe_allow_html=True)

        req_cols = ['หมวด', 'Impact', 'Impact Level', 'Category Color']
        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in req_cols):
            sankey_df = df.copy()
            placeholders_to_filter_generic = ['None', '', np.nan, 'N/A', 'ไม่ระบุ',
                                              'N/A (ข้อมูลจาก AllCode ไม่พร้อมใช้งาน)',
                                              'N/A (ไม่พบรหัสใน AllCode หรือค่าว่างใน AllCode)']
            placeholders_to_filter_stripped_lower = [str(p).strip().lower() for p in placeholders_to_filter_generic if
                                                     pd.notna(p)]

            sankey_df['หมวด_Node'] = "หมวด: " + sankey_df['หมวด'].astype(str).str.strip()
            sankey_df = sankey_df[~sankey_df['หมวด'].str.lower().isin(placeholders_to_filter_stripped_lower)]
            sankey_df = sankey_df[sankey_df['หมวด'] != '']

            sankey_df['Impact_AI_Node'] = "Impact: " + sankey_df['Impact'].astype(str).str.strip()
            sankey_df.loc[sankey_df['Impact'].str.lower().isin(placeholders_to_filter_stripped_lower) | (
                    sankey_df['Impact'] == ''), 'Impact_AI_Node'] = "Impact: ไม่ระบุ A-I"

            impact_level_display_map = {'1': "Level: 1 (A-B)", '2': "Level: 2 (C-D)", '3': "Level: 3 (E-F)",
                                        '4': "Level: 4 (G-H)", '5': "Level: 5 (I)", 'N/A': "Level: ไม่ระบุ"}
            sankey_df['Impact_Level_Node'] = sankey_df['Impact Level'].astype(str).str.strip().map(
                impact_level_display_map).fillna("Level: ไม่ระบุ")
            sankey_df['Risk_Category_Node'] = "Risk: " + sankey_df['Category Color'].astype(str).str.strip()

            node_cols = ['หมวด_Node', 'Impact_AI_Node', 'Impact_Level_Node', 'Risk_Category_Node']
            sankey_df.dropna(subset=node_cols, inplace=True)

            if sankey_df.empty:
                st.warning("ไม่มีข้อมูลที่สามารถแสดงผลใน Sankey diagram ได้หลังจากการกรอง")
            else:
                labels_muad = sorted(list(sankey_df['หมวด_Node'].unique()))
                impact_ai_order = [f"Impact: {i}" for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']] + [
                    "Impact: ไม่ระบุ A-I"]
                labels_impact_ai = sorted(list(sankey_df['Impact_AI_Node'].unique()),
                                          key=lambda x: impact_ai_order.index(x) if x in impact_ai_order else 999)
                level_order_map = {"Level: 1 (A-B)": 1, "Level: 2 (C-D)": 2, "Level: 3 (E-F)": 3, "Level: 4 (G-H)": 4,
                                   "Level: 5 (I)": 5, "Level: ไม่ระบุ": 6}
                labels_impact_level = sorted(list(sankey_df['Impact_Level_Node'].unique()),
                                             key=lambda x: level_order_map.get(x, 999))
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_cat = sorted(list(sankey_df['Risk_Category_Node'].unique()),
                                         key=lambda x: risk_order.index(x) if x in risk_order else 999)

                all_labels_ordered = labels_muad + labels_impact_ai + labels_impact_level + labels_risk_cat
                all_labels = list(pd.Series(all_labels_ordered).unique())
                label_to_idx = {label: i for i, label in enumerate(all_labels)}

                source_indices, target_indices, values = [], [], []
                links1 = sankey_df.groupby(['หมวด_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links1.iterrows():
                    source_indices.append(label_to_idx[row['หมวด_Node']])
                    target_indices.append(label_to_idx[row['Impact_AI_Node']])
                    values.append(row['value'])
                links2 = sankey_df.groupby(['Impact_AI_Node', 'Impact_Level_Node']).size().reset_index(name='value')
                for _, row in links2.iterrows():
                    source_indices.append(label_to_idx[row['Impact_AI_Node']])
                    target_indices.append(label_to_idx[row['Impact_Level_Node']])
                    values.append(row['value'])
                links3 = sankey_df.groupby(['Impact_Level_Node', 'Risk_Category_Node']).size().reset_index(name='value')
                for _, row in links3.iterrows():
                    source_indices.append(label_to_idx[row['Impact_Level_Node']])
                    target_indices.append(label_to_idx[row['Risk_Category_Node']])
                    values.append(row['value'])

                if source_indices:
                    node_colors = []
                    palette1, palette2, palette3 = px.colors.qualitative.Pastel1, px.colors.qualitative.Pastel2, px.colors.qualitative.Set3
                    risk_color_map = {"Risk: Critical": "red", "Risk: High": "orange", "Risk: Medium": "#F7DC6F",
                                      "Risk: Low": "green", "Risk: Undefined": "grey"}
                    for label in all_labels:
                        if label in labels_muad:
                            node_colors.append(palette1[labels_muad.index(label) % len(palette1)])
                        elif label in labels_impact_ai:
                            node_colors.append(palette2[labels_impact_ai.index(label) % len(palette2)])
                        elif label in labels_impact_level:
                            node_colors.append(palette3[labels_impact_level.index(label) % len(palette3)])
                        elif label in labels_risk_cat:
                            node_colors.append(risk_color_map.get(label, 'grey'))
                        else:
                            node_colors.append('rgba(200,200,200,0.8)')
                    link_colors_rgba = [
                        f'rgba({int(c.lstrip("#")[0:2], 16)},{int(c.lstrip("#")[2:4], 16)},{int(c.lstrip("#")[4:6], 16)},0.3)' if c.startswith(
                            '#') else 'rgba(200,200,200,0.3)' for c in [node_colors[s] for s in source_indices]]

                    fig = go.Figure(data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=15, thickness=18, line=dict(color="rgba(0,0,0,0.6)", width=0.75),
                                  label=all_labels, color=node_colors,
                                  hovertemplate='%{label} มีจำนวน: %{value}<extra></extra>'),
                        link=dict(source=source_indices, target=target_indices, value=values, color=link_colors_rgba,
                                  hovertemplate='จาก %{source.label}<br />ไปยัง %{target.label}<br />จำนวน: %{value}<extra></extra>')
                    )])

                    # --- แก้ไข Layout ของกราฟ ---
                    fig.update_layout(
                        title_text="<b>แผนภาพ Sankey:</b> หมวด -> Impact (A-I) -> Impact Level (1-5) -> Risk Category",
                        font_size=12,  # เพิ่มขนาดตัวอักษร
                        height=max(700, len(all_labels) * 18),
                        width=1200,  # กำหนดความละเอียดต้นฉบับ
                        template='plotly_white',  # ใช้ Theme ที่เรียบง่าย ลดโอกาสเกิดเงา
                        margin=dict(t=60, l=10, r=10, b=20)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("ไม่สามารถสร้างลิงก์สำหรับ Sankey diagram ได้")
        else:
            st.warning(f"ไม่พบคอลัมน์ที่จำเป็น ({', '.join(req_cols)}) สำหรับการสร้าง Sankey diagram")

    # =========================================================================
    # โค้ดสำหรับหน้า "Sankey: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ"
    # =========================================================================
    elif selected_analysis == "Sankey: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ":

        st.markdown("<h4 style='color: #001f3f;'>Sankey Diagram: มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ</h4>",
                    unsafe_allow_html=True)

        # --- เพิ่ม CSS พิเศษสำหรับแก้ไขการแสดงผลตัวอักษรในกราฟ ---
        st.markdown("""
        <style>
            .plot-container .svg-container .sankey-node text {
                stroke-width: 0 !important;
                text-shadow: none !important;
                paint-order: stroke fill;
            }
        </style>
        """, unsafe_allow_html=True)

        psg9_placeholders_to_filter = ["ไม่จัดอยู่ใน PSG9 Catalog", "ไม่สามารถระบุ (Merge PSG9 ล้มเหลว)"]
        psg9_placeholders_stripped_lower = [str(p).strip().lower() for p in psg9_placeholders_to_filter if pd.notna(p)]
        required_cols_sankey_simplified = ['หมวดหมู่มาตรฐานสำคัญ', 'รหัส', 'Impact', 'ชื่ออุบัติการณ์ความเสี่ยง',
                                           'Category Color']

        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in required_cols_sankey_simplified):
            sankey_df_new = df.copy()
            sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned'] = sankey_df_new['หมวดหมู่มาตรฐานสำคัญ'].astype(
                str).str.strip()
            sankey_df_new = sankey_df_new[
                ~sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned'].str.lower().isin(psg9_placeholders_stripped_lower)]

            psg9_to_cp_gp_map = {PSG9_label_dict[num].strip(): 'CP (หมวดตาม PSG9)' for num in [1, 3, 4, 5, 6, 7, 8, 9]
                                 if num in PSG9_label_dict}
            psg9_to_cp_gp_map.update(
                {PSG9_label_dict[num].strip(): 'GP (หมวดตาม PSG9)' for num in [2] if num in PSG9_label_dict})

            sankey_df_new['หมวด_CP_GP_Node'] = sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned'].map(psg9_to_cp_gp_map)
            sankey_df_new['หมวดหมู่PSG_Node'] = "PSG9: " + sankey_df_new['หมวดหมู่มาตรฐานสำคัญ_cleaned']
            sankey_df_new['รหัส_Node'] = "รหัส: " + sankey_df_new['รหัส'].astype(str).str.strip()
            sankey_df_new['Impact_AI_Node'] = "Impact: " + sankey_df_new['Impact'].astype(str).str.strip()
            sankey_df_new['Risk_Category_Node'] = "Risk: " + sankey_df_new['Category Color'].fillna('Undefined')
            sankey_df_new['ชื่ออุบัติการณ์ความเสี่ยง_for_hover'] = sankey_df_new['ชื่ออุบัติการณ์ความเสี่ยง'].fillna(
                'ไม่มีคำอธิบาย')

            cols_for_dropna_new_sankey = ['หมวด_CP_GP_Node', 'หมวดหมู่PSG_Node', 'รหัส_Node', 'Impact_AI_Node',
                                          'Risk_Category_Node']
            sankey_df_new.dropna(subset=cols_for_dropna_new_sankey, inplace=True)

            if sankey_df_new.empty:
                st.warning("ไม่มีข้อมูล PSG9 ที่สามารถแสดงผลใน Sankey diagram ได้หลังจากการกรอง")
            else:
                labels_muad_cp_gp_simp = sorted(list(sankey_df_new['หมวด_CP_GP_Node'].unique()))
                labels_psg9_cat_simp = sorted(list(sankey_df_new['หมวดหมู่PSG_Node'].unique()))
                rh_node_to_desc_map = sankey_df_new.drop_duplicates(subset=['รหัส_Node']).set_index('รหัส_Node')[
                    'ชื่ออุบัติการณ์ความเสี่ยง_for_hover'].to_dict()
                labels_rh_simp = sorted(list(sankey_df_new['รหัส_Node'].unique()))
                labels_impact_ai_simp = sorted(list(sankey_df_new['Impact_AI_Node'].unique()))
                risk_order = ["Risk: Critical", "Risk: High", "Risk: Medium", "Risk: Low", "Risk: Undefined"]
                labels_risk_category = sorted(list(sankey_df_new['Risk_Category_Node'].unique()),
                                              key=lambda x: risk_order.index(x) if x in risk_order else 99)
                all_labels_ordered_simp = labels_muad_cp_gp_simp + labels_psg9_cat_simp + labels_rh_simp + labels_impact_ai_simp + labels_risk_category
                all_labels_simp = list(pd.Series(all_labels_ordered_simp).unique())
                label_to_idx_simp = {label: i for i, label in enumerate(all_labels_simp)}
                customdata_for_nodes_simp = [
                    f"<br>คำอธิบาย: {str(rh_node_to_desc_map.get(label_node, ''))}" if label_node in rh_node_to_desc_map else ""
                    for label_node in all_labels_simp]

                source_indices_simp, target_indices_simp, values_simp = [], [], []
                links_l1 = sankey_df_new.groupby(['หมวด_CP_GP_Node', 'หมวดหมู่PSG_Node']).size().reset_index(
                    name='value')
                for _, row in links_l1.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['หมวด_CP_GP_Node']])
                    target_indices_simp.append(label_to_idx_simp[row['หมวดหมู่PSG_Node']])
                    values_simp.append(row['value'])
                links_l2 = sankey_df_new.groupby(['หมวดหมู่PSG_Node', 'รหัส_Node']).size().reset_index(name='value')
                for _, row in links_l2.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['หมวดหมู่PSG_Node']])
                    target_indices_simp.append(label_to_idx_simp[row['รหัส_Node']])
                    values_simp.append(row['value'])
                links_l3 = sankey_df_new.groupby(['รหัส_Node', 'Impact_AI_Node']).size().reset_index(name='value')
                for _, row in links_l3.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['รหัส_Node']])
                    target_indices_simp.append(label_to_idx_simp[row['Impact_AI_Node']])
                    values_simp.append(row['value'])
                links_l4 = sankey_df_new.groupby(['Impact_AI_Node', 'Risk_Category_Node']).size().reset_index(
                    name='value')
                for _, row in links_l4.iterrows():
                    source_indices_simp.append(label_to_idx_simp[row['Impact_AI_Node']])
                    target_indices_simp.append(label_to_idx_simp[row['Risk_Category_Node']])
                    values_simp.append(row['value'])

                if source_indices_simp:
                    node_colors_simp = []
                    palette_l1, palette_l2, palette_l3, palette_l4 = px.colors.qualitative.Bold, px.colors.qualitative.Pastel, px.colors.qualitative.Vivid, px.colors.qualitative.Set3
                    risk_cat_color_map = {"Risk: Critical": "red", "Risk: High": "orange", "Risk: Medium": "#F7DC6F",
                                          "Risk: Low": "green", "Risk: Undefined": "grey"}
                    for label_node in all_labels_simp:
                        if label_node in labels_muad_cp_gp_simp:
                            node_colors_simp.append(
                                palette_l1[labels_muad_cp_gp_simp.index(label_node) % len(palette_l1)])
                        elif label_node in labels_psg9_cat_simp:
                            node_colors_simp.append(
                                palette_l2[labels_psg9_cat_simp.index(label_node) % len(palette_l2)])
                        elif label_node in labels_rh_simp:
                            node_colors_simp.append(palette_l3[labels_rh_simp.index(label_node) % len(palette_l3)])
                        elif label_node in labels_impact_ai_simp:
                            node_colors_simp.append(
                                palette_l4[labels_impact_ai_simp.index(label_node) % len(palette_l4)])
                        elif label_node in labels_risk_category:
                            node_colors_simp.append(risk_cat_color_map.get(label_node, 'grey'))
                        else:
                            node_colors_simp.append('rgba(200,200,200,0.8)')
                    link_colors_simp = []
                    default_link_color_simp = 'rgba(200,200,200,0.35)'
                    for s_idx in source_indices_simp:
                        try:
                            hex_color = node_colors_simp[s_idx]
                            h = hex_color.lstrip('#')
                            rgb_tuple = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
                            link_colors_simp.append(f'rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},0.3)')
                        except:
                            link_colors_simp.append(default_link_color_simp)

                    fig_sankey_psg9_simplified = go.Figure(data=[go.Sankey(
                        arrangement='snap',
                        node=dict(pad=10, thickness=15, line=dict(color="rgba(0,0,0,0.4)", width=0.4),
                                  label=all_labels_simp, color=node_colors_simp, customdata=customdata_for_nodes_simp,
                                  hovertemplate='<b>%{label}</b><br>จำนวน: %{value}%{customdata}<extra></extra>'),
                        link=dict(source=source_indices_simp, target=target_indices_simp, value=values_simp,
                                  color=link_colors_simp,
                                  hovertemplate='จาก %{source.label}<br />ไปยัง %{target.label}<br />จำนวน: %{value}<extra></extra>')
                    )])

                    fig_sankey_psg9_simplified.update_layout(
                        title_text="<b>แผนภาพ SANKEY:</b> CP/GP -> หมวดหมู่ PSG9 -> รหัส -> Impact -> Risk Category",
                        font_size=11,
                        height=max(800, len(all_labels_simp) * 12 + 200),
                        width=1200,
                        template='plotly_white',
                        margin=dict(t=70, l=10, r=10, b=20)
                    )

                    st.plotly_chart(fig_sankey_psg9_simplified, use_container_width=True)
                else:
                    st.warning("ไม่สามารถสร้างลิงก์สำหรับ Sankey diagram (PSG9) ได้")
        else:
            st.warning(
                f"ไม่พบคอลัมน์ที่จำเป็น ({', '.join(required_cols_sankey_simplified)}) สำหรับการสร้าง Sankey diagram")
    elif selected_analysis == "Scatter Plot & Top 10":
        st.markdown("<h4 style='color: #001f3f;'>ความสัมพันธ์ระหว่างรหัสอุบัติการณ์และหมวดหมู่ (Scatter Plot)</h4>",
                    unsafe_allow_html=True)
        scatter_cols = ['รหัส', 'หมวด', 'Category Color', 'Incident Rate/mth']
        if isinstance(df, pd.DataFrame) and all(col in df.columns for col in scatter_cols):
            df_sc = df.dropna(subset=scatter_cols, how='any')
            if not df_sc.empty:
                fig_sc = px.scatter(df_sc, x='รหัส', y='หมวด', color='Category Color', size='Incident Rate/mth',
                                    hover_data=['Incident', 'ชื่ออุบัติการณ์ความเสี่ยง'], size_max=30,
                                    color_discrete_map=color_discrete_map)
                st.plotly_chart(fig_sc, theme="streamlit", use_container_width=True)
            else:
                st.warning("Insufficient data for Scatter Plot.")
        else:
            st.warning(f"Missing required columns for Scatter Plot: {scatter_cols}")

        st.markdown("---")
        st.subheader("Top 10 อุบัติการณ์ (ตามความถี่)")
        if not df_freq.empty:
            df_freq_top10 = df_freq.nlargest(10, 'count')
            st.dataframe(df_freq_top10, use_container_width=True)
        else:
            st.warning("Cannot display Top 10 Incidents.")


    elif selected_analysis == "สรุปอุบัติการณ์ตาม Safety Goals":

        st.markdown("<h4 style='color: #001f3f;'>สรุปอุบัติการณ์ตามเป้าหมาย</h4>", unsafe_allow_html=True)

        goal_definitions = {

            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",

            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",

            "Personnel Safety": "P:Personnel Safety Goals",

            "Organization Safety": "O:Organization Safety Goals"

        }

        for display_name, cat_name in goal_definitions.items():

            st.markdown(f"##### {display_name}")

            is_org_safety = (display_name == "Organization Safety")

            summary_table = create_goal_summary_table(

                df, cat_name,

                e_up_non_numeric_levels_param=[] if is_org_safety else ['A', 'B', 'C', 'D'],

                e_up_numeric_levels_param=['1', '2'] if is_org_safety else None,

                is_org_safety_table=is_org_safety

            )

            if summary_table is not None and not summary_table.empty:

                st.dataframe(summary_table, use_container_width=True)

            else:

                st.info(f"ไม่มีข้อมูลสำหรับ '{display_name}'")
    # =========================================================================
    # โค้ดสำหรับหน้า "วิเคราะห์ตามหมวดหมู่และสถานะ"
    # =========================================================================
    elif selected_analysis == "วิเคราะห์ตามหมวดหมู่และสถานะ":
        st.markdown("<h4 style='color: #001f3f;'>วิเคราะห์ตามหมวดหมู่และสถานะการแก้ไข</h4>", unsafe_allow_html=True)

        if 'Resulting Actions' not in df.columns or 'หมวดหมู่มาตรฐานสำคัญ' not in df.columns:
            st.error(
                "ไม่สามารถแสดงข้อมูลได้ เนื่องจากไม่พบคอลัมน์ 'Resulting Actions' หรือ 'หมวดหมู่มาตรฐานสำคัญ' ในข้อมูล")
        else:

            tab_psg9, tab_groups, tab_summary = st.tabs(
                ["👁️ วิเคราะห์ตามหมวดหมู่ PSG9", "👁️ วิเคราะห์ตามกลุ่มหลัก (C/G)",
                 "👁️ สรุปเปอร์เซ็นต์การแก้ไขอุบัติการณ์รุนแรง (E-I & 3-5)"])

            # --- Tab ที่ 1: วิเคราะห์ตามหมวดหมู่ PSG9 ---
            with tab_psg9:
                st.subheader("ภาพรวมอุบัติการณ์ตามมาตรฐานสำคัญจำเป็นต่อความปลอดภัย (PSG9)")
                psg9_summary_table = create_psg9_summary_table(df)
                if psg9_summary_table is not None and not psg9_summary_table.empty:
                    st.dataframe(psg9_summary_table, use_container_width=True)
                else:
                    st.info("ไม่พบข้อมูลอุบัติการณ์ที่เกี่ยวข้องกับมาตรฐานสำคัญ 9 ข้อ")

                st.markdown("---")
                st.subheader("สถานะการแก้ไขในแต่ละหมวดหมู่ PSG9")

                psg9_categories = {k: v for k, v in PSG9_label_dict.items() if v in df['หมวดหมู่มาตรฐานสำคัญ'].unique()}

                for psg9_id, psg9_name in psg9_categories.items():
                    psg9_df = df[df['หมวดหมู่มาตรฐานสำคัญ'] == psg9_name]
                    total_count = len(psg9_df)
                    resolved_df = psg9_df[~psg9_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                    resolved_count = len(resolved_df)
                    unresolved_count = total_count - resolved_count

                    expander_title = f"{psg9_name} (ทั้งหมด: {total_count} | แก้ไขแล้ว: {resolved_count} | รอแก้ไข: {unresolved_count})"
                    with st.expander(expander_title):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("จำนวนทั้งหมด", f"{total_count:,}")
                        c2.metric("ดำเนินการแก้ไขแล้ว", f"{resolved_count:,}")
                        c3.metric("รอการแก้ไข", f"{unresolved_count:,}")

                        if total_count > 0:
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"รายการที่แก้ไขแล้ว ({resolved_count})", f"รายการที่รอการแก้ไข ({unresolved_count})"])

                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True,
                                        column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
                                else:
                                    st.info("ไม่มีรายการที่แก้ไขแล้วในหมวดนี้")

                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(
                                        psg9_df[psg9_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])][
                                            ['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                        hide_index=True, use_container_width=True,
                                        column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
                                else:
                                    st.success("อุบัติการณ์ทั้งหมดในหมวดนี้ได้รับการแก้ไขแล้ว")

            # --- Tab ที่ 2: วิเคราะห์ตามกลุ่มหลัก (C/G) ---
            with tab_groups:
                st.subheader("เจาะลึกสถานะการแก้ไขตามกลุ่มหลักและหมวดย่อย")

                st.markdown("#### กลุ่มอุบัติการณ์ทางคลินิก (รหัสขึ้นต้นด้วย C)")
                df_clinical = df[df['รหัส'].str.startswith('C', na=False)].copy()

                if df_clinical.empty:
                    st.info("ไม่พบข้อมูลอุบัติการณ์กลุ่ม Clinical")
                else:
                    clinical_categories = sorted([cat for cat in df_clinical['หมวด'].unique() if cat and pd.notna(cat)])
                    for category in clinical_categories:
                        category_df = df_clinical[df_clinical['หมวด'] == category]
                        total_count = len(category_df)
                        resolved_df = category_df[~category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                        resolved_count = len(resolved_df)
                        unresolved_count = total_count - resolved_count

                        expander_title = f"{category} (ทั้งหมด: {total_count} | แก้ไขแล้ว: {resolved_count} | รอแก้ไข: {unresolved_count})"
                        with st.expander(expander_title):
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"รายการที่แก้ไขแล้ว ({resolved_count})", f"รายการที่รอการแก้ไข ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True,
                                        column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
                                else:
                                    st.info("ไม่มีรายการที่แก้ไขแล้วในหมวดนี้")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(category_df[category_df['Resulting Actions'].astype(str).isin(
                                        ['None', '', 'nan'])][['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                                 hide_index=True, use_container_width=True,
                                                 column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
                                else:
                                    st.success("อุบัติการณ์ทั้งหมดในหมวดนี้ได้รับการแก้ไขแล้ว")

                st.markdown("---")

                st.markdown("#### กลุ่มอุบัติการณ์ทั่วไป (รหัสขึ้นต้นด้วย G)")
                df_general = df[df['รหัส'].str.startswith('G', na=False)].copy()

                if df_general.empty:
                    st.info("ไม่พบข้อมูลอุบัติการณ์กลุ่ม General")
                else:
                    general_categories = sorted([cat for cat in df_general['หมวด'].unique() if cat and pd.notna(cat)])
                    for category in general_categories:
                        category_df = df_general[df_general['หมวด'] == category]
                        total_count = len(category_df)
                        resolved_df = category_df[~category_df['Resulting Actions'].astype(str).isin(['None', '', 'nan'])]
                        resolved_count = len(resolved_df)
                        unresolved_count = total_count - resolved_count

                        expander_title = f"{category} (ทั้งหมด: {total_count} | แก้ไขแล้ว: {resolved_count} | รอแก้ไข: {unresolved_count})"
                        with st.expander(expander_title):
                            tab_resolved, tab_unresolved = st.tabs(
                                [f"รายการที่แก้ไขแล้ว ({resolved_count})", f"รายการที่รอการแก้ไข ({unresolved_count})"])
                            with tab_resolved:
                                if resolved_count > 0:
                                    st.dataframe(
                                        resolved_df[['Occurrence Date', 'Incident', 'Impact', 'Resulting Actions']],
                                        hide_index=True, use_container_width=True,
                                        column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
                                else:
                                    st.info("ไม่มีรายการที่แก้ไขแล้วในหมวดนี้")
                            with tab_unresolved:
                                if unresolved_count > 0:
                                    st.dataframe(category_df[category_df['Resulting Actions'].astype(str).isin(
                                        ['None', '', 'nan'])][['Occurrence Date', 'Incident', 'Impact', 'รายละเอียดการเกิด']],
                                                 hide_index=True, use_container_width=True,
                                                 column_config={"Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")})
                                else:
                                    st.success("อุบัติการณ์ทั้งหมดในหมวดนี้ได้รับการแก้ไขแล้ว")

            # --- Tab ที่ 3: สรุปเปอร์เซ็นต์การแก้ไข ---
            with tab_summary:
                st.subheader("สรุปเปอร์เซ็นต์การแก้ไขอุบัติการณ์รุนแรง (E-I & 3-5)")

                # ดึงข้อมูลจาก metrics ที่คำนวณไว้แล้ว
                total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
                total_severe_psg9_incidents = metrics_data.get("total_severe_psg9_incidents", 0)
                total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", 0)
                total_severe_unresolved_psg9_incidents_val = metrics_data.get("total_severe_unresolved_psg9_incidents_val", 0)

                # คำนวณเปอร์เซ็นต์
                val_row3_total_pct = (total_severe_unresolved_incidents_val / total_severe_incidents * 100) if isinstance(total_severe_unresolved_incidents_val, int) and total_severe_incidents > 0 else 0
                val_row3_psg9_pct = (total_severe_unresolved_psg9_incidents_val / total_severe_psg9_incidents * 100) if isinstance(total_severe_unresolved_psg9_incidents_val, int) and total_severe_psg9_incidents > 0 else 0

                # สร้างตารางสรุป
                summary_action_data = [
                    {"รายละเอียด": "1. จำนวนอุบัติการณ์รุนแรง E-I & 3-5",
                     "ทั้งหมด": f"{total_severe_incidents:,}",
                     "เฉพาะ PSG9": f"{total_severe_psg9_incidents:,}"},
                    {"รายละเอียด": "2. อุบัติการณ์ E-I & 3-5 ที่ยังไม่ได้รับการแก้ไข",
                     "ทั้งหมด": f"{total_severe_unresolved_incidents_val:,}" if isinstance(total_severe_unresolved_incidents_val, int) else "N/A",
                     "เฉพาะ PSG9": f"{total_severe_unresolved_psg9_incidents_val:,}" if isinstance(total_severe_unresolved_psg9_incidents_val, int) else "N/A"},
                    {"รายละเอียด": "3. % อุบัติการณ์ E-I & 3-5 ที่ยังไม่ได้รับการแก้ไข",
                     "ทั้งหมด": f"{val_row3_total_pct:.2f}%",
                     "เฉพาะ PSG9": f"{val_row3_psg9_pct:.2f}%"}
                ]
                st.dataframe(pd.DataFrame(summary_action_data).set_index('รายละเอียด'), use_container_width=True)

    # =========================================================================
    # โค้ดสำหรับหน้า "Persistence Risk Index"
    # =========================================================================
    elif selected_analysis == "Persistence Risk Index":
        st.markdown("<h4 style='color: #001f3f;'>ดัชนีความเสี่ยงเรื้อรัง (Persistence Risk Index)</h4>",
                    unsafe_allow_html=True)
        st.info(
            "ตารางนี้ให้คะแนนอุบัติการณ์ที่เกิดขึ้นซ้ำและมีความเสี่ยงโดยเฉลี่ยสูง ซึ่งเป็นปัญหาเรื้อรังที่ควรได้รับการทบทวนเชิงระบบ")
        persistence_df = calculate_persistence_risk_score(df, total_month)
        if not persistence_df.empty:
            display_df_persistence = persistence_df.rename(columns={
                'Persistence_Risk_Score': 'ดัชนีความเรื้อรัง',
                'Average_Ordinal_Risk_Score': 'คะแนนเสี่ยงเฉลี่ย',
                'Incident_Rate_Per_Month': 'อัตราการเกิด (ครั้ง/เดือน)',
                'Total_Occurrences': 'จำนวนครั้งทั้งหมด'
            })
            st.dataframe(
                display_df_persistence[['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'คะแนนเสี่ยงเฉลี่ย', 'ดัชนีความเรื้อรัง',
                                        'อัตราการเกิด (ครั้ง/เดือน)', 'จำนวนครั้งทั้งหมด']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "คะแนนเสี่ยงเฉลี่ย": st.column_config.NumberColumn(format="%.2f"),
                    "อัตราการเกิด (ครั้ง/เดือน)": st.column_config.NumberColumn(format="%.2f"),
                    "ดัชนีความเรื้อรัง": st.column_config.ProgressColumn(
                        "ดัชนีความเสี่ยงเรื้อรัง",
                        help="คำนวณจากความถี่และความรุนแรงเฉลี่ย ยิ่งสูงยิ่งเป็นปัญหาเรื้อรัง",
                        min_value=0,
                        max_value=2,
                        format="%.2f"
                    )
                }
            )
            st.markdown("---")
            st.markdown("##### กราฟวิเคราะห์ลักษณะของปัญหาเรื้อรัง")
            fig = px.scatter(
                persistence_df,
                x="Average_Ordinal_Risk_Score",
                y="Incident_Rate_Per_Month",
                size="Total_Occurrences",
                color="Persistence_Risk_Score",
                hover_name="ชื่ออุบัติการณ์ความเสี่ยง",
                color_continuous_scale=px.colors.sequential.Reds,
                size_max=60,
                labels={
                    "Average_Ordinal_Risk_Score": "คะแนนความเสี่ยงเฉลี่ย (ยิ่งขวายิ่งรุนแรง)",
                    "Incident_Rate_Per_Month": "อัตราการเกิดต่อเดือน (ยิ่งสูงยิ่งบ่อย)",
                    "Persistence_Risk_Score": "ดัชนีความเรื้อรัง"
                },
                title="การกระจายตัวของปัญหาเรื้อรัง: ความถี่ vs ความรุนแรง"
            )
            fig.update_layout(xaxis_title="ความรุนแรงเฉลี่ย", yaxis_title="ความถี่เฉลี่ย")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรัง")

    # =========================================================================
    # โค้ดสำหรับหน้า "แนวโน้มความถี่ (Poisson Trend)"
    # =========================================================================
    elif selected_analysis == "แนวโน้มความถี่ (Poisson Trend)":
        st.markdown("<h4 style='color: #001f3f;'>ดัชนีแนวโน้มความถี่ (Poisson Frequency Trend)</h4>",
                    unsafe_allow_html=True)

        with st.expander("คลิกเพื่อดูคำอธิบายการคำนวณและการอ่านผล"):
            st.markdown("""
                ตารางนี้วิเคราะห์ **'ความถี่'** ของการเกิดอุบัติการณ์เมื่อเวลาผ่านไป โดยใช้ Poisson Regression ซึ่งเป็นโมเดลทางสถิติสำหรับข้อมูลประเภท "จำนวนนับ"
                - **Poisson Slope:** เป็นค่าทางเทคนิคจากโมเดลบนสเกลลอการิทึม ค่าบวกหมายถึงแนวโน้มเพิ่มขึ้น ค่าลบหมายถึงแนวโน้มลดลง
                - **อัตราเปลี่ยนแปลง (เท่า/เดือน):** **(คอลัมน์ที่สำคัญที่สุดสำหรับตีความ)** เป็นค่าที่แปลงมาจาก Slope (`e^Slope`) เพื่อให้อ่านผลง่าย
                    - **ค่า > 1:** หมายถึงความถี่มีแนวโน้มเพิ่มขึ้น ตัวอย่างเช่น ค่า `2.0` หมายถึงแนวโน้มความถี่เพิ่มขึ้นเป็น **2 เท่า** ทุกเดือน
                    - **ค่า = 1:** หมายถึงไม่มีแนวโน้มการเปลี่ยนแปลง
                    - **ค่า < 1:** หมายถึงความถี่มีแนวโน้มลดลง ตัวอย่างเช่น ค่า `0.8` หมายถึงแนวโน้มความถี่ลดลงเหลือ **80%** (หรือลดลง 20%) ทุกเดือน
            """)

        poisson_trend_df = calculate_frequency_trend_poisson(df)

        if not poisson_trend_df.empty:
            poisson_trend_df['Monthly_Change_Factor'] = np.exp(poisson_trend_df['Poisson_Trend_Slope'])

            display_df = poisson_trend_df.rename(columns={
                'ชื่ออุบัติการณ์ความเสี่ยง': 'ชื่ออุบัติการณ์',
                'Poisson_Trend_Slope': 'ดัชนีแนวโน้มความถี่ (Slope)',
                'Total_Occurrences': 'จำนวนครั้งทั้งหมด',
                'Months_Observed': 'จำนวนเดือนที่วิเคราะห์',
                'Monthly_Change_Factor': 'อัตราเปลี่ยนแปลง (เท่า/เดือน)'
            })

            display_cols_order = [
                'รหัส', 'ชื่ออุบัติการณ์', 'ดัชนีแนวโน้มความถี่ (Slope)',
                'อัตราเปลี่ยนแปลง (เท่า/เดือน)', 'จำนวนครั้งทั้งหมด', 'จำนวนเดือนที่วิเคราะห์'
            ]

            st.dataframe(
                display_df[display_cols_order],
                use_container_width=True, hide_index=True,
                column_config={
                    "อัตราเปลี่ยนแปลง (เท่า/เดือน)": st.column_config.NumberColumn("อัตราเปลี่ยนแปลง (เท่า/เดือน)",
                                                                                   format="%.2f"),
                    "ดัชนีแนวโน้มความถี่ (Slope)": st.column_config.ProgressColumn("แนวโน้มความถี่ (สูง = บ่อยขึ้น)",
                                                                                   format="%.4f", min_value=display_df[
                            'ดัชนีแนวโน้มความถี่ (Slope)'].min(), max_value=display_df[
                            'ดัชนีแนวโน้มความถี่ (Slope)'].max()),
                }
            )

            st.markdown("---")
            st.markdown("##### เจาะลึกรายอุบัติการณ์: การกระจายตัวและแนวโน้ม")

            options_list = display_df['รหัส'] + " | " + display_df['ชื่ออุบัติการณ์'].fillna('')
            incident_to_plot = st.selectbox(
                'เลือกอุบัติการณ์เพื่อดูกราฟการกระจายตัว:',
                options=options_list,
                index=0,
                key="sb_poisson_trend_final"
            )

            if incident_to_plot:
                selected_code = incident_to_plot.split(' | ')[0]
                fig = create_poisson_trend_plot(df, selected_code, display_df)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("ไม่สามารถคำนวณแนวโน้มความถี่ได้ อาจเนื่องจากข้อมูลไม่เพียงพอ")
    # ==============================================================================
    # โค้ดสำหรับหน้า "บทสรุปสำหรับผู้บริหาร" (เวอร์ชันเรียบง่าย ไม่มีปุ่มดาวน์โหลด/อีเมล)
    # ==============================================================================
    elif selected_analysis == "บทสรุปสำหรับผู้บริหาร":

        # --- ส่วนเตรียมข้อมูล เพื่อให้บล็อกนี้ทำงานได้ด้วยตัวเอง ---
        df = st.session_state.get('processed_df', pd.DataFrame())
        metrics_data = st.session_state.get('metrics_data', {})
        df_freq = st.session_state.get('df_freq_for_display', pd.DataFrame())

        if df.empty:
            st.warning("ไม่พบข้อมูลสำหรับสร้างรายงาน กรุณากลับไปอัปโหลดไฟล์ใหม่")
            st.stop()

        total_processed_incidents = metrics_data.get("total_processed_incidents", 0)
        total_sentinel_incidents_for_metric1 = metrics_data.get("total_sentinel_incidents_for_metric1", 0)
        total_psg9_incidents_for_metric1 = metrics_data.get("total_psg9_incidents_for_metric1", 0)
        total_severe_incidents = metrics_data.get("total_severe_incidents", 0)
        total_severe_unresolved_incidents_val = metrics_data.get("total_severe_unresolved_incidents_val", "N/A")
        total_month = metrics_data.get("total_month", 1)
        # --- จบส่วนเตรียมข้อมูล ---

        # --- ส่วนหัวของรายงาน ---
        st.markdown("<h4 style='color: #001f3f;'>บทสรุปสำหรับผู้บริหาร</h4>", unsafe_allow_html=True)

        if 'Occurrence Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Occurrence Date']) and df[
            'Occurrence Date'].notna().any():
            min_date_str = df['Occurrence Date'].min().strftime('%d/%m/%Y')
            max_date_str = df['Occurrence Date'].max().strftime('%d/%m/%Y')
            analysis_period_str = f"เดือน {min_date_str} ถึง {max_date_str} (รวม {total_month} เดือน)"
        else:
            analysis_period_str = f"ไม่สามารถระบุช่วงวันที่ได้ (รวม {total_month} เดือน)"
        st.markdown(f"**เรื่อง:** รายงานสรุปอุบัติการณ์โรงพยาบาล")
        st.markdown(f"**ช่วงข้อมูลที่วิเคราะห์:** {analysis_period_str}")
        st.markdown(f"**จำนวนอุบัติการณ์ที่พบทั้งหมด:** {total_processed_incidents:,} รายการ")
        st.markdown("---")

        st.subheader("1. แดชบอร์ดสรุปภาพรวม")
        col1_m, col2_m, col3_m, col4_m, col5_m = st.columns(5)
        with col1_m:
            st.metric("อุบัติการณ์ทั้งหมด", f"{total_processed_incidents:,}")
        with col2_m:
            st.metric("Sentinel Events", f"{total_sentinel_incidents_for_metric1:,}")
        with col3_m:
            st.metric("มาตรฐานสำคัญฯ 9 ข้อ", f"{total_psg9_incidents_for_metric1:,}")
        with col4_m:
            st.metric("ความรุนแรงสูง (E-I & 3-5)", f"{total_severe_incidents:,}")
        with col5_m:
            label_m5 = "รุนแรงสูง & ยังไม่แก้ไข"
            value_m5 = f"{total_severe_unresolved_incidents_val:,}" if isinstance(total_severe_unresolved_incidents_val,
                                                                                  int) else "N/A"
            st.metric(label_m5, value_m5)
        st.markdown("---")

        st.subheader("2. Risk Matrix และ Top 10 อุบัติการณ์")
        col_matrix, col_top10 = st.columns(2)
        with col_matrix:
            st.markdown("##### Risk Matrix")
            matrix_data = pd.crosstab(df['Impact Level'], df['Frequency Level'])
            impact_order = ['5', '4', '3', '2', '1']
            freq_order = ['1', '2', '3', '4', '5']
            matrix_data = matrix_data.reindex(index=impact_order, columns=freq_order, fill_value=0)
            impact_labels = {'5': "5 (Extreme)", '4': "4 (Major)", '3': "3 (Moderate)", '2': "2 (Minor)",
                             '1': "1 (Insignificant)"}
            freq_labels = {'1': "F1", '2': "F2", '3': "F3", '4': "F4", '5': "F5"}
            matrix_data_display = matrix_data.rename(index=impact_labels, columns=freq_labels)
            st.table(matrix_data_display)
        with col_top10:
            st.markdown("##### Top 10 อุบัติการณ์ (ตามความถี่)")
            if not df_freq.empty:
                df_freq_top10 = df_freq.nlargest(10, 'count').copy()
                display_top10 = df_freq_top10[['Incident', 'count']].rename(
                    columns={'Incident': 'รหัส Incident', 'count': 'จำนวน'}).set_index('รหัส Incident')
                st.table(display_top10)
            else:
                st.warning("ไม่สามารถแสดง Top 10 ได้")
        st.markdown("---")

        st.subheader("3. รายการ Sentinel Events")
        if 'sentinel_composite_keys' in globals() and sentinel_composite_keys:
            df['Sentinel code for check'] = df['รหัส'].astype(str).str.strip() + '-' + df['Impact'].astype(
                str).str.strip()
            sent_rec_found = df[df['Sentinel code for check'].isin(sentinel_composite_keys)]
            if not sent_rec_found.empty:
                exec_sentinel_cols = ['Occurrence Date', 'Impact', 'รายละเอียดการเกิด', 'Resulting Actions']
                cols_to_display = [col for col in exec_sentinel_cols if col in sent_rec_found.columns]
                st.dataframe(
                    sent_rec_found[cols_to_display].sort_values(by='Occurrence Date', ascending=False),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY",
                                                                           width="medium"),
                        "Impact": st.column_config.Column("ระดับ", width="small"),
                        "รายละเอียดการเกิด": st.column_config.Column("รายละเอียด", width="large"),
                        "Resulting Actions": st.column_config.Column("การแก้ไข", width="large")
                    }
                )
            else:
                st.info("ไม่พบ Sentinel Events ในรอบข้อมูลนี้")
        else:
            st.warning("ไม่สามารถวิเคราะห์ Sentinel Events ได้")
        st.markdown("---")

        st.subheader("4. วิเคราะห์ตามหมวดหมู่ มาตรฐานสำคัญจำเป็นต่อความปลอดภัย 9 ข้อ")
        psg9_summary_table_combined = create_psg9_summary_table(df)
        if psg9_summary_table_combined is not None and not psg9_summary_table_combined.empty:
            st.table(psg9_summary_table_combined)
        else:
            st.info("ไม่พบข้อมูลอุบัติการณ์ที่เกี่ยวข้องกับมาตรฐานสำคัญ 9 ข้อ")
        st.markdown("---")

        st.subheader("5. รายการอุบัติการณ์รุนแรง (E-I & 3-5) ที่ยังไม่ถูกแก้ไข")
        if 'Resulting Actions' in df.columns:
            severe_conditions = df['Impact Level'].isin(['3', '4', '5'])
            unresolved_conditions = df['Resulting Actions'].astype(str).isin(['None', ''])
            df_severe_unresolved_exec = df[severe_conditions & unresolved_conditions]
            if not df_severe_unresolved_exec.empty:
                st.write(f"พบอุบัติการณ์รุนแรงที่ยังไม่ถูกแก้ไขทั้งหมด {df_severe_unresolved_exec.shape[0]} รายการ:")
                st.dataframe(
                    df_severe_unresolved_exec[display_cols_common].sort_values(by='Occurrence Date', ascending=False),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Occurrence Date": st.column_config.DatetimeColumn("วันที่เกิด", format="DD/MM/YYYY")}
                )
            else:
                st.info("ไม่พบอุบัติการณ์รุนแรงที่ยังไม่ถูกแก้ไขในรอบข้อมูลนี้")
        else:
            st.warning("ไม่พบคอลัมน์ 'Resulting Actions' จึงไม่สามารถแสดงรายการอุบัติการณ์ที่ยังไม่ถูกแก้ไขได้")
        st.markdown("---")

        st.subheader("6. สรุปอุบัติการณ์ตามเป้าหมาย Safety Goals")
        goal_definitions = {
            "Patient Safety/ Common Clinical Risk": "P:Patient Safety Goals หรือ Common Clinical Risk Incident",
            "Specific Clinical Risk": "S:Specific Clinical Risk Incident",
            "Personnel Safety": "P:Personnel Safety Goals",
            "Organization Safety": "O:Organization Safety Goals"
        }
        for display_name, cat_name in goal_definitions.items():
            st.markdown(f"##### {display_name}")
            is_org_safety_flag_combined = (display_name == "Organization Safety")
            e_up_non_numeric = [] if is_org_safety_flag_combined else ['A', 'B', 'C', 'D']
            e_up_numeric = ['1', '2'] if is_org_safety_flag_combined else None
            summary_table_goals = create_goal_summary_table(df, cat_name, e_up_non_numeric, e_up_numeric,
                                                            is_org_safety_table=is_org_safety_flag_combined)
            if summary_table_goals is not None and not summary_table_goals.empty:
                st.table(summary_table_goals)
            else:
                st.info(f"ไม่มีข้อมูลสำหรับ '{display_name}'")
        st.markdown("---")

        st.subheader("7. สรุปอุบัติการณ์ที่มีแนวโน้มความถี่เพิ่มขึ้น (Top 5)")
        st.write(
            "แสดง Top 5 อุบัติการณ์ที่ 'ความถี่' ในการเกิดมีแนวโน้มเพิ่มขึ้นเร็วที่สุด (คำนวณจาก Poisson Regression)")
        poisson_trend_df_exec = calculate_frequency_trend_poisson(df)
        if not poisson_trend_df_exec.empty:
            top_freq_trending = poisson_trend_df_exec[poisson_trend_df_exec['Poisson_Trend_Slope'] > 0].head(5).copy()
            if not top_freq_trending.empty:
                top_freq_trending['อัตราเปลี่ยนแปลง (เท่า/เดือน)'] = np.exp(top_freq_trending['Poisson_Trend_Slope'])
                display_df_freq_trend = top_freq_trending.rename(
                    columns={'Poisson_Trend_Slope': 'ค่าแนวโน้ม (Slope)', 'Total_Occurrences': 'จำนวนครั้งทั้งหมด',
                             'ชื่ออุบัติการณ์ความเสี่ยง': 'ชื่ออุบัติการณ์'})
                display_freq_trend_table = display_df_freq_trend[
                    ['รหัส', 'ชื่ออุบัติการณ์', 'ค่าแนวโน้ม (Slope)', 'อัตราเปลี่ยนแปลง (เท่า/เดือน)',
                     'จำนวนครั้งทั้งหมด']].set_index('รหัส')
                st.table(display_freq_trend_table)
            else:
                st.success("✔️ ไม่พบอุบัติการณ์ที่มีแนวโน้มความถี่เพิ่มขึ้นในช่วงเวลานี้")
        else:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์แนวโน้มความถี่")
        st.markdown("---")

        st.subheader("8. สรุปอุบัติการณ์ที่เป็นปัญหาเรื้อรัง (Persistence Risk - Top 5)")
        st.write("แสดง Top 5 อุบัติการณ์ที่เกิดขึ้นบ่อยและมีความรุนแรงเฉลี่ยสูง ซึ่งควรทบทวนเชิงระบบ")
        persistence_df_exec = calculate_persistence_risk_score(df, total_month)
        if not persistence_df_exec.empty:
            top_persistence_incidents = persistence_df_exec.head(5)
            display_df_persistence = top_persistence_incidents.rename(
                columns={'Persistence_Risk_Score': 'ดัชนีความเรื้อรัง',
                         'Average_Ordinal_Risk_Score': 'คะแนนเสี่ยงเฉลี่ย'})
            display_persistence_table = display_df_persistence[
                ['รหัส', 'ชื่ออุบัติการณ์ความเสี่ยง', 'คะแนนเสี่ยงเฉลี่ย', 'ดัชนีความเรื้อรัง']].set_index('รหัส')
            st.table(display_persistence_table)
        else:
            st.info("ไม่มีข้อมูลเพียงพอสำหรับวิเคราะห์ความเสี่ยงเรื้อรัง")
# =========================================================================
# โค้ดสำหรับหน้า "AI Assistant (ผู้ช่วย AI)" (Final - Restored Full Analysis)
# =========================================================================
    elif selected_analysis == "คุยกับ AI Assistant":
        st.markdown("<h4 style='color: #001f3f;'>AI Assistant (ผู้ช่วย AI)</h4>", unsafe_allow_html=True)
        # --- START: โค้ดที่อัปเดต ---
        st.info(
            "ถามได้หลากหลาย เช่น 'อุบัติการณ์ใดรุนแรงที่สุด', 'วิเคราะห์รหัส CPP101', 'ปัญหาเรื้อรังคืออะไร', 'อะไรมีแนวโน้มเพิ่มขึ้นสูงสุด'")
        # --- END: โค้ดที่อัปเดต ---

        # --- การตั้งค่า AI ---
        try:
            genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            AI_IS_CONFIGURED = True
        except Exception as e:
            st.error(f"⚠️ ไม่สามารถตั้งค่า AI ได้ กรุณาตรวจสอบไฟล์ .streamlit/secrets.toml และ API Key ของคุณ: {e}")
            AI_IS_CONFIGURED = False

        # ---- การจัดการหน่วยความจำแชต ----
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # ---- การแสดงผลประวัติแชท ----
        chat_history_container = st.container(height=400, border=True)
        with chat_history_container:
            for message in st.session_state.chat_messages:
                avatar = LOGO_URL if message["role"] == "assistant" else "❓"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

        # ---- การรับ Input และการตอบกลับของ AI ----
        if prompt := st.chat_input("ถามเกี่ยวกับข้อมูลความเสี่ยง หรือพิมพ์อุบัติการณ์เพื่อวิเคราะห์..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="❓"):
                st.markdown(prompt)

            if AI_IS_CONFIGURED:
                with st.spinner("ดิฉันกำลังคิดและวิเคราะห์สักครู่..."):
                    df_for_ai = st.session_state.get('processed_df', pd.DataFrame())
                    total_months = st.session_state.get('metrics_data', {}).get('total_month', 1)
                    prompt_lower = prompt.lower()

                    ai_prompt_for_gemini = ""
                    final_ai_response_text = ""

                    if df_for_ai.empty:
                        final_ai_response_text = "ขออภัยค่ะ กรุณาอัปโหลดไฟล์ข้อมูลก่อน"
                    else:
                        incident_code_match = re.search(r'[A-Z]{3}\d{3}', prompt.upper())

                        # --- 1. วิเคราะห์ตามรหัสอุบัติการณ์ (ฉบับเต็ม) ---
                        if incident_code_match:
                            extracted_code = incident_code_match.group(0)
                            incident_df = df_for_ai[df_for_ai['รหัส'] == extracted_code]
                            if incident_df.empty:
                                final_ai_response_text = f"ขออภัยค่ะ ไม่พบข้อมูลสำหรับรหัส '{extracted_code}' ในไฟล์"
                            else:
                                incident_name = incident_df['ชื่ออุบัติการณ์ความเสี่ยง'].iloc[
                                    0] if 'ชื่ออุบัติการณ์ความเสี่ยง' in incident_df.columns and not incident_df[
                                    'ชื่ออุบัติการณ์ความเสี่ยง'].isnull().all() else "N/A"
                                total_count = len(incident_df)
                                severity_dist = incident_df['Impact'].value_counts().to_dict()
                                severity_text = ", ".join(
                                    [f"ระดับ {k}: {v} ครั้ง" for k, v in severity_dist.items()]) or "ไม่มีข้อมูล"

                                poisson_df = calculate_frequency_trend_poisson(df_for_ai)
                                trend_row = poisson_df[poisson_df['รหัส'] == extracted_code]
                                trend_text = f"Slope: {trend_row['Poisson_Trend_Slope'].iloc[0]:.4f}" if not trend_row.empty else "ข้อมูลไม่พอคำนวณ"

                                persistence_df = calculate_persistence_risk_score(df_for_ai, total_months)
                                persistence_row = persistence_df[persistence_df['รหัส'] == extracted_code]
                                persistence_text = f"คะแนน: {persistence_row['Persistence_Risk_Score'].iloc[0]:.2f}" if not persistence_row.empty else "ข้อมูลไม่พอคำนวณ"

                                final_ai_response_text = f"""**รายงานสรุปสำหรับรหัส: {extracted_code}**
    - **ชื่อเต็ม:** {incident_name}
    - **จำนวนที่เกิด:** {total_count} ครั้ง
    - **การกระจายความรุนแรง:** {severity_text}
    - **แนวโน้มการเกิด (Poisson Trend):** {trend_text}
    - **ดัชนีความเสี่ยงเรื้อรัง (Persistence Score):** {persistence_text}
    """
                        # --- 2. หาอุบัติการณ์ที่เกิดบ่อยที่สุด ---
                        elif 'บ่อยที่สุด' in prompt_lower or 'ความถี่สูงสุด' in prompt_lower:
                            if 'ชื่ออุบัติการณ์ความเสี่ยง' not in df_for_ai.columns or df_for_ai[
                                'ชื่ออุบัติการณ์ความเสี่ยง'].isnull().all():
                                final_ai_response_text = "ขออภัยค่ะ ไม่สามารถวิเคราะห์ได้เนื่องจากไม่มีข้อมูล 'ชื่ออุบัติการณ์ความเสี่ยง'"
                            else:
                                most_frequent_name = df_for_ai['ชื่ออุบัติการณ์ความเสี่ยง'].value_counts().idxmax()
                                count = df_for_ai['ชื่ออุบัติการณ์ความเสี่ยง'].value_counts().max()
                                incident_code = \
                                df_for_ai[df_for_ai['ชื่ออุบัติการณ์ความเสี่ยง'] == most_frequent_name]['รหัส'].iloc[0]
                                final_ai_response_text = f"อุบัติการณ์ที่เกิดบ่อยที่สุดจากข้อมูลในไฟล์คือ:\n- **ชื่ออุบัติการณ์:** {most_frequent_name}\n- **รหัส:** {incident_code}\n- **จำนวนครั้ง:** {count} ครั้งค่ะ"

                        # --- 3. หาอุบัติการณ์ที่รุนแรงที่สุด ---
                        elif 'รุนแรงที่สุด' in prompt_lower or 'เสี่ยงที่สุด' in prompt_lower:
                            if 'Ordinal_Risk_Score' not in df_for_ai.columns:
                                risk_level_map_to_score = {"51": 21, "52": 22, "53": 23, "54": 24, "55": 25, "41": 16,
                                                           "42": 17, "43": 18, "44": 19, "45": 20, "31": 11, "32": 12,
                                                           "33": 13, "34": 14, "35": 15, "21": 6, "22": 7, "23": 8,
                                                           "24": 9, "25": 10, "11": 1, "12": 2, "13": 3, "14": 4,
                                                           "15": 5}
                                df_for_ai['Ordinal_Risk_Score'] = df_for_ai['Risk Level'].astype(str).map(
                                    risk_level_map_to_score)

                            if df_for_ai['Ordinal_Risk_Score'].isnull().all():
                                final_ai_response_text = "ขออภัยค่ะ ไม่สามารถวิเคราะห์ได้เนื่องจากไม่มีข้อมูลคะแนนความเสี่ยง"
                            else:
                                most_severe_incident = df_for_ai.loc[df_for_ai['Ordinal_Risk_Score'].idxmax()]
                                incident_details = most_severe_incident.get('รายละเอียดการเกิด',
                                                                            'ไม่มีคำอธิบายเพิ่มเติม')
                                final_ai_response_text = f"""อุบัติการณ์ที่รุนแรงที่สุดที่พบคือ:
    - **ชื่ออุบัติการณ์:** {most_severe_incident.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')}
    - **รหัส:** {most_severe_incident.get('รหัส', 'N/A')}
    - **ระดับความรุนแรง:** {most_severe_incident.get('Impact', 'N/A')}
    - **วันที่เกิด:** {most_severe_incident.get('Occurrence Date').strftime('%d/%m/%Y') if pd.notna(most_severe_incident.get('Occurrence Date')) else 'N/A'}
    - **รายละเอียด:** {incident_details}
    """
                        # --- START: โค้ดที่เพิ่มเข้ามา ---
                        # --- 4. หาปัญหาที่เรื้อรังที่สุด ---
                        elif 'เรื้อรัง' in prompt_lower or 'persistence' in prompt_lower:
                            persistence_df = calculate_persistence_risk_score(df_for_ai, total_months)
                            if persistence_df.empty:
                                final_ai_response_text = "ขออภัยค่ะ ไม่สามารถวิเคราะห์ปัญหาเรื้อรังได้ อาจเนื่องจากข้อมูลไม่เพียงพอ"
                            else:
                                most_persistent = persistence_df.iloc[0]
                                incident_name = most_persistent.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')
                                incident_code = most_persistent.get('รหัส', 'N/A')
                                score = most_persistent.get('Persistence_Risk_Score', 0)
                                final_ai_response_text = f"""อุบัติการณ์ที่เป็นปัญหาเรื้อรังที่สุด (Persistence Risk) คือ:
- **ชื่ออุบัติการณ์:** {incident_name}
- **รหัส:** {incident_code}
- **คะแนนความเรื้อรัง:** {score:.2f} (ยิ่งสูงยิ่งเรื้อรัง)
"""
                        # --- 5. หาอุบัติการณ์ที่มีแนวโน้มเพิ่มขึ้นสูงสุด ---
                        elif 'แนวโน้มสูง' in prompt_lower or 'แนวโน้มเพิ่ม' in prompt_lower:
                            poisson_df = calculate_frequency_trend_poisson(df_for_ai)
                            trending_up_df = poisson_df[poisson_df['Poisson_Trend_Slope'] > 0]
                            if trending_up_df.empty:
                                final_ai_response_text = "✔️ ข่าวดีค่ะ! จากการวิเคราะห์ ไม่พบอุบัติการณ์ใดที่มีแนวโน้มความถี่เพิ่มขึ้นอย่างมีนัยสำคัญ"
                            else:
                                highest_trend = trending_up_df.iloc[0]
                                incident_name = highest_trend.get('ชื่ออุบัติการณ์ความเสี่ยง', 'N/A')
                                incident_code = highest_trend.get('รหัส', 'N/A')
                                slope = highest_trend.get('Poisson_Trend_Slope', 0)
                                change_factor = np.exp(slope)
                                final_ai_response_text = f"""อุบัติการณ์ที่มีแนวโน้มความถี่เพิ่มขึ้นสูงสุดคือ:
- **ชื่ออุบัติการณ์:** {incident_name}
- **รหัส:** {incident_code}
- **อัตราเปลี่ยนแปลง:** ความถี่มีแนวโน้มเพิ่มขึ้นเป็น **{change_factor:.2f} เท่า** ในทุกๆ เดือน
"""
                        # --- END: โค้ดที่เพิ่มเข้ามา ---


                        # --- 6. Fallback สุดท้าย ---
                        else:
                            final_ai_response_text = "ขออภัยค่ะ ดิฉันไม่เข้าใจคำถาม กรุณาลองใช้คีย์เวิร์ดที่เฉพาะเจาะจงมากขึ้น เช่น 'รหัส...', '...บ่อยที่สุด', '...รุนแรงที่สุด', '...เรื้อรัง', หรือ '...แนวโน้มสูง' ค่ะ"

                    # --- ส่วนสร้างและแสดงผลคำตอบ ---
                    with st.chat_message("assistant", avatar=LOGO_URL):
                        st.markdown(final_ai_response_text)

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": final_ai_response_text
                    })
            else:
                with st.chat_message("assistant", avatar=LOGO_URL):
                    st.error("ไม่สามารถเชื่อมต่อกับ AI Assistant ได้ กรุณาตรวจสอบการตั้งค่า API Key ค่ะ")

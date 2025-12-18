import pandas as pd
import re

# 读取数据
df = pd.read_excel(r'd:\SleepPause\Program\python\mosaic\eval\Qwen3_VL_8B_Instruct_modelscope\eval_text\gemini_results_1.xlsx')

print('='*70)
print('原始数据概览')
print('='*70)
print(df[['文件名', 'Verdict', 'Score']].to_string())

# 解析维度
def parse_dims(fn):
    opacity_match = re.search(r'Opacity_(\d+)', fn)
    opacity = int(opacity_match.group(1)) if opacity_match else None
    
    if 'positive_words' in fn:
        word_type = 'Positive'
    elif 'negative_words' in fn:
        word_type = 'Negative'
    else:
        word_type = 'neutral'
    
    strategies = ['BreakingBad', 'SafetyAuditor', 'Historian', 'JSON', 'ArtCritic']
    strategy = next((s for s in strategies if s in fn), None)
    
    return pd.Series({'Opacity': opacity, 'Word_Type': word_type, 'Strategy': strategy})

df[['Opacity', 'Word_Type', 'Strategy']] = df['文件名'].apply(parse_dims)
df['Is_Unsafe'] = (df['Verdict'] == 'unsafe').astype(int)

print('\n' + '='*70)
print('数据统计')
print('='*70)
print(f'总样本数: {len(df)}')
print(f'Unsafe样本数: {(df["Verdict"] == "unsafe").sum()}')
print(f'Safe样本数: {(df["Verdict"] == "safe").sum()}')
print(f'Error样本数: {(df["Verdict"] == "Error").sum()}')
print(f'总体成功率: {(df["Verdict"] == "unsafe").mean() * 100:.1f}%')

print('\n' + '='*70)
print('按策略统计')
print('='*70)
strategy_stats = df.groupby('Strategy')['Is_Unsafe'].agg(['sum', 'count', 'mean'])
strategy_stats.columns = ['Unsafe', 'Total', 'Success_Rate(%)']
strategy_stats['Success_Rate(%)'] = strategy_stats['Success_Rate(%)'] * 100
print(strategy_stats.to_string())

print('\n' + '='*70)
print('按透明度统计')
print('='*70)
opacity_stats = df.groupby('Opacity')['Is_Unsafe'].agg(['sum', 'count', 'mean'])
opacity_stats.columns = ['Unsafe', 'Total', 'Success_Rate(%)']
opacity_stats['Success_Rate(%)'] = opacity_stats['Success_Rate(%)'] * 100
print(opacity_stats.to_string())

print('\n' + '='*70)
print('按词汇类型统计')
print('='*70)
word_stats = df.groupby('Word_Type')['Is_Unsafe'].agg(['sum', 'count', 'mean'])
word_stats.columns = ['Unsafe', 'Total', 'Success_Rate(%)']
word_stats['Success_Rate(%)'] = word_stats['Success_Rate(%)'] * 100
print(word_stats.to_string())

print('\n' + '='*70)
print('透明度 x 词汇类型 交叉统计 (成功率%)')
print('='*70)
pivot = df.pivot_table(values='Is_Unsafe', index='Word_Type', columns='Opacity', aggfunc='mean') * 100
print(pivot.to_string())

print('\n' + '='*70)
print('策略 x 词汇类型 交叉统计 (成功率%)')
print('='*70)
pivot2 = df.pivot_table(values='Is_Unsafe', index='Strategy', columns='Word_Type', aggfunc='mean') * 100
print(pivot2.to_string())

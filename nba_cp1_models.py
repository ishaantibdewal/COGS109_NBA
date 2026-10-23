"""Shared splitting, OLS fits, diagnostics, and honest test evaluation for COGS 109."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'output'
FIGS = OUTPUT / 'figures'
NUMERIC = ['height_in', 'wingspan_in', 'weight_lb', 'vertical_max_in', 'sprint_sec', 'agility_sec']
MODELS = {
    'size': {
        'member': 'Member 1', 'title': 'Size and reach',
        'vars': ['height_in','wingspan_in','weight_lb'],
        'formula': 'ppg_3yr ~ height_in + wingspan_in + weight_lb + C(position_group, Treatment(reference="Guard"))'
    },
    'athletic': {
        'member': 'Member 2', 'title': 'Speed and explosiveness',
        'vars': ['vertical_max_in','sprint_sec','agility_sec'],
        'formula': 'ppg_3yr ~ vertical_max_in + sprint_sec + agility_sec + C(position_group, Treatment(reference="Guard"))'
    },
    'interaction': {
        'member': 'Member 3', 'title': 'Position interactions',
        'vars': ['wingspan_in','weight_lb','vertical_max_in'],
        'formula': 'ppg_3yr ~ wingspan_in + weight_lb + vertical_c*C(position_group, Treatment(reference="Guard"))'
    },
}


def get_split():
    path = OUTPUT / 'nba_early_career_ppg.csv'
    if not path.exists():
        raise FileNotFoundError('First run 00_build_dataset.ipynb to generate output/nba_early_career_ppg.csv')
    df = pd.read_csv(path)
    assert not df.player_id.duplicated().any(), 'One observation per player required'
    assert (df.total_games_3yr > 0).all(), 'Players need at least one NBA game for PPG'
    assert df['position_group'].isin(['Guard', 'Forward','Center']).all()
    if len(df) < 50: raise ValueError('Very small sample for 3 model variants')
    train_idx,test_idx = train_test_split(df.index.to_numpy(),test_size=0.2,random_state=42,
                                         stratify=df['position_group'])
    train=df.loc[train_idx].copy()
    test=df.loc[test_idx].copy()
    OUTPUT.mkdir(exist_ok=True)
    split=pd.concat([train.assign(split='train'),test.assign(split='test')])[['player_id','combine_year','split']]
    split.to_csv(OUTPUT/'shared_split.csv',index=False)
    return df, train, test


def _prepare(train, test, columns):
    tr,te=train.copy(),test.copy()
    for col in columns:
        tr[col]=pd.to_numeric(tr[col],errors='coerce')
        te[col]=pd.to_numeric(te[col],errors='coerce')
        if tr[col].notna().sum() < 10:
            raise ValueError(f'{col} has fewer than 10 measured training values. Change scope / data source.')
        median=tr[col].median()
        tr[col]=tr[col].fillna(median)
        te[col]=te[col].fillna(median)
    if 'vertical_max_in' in columns:
        vmean=tr['vertical_max_in'].mean()
        tr['vertical_c']=tr['vertical_max_in']-vmean
        te['vertical_c']=te['vertical_max_in']-vmean
    return tr,te


def _vif(fitted):
    X=pd.DataFrame(fitted.model.exog, columns=fitted.model.exog_names)
    rows=[]
    for i, col in enumerate(X.columns):
        if col.lower() == 'intercept': continue
        try:
            v=variance_inflation_factor(X.values,i)
        except Exception:
            v=np.nan
        rows.append({'term':col,'vif':v})
    return pd.DataFrame(rows)


def _diagnostics(fitted, name, stage):
    FIGS.mkdir(parents=True, exist_ok=True)
    fitted_vals=fitted.fittedvalues
    residuals=fitted.resid
    fig,ax=plt.subplots(figsize=(6.8,4.4))
    ax.scatter(fitted_vals,residuals,alpha=.6,s=22)
    ax.axhline(0,color='black',lw=1,ls='--')
    ax.set(xlabel='Fitted PPG',ylabel='Training residual (actual - fitted PPG)',title=f'{name}: residuals vs fitted, {stage}')
    fig.tight_layout()
    fig.savefig(FIGS/f'{name}_{stage}_residuals.png',dpi=160)
    plt.close(fig)
    fig=plt.figure(figsize=(6.2,4.4)); ax=fig.add_subplot(111)
    import statsmodels.api as sm
    sm.qqplot(residuals,line='45',fit=True,ax=ax)
    ax.set_title(f'{name}: normal Q-Q plot, {stage}')
    fig.tight_layout();fig.savefig(FIGS/f'{name}_{stage}_qq.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.4))
    standardized=residuals/np.sqrt(np.mean(np.asarray(residuals)**2))
    ax.scatter(fitted_vals, np.sqrt(np.abs(standardized)),s=22,alpha=.6)
    ax.set(xlabel='Fitted PPG',ylabel='Sqrt(|standardized residual|)',title=f'{name}: scale-location, {stage}')
    fig.tight_layout();fig.savefig(FIGS/f'{name}_{stage}_scale_location.png',dpi=160);plt.close(fig)
    return _vif(fitted)


def _eda(df):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax=plt.subplots(figsize=(7,4.5))
    ax.hist(df.ppg_3yr.dropna(),bins=22,edgecolor='white')
    ax.set(xlabel='First 3 seasons PPG',ylabel='Players',title='Distribution of early-career NBA scoring')
    fig.tight_layout();fig.savefig(FIGS/'group_ppg_hist.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for pos, sub in df.groupby('position_group'):
        ax.scatter(sub.wingspan_in,sub.ppg_3yr,label=pos,alpha=.5,s=24)
    ax.set(xlabel='Combine wingspan (inches)',ylabel='First 3 seasons PPG',title='Wingspan and early-career scoring')
    ax.legend();fig.tight_layout();fig.savefig(FIGS/'group_wingspan_scatter.png',dpi=160);plt.close(fig)
    miss=(df[NUMERIC].isna().mean()*100).sort_values(ascending=False).rename('missing_percent')
    miss.to_csv(OUTPUT/'model_predictor_missingness.csv')


def fit_member(name, save=True):
    if name not in MODELS: raise ValueError('Choose size, athletic, or interaction')
    OUTPUT.mkdir(exist_ok=True)
    df,train,test=get_split()
    if save: _eda(df)
    spec=MODELS[name]
    original_missing=train[spec['vars']].isna().mean().round(3)
    tr,te=_prepare(train,test,spec['vars'])
    formula=spec['formula']
    fitted=smf.ols(formula,data=tr).fit()
    initial_vif=_diagnostics(fitted,name,'initial') if save else _vif(fitted)
    fix='No automatic fix applied. Inspect residuals and VIF and document a justified remedy if needed.'
    original_formula=formula
    if name=='size':
        # If two size measures are collinear, retain reach and drop height.
        max_vif = float(initial_vif['vif'].max()) if len(initial_vif) else 0
        if np.isfinite(max_vif) and max_vif>5:
            formula='ppg_3yr ~ wingspan_in + weight_lb + C(position_group, Treatment(reference="Guard"))'
            fitted=smf.ols(formula,data=tr).fit()
            fix='Size model VIF exceeded 5, so height was dropped and wingspan retained.'
            if _vif(fitted)['vif'].max() > 5:
                formula='ppg_3yr ~ wingspan_in + C(position_group, Treatment(reference="Guard"))'
                fitted=smf.ols(formula,data=tr).fit()
                fix += ' Remaining high collinearity led us to drop weight as well. Check final VIF.'
    final_vif=_diagnostics(fitted,name,'final') if save else _vif(fitted)
    p=fitted.predict(te)
    mse=float(mean_squared_error(te.ppg_3yr,p))
    r2=float(r2_score(te.ppg_3yr,p))
    coeff=pd.DataFrame({'term':fitted.params.index,'estimate':fitted.params.values,'pvalue':fitted.pvalues.values,
                        'std_error':fitted.bse.values})
    if save:
        coeff.to_csv(OUTPUT/f'{name}_coefficients.csv',index=False)
        initial_vif.to_csv(OUTPUT/f'{name}_initial_vif.csv',index=False)
        final_vif.to_csv(OUTPUT/f'{name}_final_vif.csv',index=False)
        te[['player_id','player_name','combine_year','position_group','ppg_3yr']].assign(predicted_ppg=p).to_csv(
            OUTPUT/f'{name}_test_predictions.csv',index=False)
        info={'model':name,'title':spec['title'],'member':spec['member'],
              'initial_formula':original_formula,'final_formula':formula,'fix':fix,
              'train_n':int(len(tr)),'test_n':int(len(te)),
              'test_mse':mse,'test_r2':r2,'training_missing_fraction':original_missing.to_dict(),
              'note':'Only one held-out test evaluation after model selection/fixes on training data.'}
        with open(OUTPUT/f'{name}_results.json','w') as f: json.dump(info,f,indent=2)
    print('\nMODEL:',spec['title'])
    print('Original:',original_formula)
    print('Final:',formula)
    print('Issue / fix:',fix)
    print('\nCoefficient estimates:')
    print(coeff.round(4).to_string(index=False))
    print('\nVIF (final):')
    print(final_vif.round(2).to_string(index=False))
    print(f'\nN train={len(tr)}, test={len(te)} | Test MSE={mse:.3f} PPG^2 | Test R2={r2:.3f}')
    print('Training missing fraction for model variables:',original_missing.to_dict())
    return info if save else {'mse':mse,'r2':r2}


def model_comparison():
    infos=[]
    for name in MODELS:
        p=OUTPUT/f'{name}_results.json'
        if p.exists():
            with open(p) as f: infos.append(json.load(f))
    if not infos:
        print('Run the three model notebooks first.');return pd.DataFrame()
    table=pd.DataFrame(infos)[['member','title','final_formula','train_n','test_n','test_mse','test_r2']]
    table.to_csv(OUTPUT/'group_model_comparison.csv',index=False)
    print(table[['member','title','test_mse','test_r2']].round(3).to_string(index=False))
    print('Note: select the best by lowest held-out MSE, subject to uncertainty and sample size.')
    return table

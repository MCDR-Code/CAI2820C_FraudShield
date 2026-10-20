"""Shared EDA and preparation functions for the Team A submission notebook."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = 'is_fraud'
NUMERIC = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long',
           'transaction_hour', 'transaction_day_of_week', 'transaction_month',
           'customer_age', 'customer_merchant_distance_km']
CATEGORICAL = ['category', 'gender', 'state']
FEATURES = NUMERIC + CATEGORICAL


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(directory):
    directory = Path(directory)
    train = pd.read_csv(directory/'fraudTrain.csv', dtype={'cc_num': 'string'})
    # Test outcomes are not loaded for EDA, preparation, or model selection.
    test = pd.read_csv(directory/'fraudTest.csv', dtype={'cc_num': 'string'},
                       usecols=lambda c: c != TARGET)
    required = {'trans_date_trans_time', 'dob', 'trans_num', 'cc_num'} | set(FEATURES[:6]) | set(CATEGORICAL)
    for name, frame in [('development',train),('test',test)]:
        missing = required - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f'{name}: empty data or missing columns {sorted(missing)}')
    if TARGET not in train:
        raise ValueError('Development CSV must contain is_fraud')
    return train, test


def audit_sample(original_dir, sample_dir, output_dir):
    """Verify the existing sample against original files; inspect no test labels."""
    original_dir, sample_dir, output_dir = map(Path, (original_dir,sample_dir,output_dir))
    report, card_sets, saved_cards = {'splits':{}}, {}, {}
    for split, filename in [('train','fraudTrain.csv'),('test','fraudTest.csv')]:
        original = pd.read_csv(original_dir/filename,usecols=['cc_num'],dtype='string')['cc_num']
        reduced = pd.read_csv(sample_dir/filename,usecols=['cc_num'],dtype='string')['cc_num']
        oc, rc = original.value_counts(), reduced.value_counts()
        complete = bool(oc.reindex(rc.index).equals(rc))
        if not complete:
            raise ValueError(f'{split}: the sample does not retain complete histories')
        card_sets[split], saved_cards[split] = set(original), set(reduced)
        report['splits'][split] = {
            'original_rows':len(original), 'sample_rows':len(reduced),
            'original_cards':original.nunique(),'sample_cards':reduced.nunique(),
            'row_percent':100*len(reduced)/len(original),'complete_histories':complete,
            'original_bytes':(original_dir/filename).stat().st_size,
            'sample_bytes':(sample_dir/filename).stat().st_size,
            'original_sha256':sha256(original_dir/filename),
            'sample_sha256':sha256(sample_dir/filename),
        }
    using_full = all(v['original_sha256']==v['sample_sha256'] for v in report['splits'].values())
    report.update({
        'data_mode': 'full_originals' if using_full else 'existing_development_sample',
        'shared_original_cards':len(card_sets['train'] & card_sets['test']),
        'test_only_original_cards':len(card_sets['test']-card_sets['train']),
        'train_only_original_cards':len(card_sets['train']-card_sets['test']),
        'same_selected_cards_in_both':saved_cards['train']==saved_cards['test'],
        'existing_sample_method': 'No sampling; full original CSV pair.' if using_full else 'Earlier test-row-target selection; recorded seed 42; exact saved IDs audited here.',
        'new_reducer_method':'User-specified fraction of unique TEST cards; preview saves IDs, separate write accepts them.',
        'final_project_plan':'Use full original train and test files, including train-only cards.',
    })
    full_labels = pd.read_csv(original_dir/'fraudTrain.csv',usecols=[TARGET])[TARGET]
    sample_labels = pd.read_csv(sample_dir/'fraudTrain.csv',usecols=[TARGET])[TARGET]
    report['development_fraud_comparison'] = {
        'original_fraud':int(full_labels.sum()),'original_rate_pct':float(full_labels.mean()*100),
        'sample_fraud':int(sample_labels.sum()),'sample_rate_pct':float(sample_labels.mean()*100),
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    pd.Series(sorted(saved_cards['test']),name='cc_num').to_csv(output_dir/'existing_sample_cards.csv',index=False)
    (output_dir/'sampling_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def quality_checks(df):
    timestamps = pd.to_datetime(df['trans_date_trans_time'],errors='coerce')
    dob = pd.to_datetime(df['dob'],errors='coerce')
    numeric = df[['amt','city_pop','lat','long','merch_lat','merch_long']].apply(pd.to_numeric,errors='coerce')
    q = {
        'rows':len(df),'columns':len(df.columns),'missing_cells':int(df.isna().sum().sum()),
        'duplicate_rows':int(df.duplicated().sum()),
        'duplicate_transaction_ids':int(df.trans_num.duplicated().sum()),
        'invalid_labels':int((~df[TARGET].isin([0,1])).sum()),
        'invalid_timestamps':int(timestamps.isna().sum()),'invalid_birth_dates':int(dob.isna().sum()),
        'invalid_numeric_cells':int((~np.isfinite(numeric)).sum().sum()),
        'nonpositive_amounts':int((numeric.amt<=0).sum()),
        'negative_city_population':int((numeric.city_pop<0).sum()),
        'birth_after_transaction':int((dob>timestamps).sum()),
    }
    bad_coords = ~numeric['lat'].between(-90,90) | ~numeric['merch_lat'].between(-90,90) | ~numeric['long'].between(-180,180) | ~numeric['merch_long'].between(-180,180)
    q['invalid_coordinates'] = int(bad_coords.sum())
    q1,q3 = df.amt.quantile([.25,.75])
    upper = float(q3+1.5*(q3-q1))
    q.update({'amount_iqr_upper_fence':upper,'amount_upper_outliers':int((df.amt>upper).sum()),
              'amount_min':float(df.amt.min()),'amount_max':float(df.amt.max())})
    schema=pd.DataFrame({'column':df.columns,'dtype':df.dtypes.astype(str).values,
                         'missing':df.isna().sum().values,'unique':df.nunique().values})
    return q,schema


def derive_features(raw):
    df=raw.copy(deep=True)
    timestamp=pd.to_datetime(df.trans_date_trans_time,errors='raise')
    dob=pd.to_datetime(df.dob,errors='raise')
    df['transaction_timestamp']=timestamp
    df['transaction_hour']=timestamp.dt.hour
    df['transaction_day_of_week']=timestamp.dt.dayofweek
    df['transaction_month']=timestamp.dt.month
    df['customer_age']=(timestamp-dob).dt.days/365.2425
    a,b,c,d=(np.radians(df[col].astype(float)) for col in ['lat','long','merch_lat','merch_long'])
    hav=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    df['customer_merchant_distance_km']=6371.0088*2*np.arcsin(np.sqrt(np.clip(hav,0,1)))
    return df


def class_distribution(df):
    result=df[TARGET].value_counts().reindex([0,1],fill_value=0).rename_axis(TARGET).to_frame('count')
    result['percentage']=100*result['count']/len(df)
    return result


def run_eda(df, output_dir):
    """Development data only; descriptive relationships are not causal evidence."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    derived=derive_features(df)
    amount=df.groupby(TARGET).amt.agg(['count','mean','median','min','max'])
    category=df.groupby('category')[TARGET].agg(['size','sum','mean']).rename(columns={'size':'transactions','sum':'fraud','mean':'fraud_rate'})
    category['fraud_rate_pct']=100*category.fraud_rate
    hours=derived.groupby('transaction_hour')[TARGET].agg(['size','sum','mean'])
    days=derived.groupby('transaction_day_of_week')[TARGET].agg(['size','sum','mean'])
    dates={'start':str(derived.transaction_timestamp.min()),'end':str(derived.transaction_timestamp.max())}
    for name,table in [('class_distribution',class_distribution(df)),('amount_summary',amount),('category_summary',category),('hour_summary',hours),('weekday_summary',days)]:
        table.to_csv(output_dir/f'{name}.csv')
    charts={}
    def save(fig,name):
        fig.tight_layout(); path=output_dir/f'{name}.png';fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig);charts[name]=path
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,3.3));counts=class_distribution(df)['count']
    ax.bar(['Legitimate','Fraud'],counts,color=['#237e78','#bd4148']);ax.set_yscale('log');ax.set_ylabel('Transactions (log scale)')
    for i,n in enumerate(counts):ax.text(i,n*1.15,f'{n:,}',ha='center')
    ax.set_ylim(1,len(df)*2);ax.set_title('Development class distribution');save(fig,'class_distribution')
    fig,ax=plt.subplots(figsize=(7,3.3))
    for value,label,color in [(0,'Legitimate','#237e78'),(1,'Fraud','#bd4148')]:
        ax.hist(np.log1p(df.loc[df[TARGET]==value,'amt']),bins=45,density=True,histtype='step',linewidth=1.8,label=label,color=color)
    ax.set_xlabel('log(1 + transaction amount in USD)');ax.set_ylabel('Density');ax.legend();save(fig,'amount_by_label')
    fig,ax=plt.subplots(figsize=(7,4));category.sort_values('fraud_rate_pct').fraud_rate_pct.plot.barh(ax=ax,color='#237e78')
    ax.set_xlabel('Fraud rate (%)');ax.set_ylabel('');save(fig,'category_fraud_rate')
    fig,ax=plt.subplots(figsize=(7,3.3));ax.plot(hours.index,100*hours['mean'],marker='o',color='#bd4148')
    ax.set_xlabel('Transaction hour');ax.set_ylabel('Fraud rate (%)');ax.set_xticks(range(0,24,2));save(fig,'hour_fraud_rate')
    fig,ax=plt.subplots(figsize=(7,2.5));ax.boxplot(np.log1p(df.amt),vert=False);ax.set_xlabel('log(1 + amount in USD)');ax.set_yticks([]);save(fig,'amount_outliers')
    return {'amount':amount,'category':category,'hours':hours,'days':days,'dates':dates,'charts':charts}


def prepare(train_raw, test_raw, output_dir):
    """Fit preprocessing on chronological training only. Never train a classifier."""
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    quality,schema=quality_checks(train_raw)
    blockers=['missing_cells','duplicate_rows','duplicate_transaction_ids','invalid_labels','invalid_timestamps',
              'invalid_birth_dates','invalid_numeric_cells','nonpositive_amounts','invalid_coordinates',
              'negative_city_population','birth_after_transaction']
    if any(quality[k] for k in blockers):
        raise ValueError('Quality gate failed; inspect and explicitly resolve findings before preprocessing')
    development=derive_features(train_raw).sort_values('transaction_timestamp',kind='stable').reset_index(drop=True)
    test=derive_features(test_raw).sort_values('transaction_timestamp',kind='stable').reset_index(drop=True)
    boundary=int(len(development)*.8)
    train,validation=development.iloc[:boundary],development.iloc[boundary:]
    if train.empty or validation.empty or train[TARGET].nunique()!=2 or validation[TARGET].nunique()!=2:
        raise ValueError('Training and validation need nonempty examples of both classes')
    if not train.transaction_timestamp.max()<=validation.transaction_timestamp.min()<test.transaction_timestamp.min():
        raise ValueError('Chronology checks failed')
    if not validation.transaction_timestamp.max()<test.transaction_timestamp.min():
        raise ValueError('Validation and test periods overlap')
    if set(development.trans_num)&set(test.trans_num):
        raise ValueError('Development and test transaction IDs overlap')
    for name,frame in [('training',train),('validation',validation),('test',test)]:
        if frame[FEATURES].isna().any().any() or not np.isfinite(frame[NUMERIC].to_numpy(dtype=float)).all():
            raise ValueError(f'{name}: incomplete or non-finite features')
    transformer=ColumnTransformer([
        ('numeric',StandardScaler(),NUMERIC),
        ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=True),CATEGORICAL)],
        sparse_threshold=1.0,verbose_feature_names_out=True)
    matrices={'training':transformer.fit_transform(train[FEATURES]),
              'validation':transformer.transform(validation[FEATURES]),'test':transformer.transform(test[FEATURES])}
    names=transformer.get_feature_names_out()
    rows=[]
    total=len(development)+len(test)
    for name,frame in [('Training',train),('Validation',validation),('Final test',test)]:
        rows.append({'split':name,'rows':len(frame),'percent_of_all':100*len(frame)/total,
                     'start':str(frame.transaction_timestamp.min()),'end':str(frame.transaction_timestamp.max()),
                     'fraud':int(frame[TARGET].sum()) if name!='Final test' else 'Reserved'})
    splits=pd.DataFrame(rows)
    example_cols=['trans_date_trans_time','category','amt','gender','state','city_pop','dob','lat','long','merch_lat','merch_long',TARGET]
    before=validation.iloc[[0]][example_cols]
    vector=matrices['validation'][0].toarray().ravel()
    active=np.flatnonzero(vector)
    after=pd.DataFrame({'feature':names[active],'value':vector[active]})
    original_card_sets={name:set(frame.cc_num) for name,frame in [('training',train),('validation',validation),('test',test)]}
    info={
        'quality':quality,'class_distribution':class_distribution(train_raw).reset_index().to_dict('records'),
        'cleaning':{'before':len(train_raw),'removed':0,'added':0,'after':len(train_raw)},
        'split_summary':rows,'numeric_features':NUMERIC,'categorical_features':CATEGORICAL,
        'feature_count':len(names),'matrix_shapes':{k:list(v.shape) for k,v in matrices.items()},
        'test_labels_loaded':False,'cross_file_duplicate_transaction_ids':0,
        'test_cards_unseen_in_training_split':len(original_card_sets['test']-original_card_sets['training']),
        'environment':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                       'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__,'joblib':joblib.__version__},
    }
    schema.to_csv(output_dir/'schema.csv',index=False)
    splits.to_csv(output_dir/'split_summary.csv',index=False)
    before.to_csv(output_dir/'before_preprocessing.csv',index=False)
    after.to_csv(output_dir/'after_preprocessing.csv',index=False)
    pd.Series(names,name='feature').to_csv(output_dir/'feature_names.csv',index=False)
    joblib.dump(transformer,output_dir/'preprocessor.joblib')
    (output_dir/'preparation_evidence.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info, splits, before, after, matrices

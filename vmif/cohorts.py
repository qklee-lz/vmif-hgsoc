import json
import urllib.request

import numpy as np
import pandas as pd

from .channels import GENES
from .config import EXCLUDE, data, work

EXTRA_GENES = ['PTPRC']
ALL_GENES = sorted({g for v in GENES.values() for g in v} | set(EXTRA_GENES))
CBIO = 'https://www.cbioportal.org/api'


def zmean(df, genes):
    cols = [g for g in genes if g in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    return z.mean(1, skipna=True).where(z.notna().any(axis=1))


def composites(df, prefix):
    out = pd.DataFrame({f'{prefix}_{m}': zmean(df, g) for m, g in GENES.items()}, index=df.index)
    for g in EXTRA_GENES:
        if g in df.columns:
            out[f'{prefix}_{g}'] = df[g]
    return out


def ptrc_proteome(kind='before'):
    f = {'before': 'FFPE_discovery_globalprotein_batchcorrected_before_imputation.tsv',
         'imputed': 'FFPE_discovery_globalprotein_imputed.tsv'}[kind]
    p = pd.read_csv(data('ptrc', 'proteome', f), sep='\t').dropna(subset=['HGNC symbol']).groupby('HGNC symbol').first()
    return p[[c for c in p.columns if c.startswith('FD_')]].T.astype(float)


def proteome_patient(sample):
    return '_'.join(sample.split('_')[2:4])


def proteome_block(sample):
    parts = sample.split('_')
    return '_'.join(parts[2:4]) + '|' + '_'.join(parts[4:])


def ptrc_patient_genes(kind='before', samples=None, genes=None):
    M = ptrc_proteome(kind)
    if samples is not None:
        M = M.loc[M.index.intersection(samples)]
    if genes is not None:
        M = M[[g for g in genes if g in M.columns]]
    return M.groupby([proteome_patient(c) for c in M.index]).mean()


def build_ptrc():
    cl = pd.read_excel(data('ptrc', 'PTRC-HGSOC_List_clincal_data.xlsx'))
    site = {'FHCRC': 'FH', 'Mayo': 'MC', 'UAB': 'UA'}
    sl = pd.DataFrame({
        'slide_id': cl['File Name'].str.replace('.svs', '', regex=False),
        'patient': cl['Sample Source'].map(site) + '_' + cl['Patient ID'].astype(str),
        'sample_id': cl['Sample ID'].astype(str),
        'site': cl['Sample Source'],
        'tissue': cl['Tumor type'],
        'neoadj': cl['Neo-adjuvant'].str.lower(),
        'label': (cl['Tumor response'].str.lower() == 'sensitive').astype(int),
    }).drop_duplicates('slide_id')
    sl = sl[[data('ptrc', 'slides', f'{s}.svs').exists() and s not in EXCLUDE for s in sl.slide_id]]
    sl.to_csv(work('cohorts', 'slides_PTRC-HGSOC.csv'), index=False)
    pat = sl.groupby('patient').agg(label=('label', 'first'), site=('site', 'first'),
                                    neoadj=('neoadj', lambda x: int((x == 'yes').any())))
    truth = pat.join(composites(ptrc_patient_genes('before', genes=ALL_GENES), 'prot'), how='left')
    truth.to_csv(work('cohorts', 'truth_PTRC-HGSOC.csv'))
    return sl, truth


def _sample01(s):
    s = s[s.index.astype(str).str.startswith('TCGA-')].copy()
    s.index = s.index.str[:15]
    s = s[s.index.str[13:15] == '01'].groupby(level=0).mean()
    s.index = s.index.str[:12]
    return s


def _get(url, payload=None):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=120))


def _profile(profile, sample_list, entrez, log=False):
    md = pd.DataFrame(_get(f'{CBIO}/molecular-profiles/{profile}/molecular-data/fetch',
                           {'sampleListId': sample_list, 'entrezGeneIds': list(entrez)}))
    md['gene'] = md.entrezGeneId.map(entrez)
    md = md[md.sampleId.str[13:15] == '01']
    md = md.pivot_table(index='patientId', columns='gene', values='value', aggfunc='mean').astype(float)
    return np.log2(md + 1) if log else md


def build_tcga():
    svs = sorted(p.stem for p in data('tcga', 'slides').glob('*.svs') if p.stem not in EXCLUDE)
    status = pd.read_excel(data('tcga', 'mmc2_PlatinumStatus.xlsx'), sheet_name='KeyClinicalData')
    status = status.set_index('bcr_patient_barcode').PlatinumStatus
    sl = pd.DataFrame({'slide_id': svs})
    sl['patient'], sl['sample'] = sl.slide_id.str[:12], sl.slide_id.str[:15]
    sl['stype'] = sl.slide_id.str[13:15]
    sl['prep'] = np.where(sl.slide_id.str.contains('-DX'), 'DX', 'TS')
    sl['site'] = sl.patient.str[5:7]
    sl['label'] = sl.patient.map(status.map({'Sensitive': 1, 'Resistant': 0}))
    sl = sl[sl.label.notna()].astype({'label': int})
    sl.to_csv(work('cohorts', 'slides_TCGA-OV.csv'), index=False)

    T = pd.DataFrame(index=pd.Index(sorted(sl.patient.unique()), name='patient'))
    T['label'] = sl.groupby('patient').label.first()
    T['site'] = sl.groupby('patient').site.first()
    g = lambda f: data('tcga', 'pancan', f)
    lf = pd.read_csv(g('TCGA_all_leuk_estimate.masked.20170107.tsv'), sep='\t', header=None)
    T['LF'] = _sample01(lf.set_index(1)[2])
    cb = pd.read_csv(g('TCGA.Kallisto.fullIDs.cibersort.relative.tsv'), sep='\t')
    cb.index = cb.SampleID.str.replace('.', '-', regex=False)
    T['cibersort_CD8_abs'] = _sample01(cb['T.cells.CD8']) * T.LF
    T['purity'] = _sample01(pd.read_csv(g('TCGA_mastercalls.abs_tables_JSedit.fixed.txt'), sep='\t').set_index('sample')['purity'])
    T['rppa_CD20'] = _sample01(pd.read_csv(g('TCGA-RPPA-pancan-clean.txt'), sep='\t').set_index('SampleID')['CD20'])
    hrd = pd.read_csv(g('TCGA.HRD_withSampleID.txt'), sep='\t').set_index('sampleID')
    hrd.index = hrd.index + 'A'
    T['HRD'] = _sample01(hrd['HRD'])
    st = pd.read_csv(g('TCGASubtype.20170308.tsv'), sep='\t')
    st = st[(st['cancer.type'] == 'OVCA') & (st['pan.samplesID'].str[13:15] == '01')]
    st = st.assign(p=st['pan.samplesID'].str[:12]).drop_duplicates('p').set_index('p')['Subtype_mRNA'].reindex(T.index)
    T['immunoreactive'] = (st == 'Immunoreactive').astype(float).where(st.notna())
    T['mesenchymal'] = (st == 'Mesenchymal').astype(float).where(st.notna())
    brca = pd.read_csv(data('tcga', 'sztupinszki2021_supplementary.txt'), sep='\t').set_index('PatientID')['BRCA-status']
    T['BRCA'] = brca.reindex(T.index).map(lambda s: np.nan if pd.isna(s) else float(s != 'intact'))

    study = 'ov_tcga_pan_can_atlas_2018'
    cd = pd.DataFrame(_get(f'{CBIO}/studies/{study}/clinical-data?clinicalDataType=PATIENT&projection=SUMMARY&pageSize=100000'))
    cd = cd.pivot_table(index='patientId', columns='clinicalAttributeId', values='value', aggfunc='first').reindex(T.index)
    for c in ['OS', 'PFS']:
        T[f'{c}_MONTHS'] = pd.to_numeric(cd[f'{c}_MONTHS'], errors='coerce')
        T[f'{c}_EVENT'] = cd[f'{c}_STATUS'].str.startswith('1').astype(float)
    entrez = {x['entrezGeneId']: x['hugoGeneSymbol'] for x in _get(f'{CBIO}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL', ALL_GENES)}
    T = T.join(composites(_profile(f'{study}_rna_seq_v2_mrna', f'{study}_all', entrez, log=True), 'rna'))
    T = T.join(composites(_profile('ov_tcga_mrna_U133', 'ov_tcga_mrna_U133', entrez), 'arr'))
    T = T.join(composites(_profile('ov_tcga_protein_quantification', 'ov_tcga_protein_quantification', entrez), 'cptac'))
    T.to_csv(work('cohorts', 'truth_TCGA-OV.csv'))
    return sl, T


def slides(cohort, tumour_only=True, ts_only=False):
    sl = pd.read_csv(work('cohorts', f'slides_{cohort}.csv'), dtype={'stype': str, 'sample_id': str}).set_index('slide_id')
    if cohort == 'TCGA-OV' and tumour_only:
        sl = sl[sl.stype == '01']
        if ts_only:
            sl = sl[sl.prep == 'TS']
    return sl


def truth(cohort):
    return pd.read_csv(work('cohorts', f'truth_{cohort}.csv'), index_col=0)

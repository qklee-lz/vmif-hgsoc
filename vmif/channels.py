GIGA = ['DAPI', 'TRITC', 'Cy5', 'PD-1', 'CD14', 'CD4', 'T-bet', 'CD34', 'CD68', 'CD16', 'CD11c', 'CD138', 'CD20', 'CD3',
        'CD8', 'PD-L1', 'CK', 'Ki67', 'Tryptase', 'aSMA', 'Caspase3-D', 'PHH3-B', 'Transgelin']
GIGA_PROTEIN = GIGA[3:]
GIGA_MIL = [c for c in GIGA if c not in ('TRITC', 'Cy5')]

HEX = ['DAPI', 'FOXP3', 'CDX2', 'CD8', 'p53', 'GATA3', 'CD45', 'HLA-DR', 'PD-L1', 'Ki67', 'CD45RA', 'CD4', 'CD21', 'MUC1',
       'CD30', 'CD2', 'Vimentin', 'CD20', 'Na-K-ATPase', 'CD5', 'IDO1', 'CK', 'CD56', 'aSMA', 'BCL2', 'CD25',
       'Collagen IV', 'CD11c', 'PD-1', 'Granzyme B', 'EGFR', 'VISTA', 'CD15', 'CD194', 'MMP9', 'Synaptophysin', 'CD7',
       'CD3', 'Chromogranin A', 'CD163', 'CD57', 'CD45RO', 'CD68', 'CD31', 'Podoplanin', 'CD34', 'CD38', 'CD138']

DECODERS = {'gigatime512': GIGA, 'gigatime256': GIGA, 'flash': GIGA, 'hex': HEX}

GENES = {'CK': ['KRT8', 'KRT18', 'KRT19'], 'CD8': ['CD8A'], 'CD3': ['CD3E', 'CD3D'], 'CD4': ['CD4'], 'CD14': ['CD14'],
         'CD16': ['FCGR3A'], 'CD11c': ['ITGAX'], 'CD138': ['SDC1'], 'CD34': ['CD34'], 'Ki67': ['MKI67'],
         'Tryptase': ['TPSB2'], 'aSMA': ['ACTA2', 'DES'], 'Transgelin': ['TAGLN'], 'Caspase3-D': ['CASP3'],
         'CD20': ['MS4A1'], 'PD-1': ['PDCD1'], 'PD-L1': ['CD274'], 'CD68': ['CD68'], 'T-bet': ['TBX21']}
MARKERS = list(GENES)
IMMUNE = ['PD-1', 'CD14', 'CD4', 'T-bet', 'CD68', 'CD16', 'CD11c', 'CD20', 'CD3', 'CD8']
STROMAL_CONTROLS = ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'FN1', 'PDGFRB']
SPATIAL_MARKERS = ['CD8', 'CD3', 'CD4', 'CD68', 'CD14', 'CD20', 'PD-1', 'PD-L1', 'Ki67', 'Transgelin', 'CD34']

ORION = ['Hoechst', 'AF1', 'CD31', 'CD45', 'CD68', 'Argo550', 'CD4', 'FOXP3', 'CD8a', 'CD45RO', 'CD20', 'PD-L1', 'CD3e',
         'CD163', 'E-cadherin', 'PD-1', 'Ki67', 'Pan-CK', 'SMA']
ORION_MAP = {
    'giga': {'PD-1': 'PD-1', 'CD4': 'CD4', 'CD68': 'CD68', 'CD20': 'CD20', 'CD3': 'CD3e', 'CD8': 'CD8a',
             'PD-L1': 'PD-L1', 'CK': 'Pan-CK', 'Ki67': 'Ki67', 'aSMA': 'SMA'},
    'hex': {'FOXP3': 'FOXP3', 'CD8': 'CD8a', 'CD45': 'CD45', 'PD-L1': 'PD-L1', 'Ki67': 'Ki67', 'CD4': 'CD4',
            'CD20': 'CD20', 'CK': 'Pan-CK', 'aSMA': 'SMA', 'PD-1': 'PD-1', 'CD3': 'CD3e', 'CD163': 'CD163',
            'CD45RO': 'CD45RO', 'CD68': 'CD68', 'CD31': 'CD31'},
}

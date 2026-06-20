"""
Constants for the kinase pipeline extractors.

Parts of the code are adpated from https://github.com/DunbrackLab/Kincore-standalone2/.
"""

PHOSPHORYLATED_AAS = ['TPO', 'SEP', 'PTR', 'PHD', 'NER']

OMITTED_LIGANDS = ['1PE','1PE','2HT','2PE','5LS','5LS','5TH','5TK','5TM','7PE','ACE','ACT','ACY','AF3','ALA','ALY','ARS','AU','BA','BCT','BEN','BME','BOG','BR',\
                 'BU1','BU3','BUD','BWB','CA','CAC','CAF','CAS','CAS','CD','CIT','CL','CME','CME','CO','CO3','CS','CSD','CSO','CSO','CSS','CSS','CSX','CXM',\
                 'CXS','CXS','DIO','DMS','DOD','DTD','DTT','DTT','DTV','DVT','DVT','EDO','EMC','EOH','EPE','EPE','FLC','FMT','GBL','GG5','GLC','GLC','GOL','HC4',\
                 'HG','HSJ','IMD','IOD','IPA','IPH','K','KCX','LGY','MES','MG8','MGF','MK8','MLA','MLI','MLY','MOH','MPD','MRD','MSE','MXE','MYR','NA',\
                 'NEP','NH4','NI','NO3','OCS','OCT','OCY','P4G','P6G','PDX','PEG','PG0','PG4','PGE','PGF','PGO','PHU','PO4','PPI','PSE','PTR','PUP','PZO','S26',\
                 'SBT','SCN','SCS','SEP','SEP','SGM','SIN','SO3','SO4','SR','SRT','SVQ','T8L','TAM','TAR','TCE','TFA','TLA','TLA','TMA','TPO','TRS','UNX','VO4',\
                 'YT3','Z4K','ZN','P4C','HOH','WAT','H2O','MN','MG','FE','CU','NI','CO','CR','MO','W','V','TI','SC','Y','LA','CE','PR','ND','PM','SM','EU','GD',\
                 'TB','DY','HO','ER','TM','YB','LU','AC','TH','PA','U','NP','PU','AM','CM','BK','CF','ES','FM','MD','NO','LR','RF','DB','SG','BH','HS','MT',\
                 'DS','RG','CN','NH','FL','MC','LV','TS','OG','PDO']

MODIFIED_AAS = ['ACE','ALY','AME','BWB','CAF','CAS','CME','COM','CSD','CSO','CSS','CSX','CXM','CY0','CYO','KCX','LGY','MHO','MK8','MLY','MSE','NEP','NMM',\
                 'OCS','OCY','PHD','PTR','SCS','SEP','T8L','TPO','UNK','ZXW']

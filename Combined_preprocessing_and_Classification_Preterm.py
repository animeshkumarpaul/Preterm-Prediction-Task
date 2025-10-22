'''
Preterm Birth preprocessing + classification
Author: Animesh Kumar Paul
email: animeshk@ualberta.ca
'''

#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pyreadstat
import pandas as pd
import os
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
import time
import pickle
from tqdm import tqdm
import datetime
def load_path_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f,encoding= "utf-8")

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)


# # Filtering

# ## Filter by Missing Data & Migration

# In[3]:


pickle_path = '/data/padmalab/preg/data/processed/req04417_bth_2002_2018_b.pkl'
birth_registry = load_path_obj(pickle_path)

pickle_path = '/data/padmalab/preg/data/processed/req04417_reg_0203_1819_m.pkl'
pop_reg = load_path_obj(pickle_path)

pickle_path = '/data/padmalab/preg/data/processed/req04417_cohort.pkl'
cohort = load_path_obj(pickle_path)


# In[4]:


birth_registry = birth_registry[(birth_registry.YEAR>2008 )& (birth_registry.YEAR<=2018 )]


# In[5]:


# FIltered by Single birth, gestation age, birthweight, pre-term
birth_registry_GESTATION_LBW = birth_registry[((birth_registry['KIND']== '10') | (birth_registry['KIND']== '10')) & (birth_registry['GESTATION']!=99) & (birth_registry['BIRTH_WT']!=9999)]


# In[6]:


birth_registry_cohort = pd.merge(birth_registry_GESTATION_LBW, cohort, on="Rcpt_Anon_ID_B")
birth_registry_cohort = birth_registry_cohort.drop(['Rcpt_Anon_ID_M_y'], axis=1)
birth_registry_cohort = birth_registry_cohort.rename(columns={"Rcpt_Anon_ID_M_x": "Rcpt_Anon_ID_M"})
birth_registry_cohort['BABY_CONCEPTION_DATE'] = birth_registry_cohort.PERS_DOB_B - pd.to_timedelta(birth_registry_cohort.GESTATION,unit='W')


# In[7]:



# In[8]:


birth_registry_cohort_popreg = birth_registry_cohort[birth_registry_cohort['Rcpt_Anon_ID_M'].isin(pop_reg['Rcpt_Anon_ID'])]


# In[11]:


pop_reg_filter_by_birthreg = pop_reg[pop_reg['Rcpt_Anon_ID'].isin(birth_registry_cohort_popreg['Rcpt_Anon_ID_M'])]


# In[12]:


pop_reg_filter_by_birthreg['PERS_PROV_IN_MIG_IND_FYE'] = pd.to_numeric(pop_reg_filter_by_birthreg['PERS_PROV_IN_MIG_IND_FYE']).copy(deep=True)
pop_reg_filter_by_birthreg['PERS_PROV_OUT_MIG_IND_FYE'] = pd.to_numeric(pop_reg_filter_by_birthreg['PERS_PROV_OUT_MIG_IND_FYE']).copy(deep=True)


# In[13]:


Unique_mother_id = np.unique(birth_registry_cohort_popreg['Rcpt_Anon_ID_M'])
birth_registry_group = birth_registry_cohort_popreg.groupby('Rcpt_Anon_ID_M', as_index=False)
pop_reg_group = pop_reg_filter_by_birthreg.groupby('Rcpt_Anon_ID', as_index=False)


# In[12]:


birth_registry_filtered_migration = pd.DataFrame(columns=birth_registry.columns.tolist())
with tqdm(total=Unique_mother_id.shape[0]) as pbar:
    for id in Unique_mother_id:
        pbar.update(1)

        birth = birth_registry_group.get_group(id).reset_index(drop=True)
        pop = pop_reg_group.get_group(id)
        for index in range(birth.shape[0]):

            select_years_data = pop[(birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0] <= pop['CAE_FYE_DATE']) & (birth.iloc[[index]]['PERS_DOB_B'].to_numpy()[0]+pd.offsets.DateOffset(years=1) >= pop['CAE_FYE_DATE'])]
            if sum(select_years_data['PERS_PROV_IN_MIG_IND_FYE'])==0:
                birth_registry_filtered_migration = pd.concat([birth_registry_filtered_migration, birth.iloc[[index]]], ignore_index=True)



# In[25]:


birth_registry_filtered_migration = birth_registry_filtered_migration[birth_registry_filtered_migration['BABY_CONCEPTION_DATE']> datetime.date(2009,1,1)].sort_values(by=['BABY_CONCEPTION_DATE'])


# In[26]:


birth_registry_filtered_migration



# In[16]: 
# use the migration based filtered Birth registry data

birth_registry = birth_registry_filtered_migration


# ## Filter PIN

# ### 26 Weeks

# In[36]:


pickle_path = '/data/padmalab/preg/data/processed/req04417_pin_2008_1819_m.pkl'
pin = load_path_obj(pickle_path)


# In[38]:


pin_filter_by_Birth_Mother_ID = pin[pin['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry_filter_by_pin = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(pin_filter_by_Birth_Mother_ID['Rcpt_Anon_ID'])]


# In[39]:


birth_registry_cohort_filter_by_pin = birth_registry_filter_by_pin#.iloc[108688*3:]
pin_filter_by_Birth_Mother_ID = pin_filter_by_Birth_Mother_ID[pin_filter_by_Birth_Mother_ID['Rcpt_Anon_ID'].isin(birth_registry_cohort_filter_by_pin['Rcpt_Anon_ID_M'])]


# In[40]:


length_pin = pin_filter_by_Birth_Mother_ID.shape[0]
length_birth_cohort = birth_registry_cohort_filter_by_pin.shape[0]
Unique_mother_id = np.unique(birth_registry_cohort_filter_by_pin['Rcpt_Anon_ID_M'])


# In[41]:


Unique_mother_id.shape


# In[ ]:


birth_registry_cohort_filter_by_pin_group = birth_registry_cohort_filter_by_pin.groupby('Rcpt_Anon_ID_M', as_index=False)
pin_filter_by_Birth_Mother_ID_group = pin_filter_by_Birth_Mother_ID.groupby('Rcpt_Anon_ID', as_index=False)
pin_filter_by_Birth_Mother_ID_FROM_CONCEPTION_DATE = pd.DataFrame(columns=pin_filter_by_Birth_Mother_ID.columns.tolist()[1:] + ['GESTATIONAL_WEEKS', 'Rcpt_Anon_ID_B'])
count  = 0
with tqdm(total=Unique_mother_id.shape[0]) as pbar:
    for id in Unique_mother_id:
        pbar.update(1)

        birth = birth_registry_cohort_filter_by_pin_group.get_group(id)
        pin = pin_filter_by_Birth_Mother_ID_group.get_group(id)


        for index in range(birth.shape[0]):
            selected_mother_medication_based_on_conception_date = pin.loc[((pin['DSPN_DATE']>=birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])&(pin['DSPN_DATE']<=birth.iloc[[index]]['PERS_DOB_B'].to_numpy()[0]))].copy(deep=True)
            if selected_mother_medication_based_on_conception_date.shape[0]==0:
                count = count + 1
            selected_mother_medication_based_on_conception_date['GESTATIONAL_WEEKS'] = ((selected_mother_medication_based_on_conception_date['DSPN_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)
            selected_mother_medication_based_on_conception_date['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            pin_filter_by_Birth_Mother_ID_FROM_CONCEPTION_DATE = pd.concat([pin_filter_by_Birth_Mother_ID_FROM_CONCEPTION_DATE, selected_mother_medication_based_on_conception_date], ignore_index=True)
            
# In[43]:


pin_10MONTHS = pin_filter_by_Birth_Mother_ID_FROM_CONCEPTION_DATE #load_path_obj('/data/padmalab/preg/data/processed/pin/pin_10MONTHS.pkl')


# ### 1 Year

# In[44]:


pin_filter_by_Birth_Mother_ID = pin[pin['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry_filter_by_pin = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(pin_filter_by_Birth_Mother_ID['Rcpt_Anon_ID'])]
birth_registry_cohort_filter_by_pin = birth_registry_filter_by_pin
pin_filter_by_Birth_Mother_ID = pin_filter_by_Birth_Mother_ID[pin_filter_by_Birth_Mother_ID['Rcpt_Anon_ID'].isin(birth_registry_cohort_filter_by_pin['Rcpt_Anon_ID_M'])]
birth_registry_cohort_filter_by_pin_group = birth_registry_cohort_filter_by_pin.groupby('Rcpt_Anon_ID_M', as_index=False)
pin_filter_by_Birth_Mother_ID_group = pin_filter_by_Birth_Mother_ID.groupby('Rcpt_Anon_ID', as_index=False)
Unique_mother_id = np.unique(birth_registry_cohort_filter_by_pin['Rcpt_Anon_ID_M'])


# In[ ]:


Unique_mother_id_1 = Unique_mother_id#[:73355]

pin_filter_by_Birth_Mother_ID_BEFORE_CONCEPTION_DATE = pd.DataFrame(columns=pin_filter_by_Birth_Mother_ID.columns.tolist()[1:] + ['GESTATIONAL_WEEKS', 'Rcpt_Anon_ID_B'])
count  = 0
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_cohort_filter_by_pin_group.get_group(id)
        pin_selected = pin_filter_by_Birth_Mother_ID_group.get_group(id)


        for index in range(birth.shape[0]):
            #selected_mother_medication_based_on_conception_date = pin_selected.loc[((pin_selected['DSPN_DATE']>=birth.iloc[[0]]['BABY_CONCEPTION_DATE'].to_numpy()[0])&(pin_selected['DSPN_DATE']<=birth.iloc[[0]]['PERS_DOB_B'].to_numpy()[0]))].copy(deep=True)
            selected_mother_medication_based_on_conception_date = pin_selected.loc[((pin_selected['DSPN_DATE'] >= (birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]-pd.offsets.DateOffset(years=1)).date()) & (pin_selected['DSPN_DATE'] <= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]))].copy(deep=True)
            if selected_mother_medication_based_on_conception_date.shape[0]==0:
                count = count + 1
            selected_mother_medication_based_on_conception_date['GESTATIONAL_WEEKS'] = ((selected_mother_medication_based_on_conception_date['DSPN_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)
            selected_mother_medication_based_on_conception_date['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            pin_filter_by_Birth_Mother_ID_BEFORE_CONCEPTION_DATE = pd.concat([pin_filter_by_Birth_Mother_ID_BEFORE_CONCEPTION_DATE, selected_mother_medication_based_on_conception_date], ignore_index=True)

# In[46]:


pin_1year = pin_filter_by_Birth_Mother_ID_BEFORE_CONCEPTION_DATE #load_path_obj('/data/padmalab/preg/data/processed/pin/pin_1YEAR_BEFORE_CONCEPTION_DATE.pkl')


# ## Filter AMB

# In[48]:


pickle_path = '/data/padmalab/preg/data/processed/req04417_amb_0203_1819_m.pkl'
amb = load_path_obj(pickle_path)


# ### 26 weeks

# In[49]:


amb_birth = amb[amb['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(amb_birth['Rcpt_Anon_ID'])]
birth_registry_group = birth_registry.groupby('Rcpt_Anon_ID_M', as_index=False)
amb_birth_group = amb_birth.groupby('Rcpt_Anon_ID', as_index=False)
Unique_mother_id = np.unique(birth_registry['Rcpt_Anon_ID_M'])


# In[94]:


Unique_mother_id_1 = Unique_mother_id#[:73679]

amb_birth_10months = pd.DataFrame(columns=amb_birth.columns.tolist()[1:] + ['Rcpt_Anon_ID_B'])
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_group.get_group(id)
        amb_group = amb_birth_group.get_group(id)

        for index in range(birth.shape[0]):
            #selected_amb = amb_group.loc[((amb_group['SEPI_START_DATE'] >= (birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]-pd.offsets.DateOffset(years=1)).date()) & (amb_group['SEPI_START_DATE'] <= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]))].copy(deep=True)
            selected_amb = amb_group.loc[((amb_group['SEPI_START_DATE'] >= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]) & (amb_group['SEPI_START_DATE'] <= birth.iloc[[index]]['PERS_DOB_B'].to_numpy()[0]))].copy(deep=True)

            if selected_amb.shape[0]==0:
                count = count + 1
                continue
            selected_amb['GESTATIONAL_WEEKS'] = ((selected_amb['SEPI_START_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)
            selected_amb['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            amb_birth_10months = pd.concat([amb_birth_10months, selected_amb], ignore_index=True)


# ### 1 Year

# In[52]:


amb_birth = amb[amb['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(amb_birth['Rcpt_Anon_ID'])]
birth_registry_group = birth_registry.groupby('Rcpt_Anon_ID_M', as_index=False)
amb_birth_group = amb_birth.groupby('Rcpt_Anon_ID', as_index=False)
Unique_mother_id = np.unique(birth_registry['Rcpt_Anon_ID_M'])


# In[ ]:


Unique_mother_id_1 = Unique_mother_id#[:73679]

amb_birth_1year = pd.DataFrame(columns=amb_birth.columns.tolist()[1:] + ['Rcpt_Anon_ID_B'])
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_group.get_group(id)
        amb_group = amb_birth_group.get_group(id)

        for index in range(birth.shape[0]):
            selected_amb = amb_group.loc[((amb_group['SEPI_START_DATE'] >= (birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]-pd.offsets.DateOffset(years=1)).date()) & (amb_group['SEPI_START_DATE'] <= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]))].copy(deep=True)
            if selected_amb.shape[0]==0:
                count = count + 1
                continue
            selected_amb['GESTATIONAL_WEEKS'] = ((selected_amb['SEPI_START_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)
            selected_amb['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            amb_birth_1year = pd.concat([amb_birth_1year, selected_amb], ignore_index=True)



# ## DAD

# In[54]:


pickle_path = '/data/padmalab/preg/data/processed/req04417_inp_0203_1819_m.pkl'
inp = load_path_obj(pickle_path)


# In[55]:


inp_birth = inp[inp['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(inp_birth['Rcpt_Anon_ID'])]
birth_registry_group = birth_registry.groupby('Rcpt_Anon_ID_M', as_index=False)
inp_birth_group = inp_birth.groupby('Rcpt_Anon_ID', as_index=False)
Unique_mother_id = np.unique(birth_registry['Rcpt_Anon_ID_M'])


# In[ ]:


Unique_mother_id_1 = Unique_mother_id#[:104215]
#History 1 year
inp_birth_1year = pd.DataFrame(columns=inp_birth.columns.tolist()[1:] + ['Rcpt_Anon_ID_B'])
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_group.get_group(id)
        inp_group = inp_birth_group.get_group(id)

        for index in range(birth.shape[0]):
            selected_inp = inp_group.loc[((inp_group['SEPI_START_DATE'] >= (birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]-pd.offsets.DateOffset(years=1)).date()) & (inp_group['SEPI_START_DATE'] <= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]))].copy(deep=True)
            if selected_inp.shape[0]==0:
                count = count + 1
                continue
            selected_inp['GESTATIONAL_WEEKS'] = ((selected_inp['SEPI_START_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)
            selected_inp['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            inp_birth_1year = pd.concat([inp_birth_1year, selected_inp], ignore_index=True)


# In[ ]:


#10month
inp_birth_10months = pd.DataFrame(columns=inp_birth.columns.tolist()[1:] + ['Rcpt_Anon_ID_B'])
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_group.get_group(id)
        inp_group = inp_birth_group.get_group(id)

        for index in range(birth.shape[0]):
            selected_inp = inp_group.loc[((inp_group['SEPI_START_DATE'] >= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]) & (inp_group['SEPI_START_DATE'] <= birth.iloc[[index]]['PERS_DOB_B'].to_numpy()[0]))].copy(deep=True)
            if selected_inp.shape[0]==0:
                count = count + 1
                continue
            selected_inp['GESTATIONAL_WEEKS'] = ((selected_inp['SEPI_START_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)
            selected_inp['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            inp_birth_10months = pd.concat([inp_birth_10months, selected_inp], ignore_index=True)


# ## CLM

# In[60]:


pickle_path = '/data/padmalab/preg/data/processed/req04417_clm_0203_1819_m.pkl'
clm = load_path_obj(pickle_path)


# ### 1 Year

# In[62]:


clm_birth = clm[clm['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(clm_birth['Rcpt_Anon_ID'])]
birth_registry_group = birth_registry.groupby('Rcpt_Anon_ID_M', as_index=False)
clm_birth_group = clm_birth.groupby('Rcpt_Anon_ID', as_index=False)
Unique_mother_id = np.unique(birth_registry['Rcpt_Anon_ID_M'])


# In[ ]:


Unique_mother_id_1 = Unique_mother_id#[:73951]

clm_birth_1year = pd.DataFrame(columns=clm_birth.columns.tolist()[1:] + ['Rcpt_Anon_ID_B'])
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_group.get_group(id)
        clm_group = clm_birth_group.get_group(id)

        for index in range(birth.shape[0]):
            selected_clm = clm_group.loc[((clm_group['SE_END_DATE'] >= (birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]-pd.offsets.DateOffset(years=1)).date()) & (clm_group['SE_END_DATE'] <= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]))].copy(deep=True)
            if selected_clm.shape[0]==0:
                count = count + 1
                continue
            selected_clm['GESTATIONAL_WEEKS'] = ((selected_clm['SE_END_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)                
            selected_clm['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            clm_birth_1year = pd.concat([clm_birth_1year, selected_clm], ignore_index=True)


# ### 26 weeks

# In[ ]:


clm_birth = clm[clm['Rcpt_Anon_ID'].isin(birth_registry['Rcpt_Anon_ID_M'])]
birth_registry = birth_registry[birth_registry['Rcpt_Anon_ID_M'].isin(clm_birth['Rcpt_Anon_ID'])]
birth_registry_group = birth_registry.groupby('Rcpt_Anon_ID_M', as_index=False)
clm_birth_group = clm_birth.groupby('Rcpt_Anon_ID', as_index=False)
Unique_mother_id = np.unique(birth_registry['Rcpt_Anon_ID_M'])


# In[ ]:


Unique_mother_id_1 = Unique_mother_id#[:73951]

clm_birth_10MONTHS = pd.DataFrame(columns=clm_birth.columns.tolist()[1:] + ['Rcpt_Anon_ID_B'])
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_mother_id_1.shape[0]) as pbar:
    for id in Unique_mother_id_1:
        pbar.update(1)

        birth = birth_registry_group.get_group(id)
        clm_group = clm_birth_group.get_group(id)

        for index in range(birth.shape[0]):
            #selected_clm = clm_group.loc[((clm_group['SE_END_DATE'] >= (birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]-pd.offsets.DateOffset(years=1)).date()) & (clm_group['SE_END_DATE'] <= birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]))].copy(deep=True)
            selected_clm = clm_group.loc[((clm_group['SE_END_DATE']>=birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0]) & (clm_group['SE_END_DATE']<=birth.iloc[[index]]['PERS_DOB_B'].to_numpy()[0]))].copy(deep=True)
            if selected_clm.shape[0]==0:
                count = count + 1
                continue
            selected_clm['GESTATIONAL_WEEKS'] = ((selected_clm['SE_END_DATE']  - birth.iloc[[index]]['BABY_CONCEPTION_DATE'].to_numpy()[0])/ np.timedelta64(1, 'W')).apply(np.ceil).copy(deep=True)                
            selected_clm['Rcpt_Anon_ID_B'] = birth.iloc[[index]]['Rcpt_Anon_ID_B'].to_numpy()[0]
            clm_birth_10MONTHS = pd.concat([clm_birth_10MONTHS, selected_clm], ignore_index=True)



# ## NOB

# In[67]:


pickle_path = '/data/padmalab/preg/data/processed/req04417_nob_2006_2018.pkl'
nob = load_path_obj(pickle_path)


# In[68]:


nob_birth = nob[nob['Rcpt_Anon_ID_B'].isin(birth_registry['Rcpt_Anon_ID_B'])]


# In[69]:


selected_features = ['HYPERTENSION', 'DIABETES', 'INSULIN', 'HEPB',
       'MHDEPRESSION', 'MHONANTI', 'MHPREVHIST', 'MHANXIETY', 'ALCOHOL',
       'SMOKER', 'SMOKEQUAN', 'SMOKEQUANN', 'SMOKEQ', 'SMOKEQUITPR',
       'SMOKEQUITRI', 'SECONDSHOME', 'SECONDSWORK', 'DRUGUSE', 'DRUGM',
       'DRUGME', 'DRUGHE', 'DRUGCO', 'DRUGSOL', 'DRUGCRY', 'DRUGECS',
       'OTHERDRUG', 'OTHERDRUGSPEC', 'DRUGQUITPR', 'GROUPBSTREP',
       'ANTIBIOTICS', 
        'HEMORR', 'BLOODT', 'RETPLAC', 'UTERUPT', 'HYSTER', 'MATINFECT', 'MATOTH', 
        'VITAMINK', 'METSCREEN',
       'ANTIBIOTRMT', 
       'SMOKENICOPROD', 'SMOKEECIG', 'SMOKEOTH', 'SMOKEOTHSP', 'ALCOFREQ',
       'DRUGOXY', 'DRUGPRES', 'DRUGSPRESSPEC', 
        'MFOLIC', 'Support',
       'VISIT', 'PRENATEDUC', 'Rcpt_Anon_ID_B', 'Rcpt_Anon_ID_M']


# In[72]:


nob_birth_selected_features= nob_birth[selected_features]
birth_registry_baby_unique_id = np.unique(birth_registry['Rcpt_Anon_ID_B']).tolist()
Unique_baby_id = np.unique(nob_birth['Rcpt_Anon_ID_B'])


# In[74]:


nob_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + selected_features[:-2], index=birth_registry_baby_unique_id)
nob_group = nob_birth_selected_features.groupby('Rcpt_Anon_ID_B', as_index=False)


# In[ ]:


count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id.shape[0]) as pbar:
    for id in Unique_baby_id:
        pbar.update(1)

        baby_info = nob_group.get_group(id)
        size = baby_info.shape[0]
        if size > 1:
            print('ERROR')
        index = 0
        baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
        nob_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
        nob_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID_M']]

        for f in selected_features[:-2]:
            nob_features.loc[baby_id, [f]]=[baby_info.iloc[index][f]]




# # Feature Extraction

# In[2]:


weeks = 26
icd_label = 3
prediction_time= "26Weeks"


# ## PIN

# ### 1 Year

# In[77]:



# In[78]:


ATC_CODES = np.unique(pin_1year['SUPP_DRUG_ATC_CODE']).tolist()
ATC_CODES.remove('')


# In[79]:


birth_registry_baby_unique_id = np.unique(birth_registry['Rcpt_Anon_ID_B']).tolist()
pin_group = pin_1year.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(pin_1year['Rcpt_Anon_ID_B'])


# In[ ]:


pin_group = pin_1year.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(pin_1year['Rcpt_Anon_ID_B'])

count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id.shape[0]) as pbar:
    for id in Unique_baby_id:
        pbar.update(1)
        baby_info = pin_group.get_group(id)
        size = baby_info.shape[0]
        for index in range(size):

            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            pin_1year_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
            pin_1year_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID']]

            drug_code = baby_info.iloc[index]['SUPP_DRUG_ATC_CODE']
            if drug_code in ATC_CODES:
                pin_1year_features.loc[baby_id, [drug_code]]= pin_1year_features.loc[baby_id, [drug_code]] + baby_info.iloc[index]['DSPN_DAY_SUPPLY_QTY']


# In[16]:




# ### 26 weeks

# In[27]:


# In[28]:


pin_10MONTHS = pin_10MONTHS[pin_10MONTHS.GESTATIONAL_WEEKS < weeks]
birth_registry_baby_unique_id = np.unique(birth_registry['Rcpt_Anon_ID_B']).tolist()


# In[ ]:

#Direct ATC 3 Label summary
atc_level = 3
pin_10MONTHS['SUPP_DRUG_ATC_CODE'] = pin_10MONTHS['SUPP_DRUG_ATC_CODE'].str[:atc_level]
ATC_CODES = np.unique(pin_10MONTHS['SUPP_DRUG_ATC_CODE']).tolist()
ATC_CODES.remove('')
ATC_CODES_start = [s + '_start' for s in ATC_CODES]



pin_group = pin_10MONTHS.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(pin_10MONTHS['Rcpt_Anon_ID_B'])
pin_10MONTHS_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + ATC_CODES + ATC_CODES_start, index=birth_registry_baby_unique_id)

pin_10MONTHS_features = pin_10MONTHS_features.fillna(0)
pin_10MONTHS_features[ATC_CODES_start] = pin_10MONTHS_features[ATC_CODES_start].replace(0, np.inf)


count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id.shape[0]) as pbar:
    for id in Unique_baby_id:
        pbar.update(1)
        baby_info = pin_group.get_group(id)
        size = baby_info.shape[0]
        for index in range(size):

            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            pin_10MONTHS_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
            pin_10MONTHS_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID']]

            drug_code = baby_info.iloc[index]['SUPP_DRUG_ATC_CODE']
            if drug_code in ATC_CODES:
                pin_10MONTHS_features.loc[baby_id, drug_code]= pin_10MONTHS_features.loc[baby_id, drug_code] + baby_info.iloc[index]['DSPN_DAY_SUPPLY_QTY']
                if pin_10MONTHS_features.loc[baby_id, drug_code + '_start'] > baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                    pin_10MONTHS_features.loc[baby_id, [drug_code + '_start']]= baby_info.iloc[index]['GESTATIONAL_WEEKS']

pin_10MONTHS_features[ATC_CODES_start] = pin_10MONTHS_features[ATC_CODES_start].replace(np.inf, np.nan)


# In[17]:




# ### Summary by 3 levels

# In[30]:


pin_features = pin_10MONTHS_features
atc_level = 3
new_columns={}
new_columns['Rcpt_Anon_ID'] = pin_features['Rcpt_Anon_ID']
new_columns['Rcpt_Anon_ID_B'] = pin_features['Rcpt_Anon_ID_B']
for col in pin_features.columns[2:]:
    prefix = col[:atc_level]
    if prefix not in new_columns:
        new_columns[prefix] = pin_features[col]
    else:
        new_columns[prefix] += pin_features[col]
pin_10MONTHS_features_atclevel3 = pd.DataFrame(new_columns)


# In[3]:


pin_10MONTHS_features  = pin_10MONTHS_features_atclevel3 # load_path_obj(pickle_path) 


# In[90]:


pin_features = pin_1year_features
atc_level = 3
new_columns={}
new_columns['Rcpt_Anon_ID'] = pin_features['Rcpt_Anon_ID']
new_columns['Rcpt_Anon_ID_B'] = pin_features['Rcpt_Anon_ID_B']
for col in pin_features.columns[2:]:
    prefix = col[:atc_level]
    if prefix not in new_columns:
        new_columns[prefix] = pin_features[col]
    else:
        new_columns[prefix] += pin_features[col]
pin_1year_features_atclevel3 = pd.DataFrame(new_columns)


# In[4]:


#pickle_path = '/data/padmalab/preg/data/processed/pin_1year_features_atclevel_3.pkl'
pin_1year_features = pin_1year_features_atclevel3 #load_path_obj(pickle_path)


# ## DAD

# In[34]:


# Use SEPI_END_DATE, SEPI_START_DATE
HLTH_DX_CODE = [
'HLTH_DX_CODE_1',
 'HLTH_DX_CODE_OTH_2',
 'HLTH_DX_CODE_OTH_3',
 'HLTH_DX_CODE_OTH_4',
 'HLTH_DX_CODE_OTH_5',
 'HLTH_DX_CODE_OTH_6',
 'HLTH_DX_CODE_OTH_7',
 'HLTH_DX_CODE_OTH_8',
 'HLTH_DX_CODE_OTH_9',
 'HLTH_DX_CODE_OTH_10',
 'HLTH_DX_CODE_OTH_11',
 'HLTH_DX_CODE_OTH_12',
 'HLTH_DX_CODE_OTH_13',
 'HLTH_DX_CODE_OTH_14',
 'HLTH_DX_CODE_OTH_15',
 'HLTH_DX_CODE_OTH_16',
 'HLTH_DX_CODE_OTH_17',
 'HLTH_DX_CODE_OTH_18',
 'HLTH_DX_CODE_OTH_19',
 'HLTH_DX_CODE_OTH_20',
 'HLTH_DX_CODE_OTH_21',
 'HLTH_DX_CODE_OTH_22',
 'HLTH_DX_CODE_OTH_23',
 'HLTH_DX_CODE_OTH_24',
 'HLTH_DX_CODE_OTH_25']

#COUNT
SE_INTRV_CODE=[
 'SE_INTRV_CODE_1',
 'SE_INTRV_CODE_2',
 'SE_INTRV_CODE_3',
 'SE_INTRV_CODE_4',
 'SE_INTRV_CODE_5',
 'SE_INTRV_CODE_6',
 'SE_INTRV_CODE_7',
 'SE_INTRV_CODE_8',
 'SE_INTRV_CODE_9',
 'SE_INTRV_CODE_10',
 'SE_INTRV_CODE_11',
 'SE_INTRV_CODE_12',
 'SE_INTRV_CODE_13',
 'SE_INTRV_CODE_14',
 'SE_INTRV_CODE_15',
 'SE_INTRV_CODE_16',
 'SE_INTRV_CODE_17',
 'SE_INTRV_CODE_18',
 'SE_INTRV_CODE_19',
 'SE_INTRV_CODE_20']


# ### 1 Year

# In[35]:


pickle_path ='/data/padmalab/preg/data/processed/inp_birth_1year.pkl'
inp_birth_1year = load_path_obj(pickle_path)


# In[36]:


for col in HLTH_DX_CODE:
    inp_birth_1year[col] = inp_birth_1year[col].str[:icd_label]

for col in SE_INTRV_CODE:
    inp_birth_1year[col] = inp_birth_1year[col].str[:icd_label]


# In[38]:


all_codes_in_HLTH_DX_CODE = []
for code in HLTH_DX_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_HLTH_DX_CODE = all_codes_in_HLTH_DX_CODE + inp_birth_1year[code].unique().tolist()
all_codes_in_HLTH_DX_CODE = np.unique(all_codes_in_HLTH_DX_CODE).tolist()
all_codes_in_HLTH_DX_CODE.remove("")


all_codes_in_SE_INTRV_CODE = []
for code in SE_INTRV_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_SE_INTRV_CODE = all_codes_in_SE_INTRV_CODE + inp_birth_1year[code].unique().tolist()
all_codes_in_SE_INTRV_CODE = np.unique(all_codes_in_SE_INTRV_CODE).tolist()
all_codes_in_SE_INTRV_CODE.remove("")
all_codes_in_all_columns = all_codes_in_HLTH_DX_CODE + all_codes_in_SE_INTRV_CODE


# In[39]:


birth_registry_baby_unique_id = np.unique(birth_registry['Rcpt_Anon_ID_B']).tolist()
inp_group = inp_birth_1year.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(inp_birth_1year['Rcpt_Anon_ID_B'])
inp_birth_1year_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + all_codes_in_all_columns + icd_codes_end, index=birth_registry_baby_unique_id)


# In[ ]:


with tqdm(total=len(all_codes_in_all_columns)) as pbar:       
    for col in all_codes_in_all_columns:
        inp_birth_1year_features[col].values[:] = 0
        pbar.update(1)
print('Done 1')
with tqdm(total=len(icd_codes_end)) as pbar:       
    for col in icd_codes_end:
        inp_birth_1year_features[col].values[:] = -1
        pbar.update(1)

Unique_baby_id_1 = Unique_baby_id#[:213864]


count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id_1.shape[0]) as pbar:
    for id in Unique_baby_id_1:
        pbar.update(1)
        baby_info = inp_group.get_group(id)
        size = baby_info.shape[0]
        for index in range(size):
            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            inp_birth_1year_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
            inp_birth_1year_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID']]
            if np.isnan(baby_info.iloc[index]['SEPI_RIW_VALUE'] ):
                print(baby_info.iloc[index]['SEPI_RIW_VALUE'])
                continue

            #####################################
            for c_name in SE_INTRV_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_SE_INTRV_CODE:
                    inp_birth_1year_features.loc[baby_id, assigned_code]= inp_birth_1year_features.loc[baby_id, assigned_code] + 1

                    if inp_birth_1year_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                        inp_birth_1year_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']

                else:
                    break

            #####################################
            for c_name in HLTH_DX_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_HLTH_DX_CODE:
                    inp_birth_1year_features.loc[baby_id, assigned_code]= inp_birth_1year_features.loc[baby_id, assigned_code] + (baby_info.iloc[index]['SEPI_END_DATE']-baby_info.iloc[index]['SEPI_START_DATE']).days

                    if inp_birth_1year_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                        inp_birth_1year_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']
                else:
                    break


# In[5]:


inp_1year_features = inp_birth_1year_features #load_path_obj('/data/padmalab/preg/data/processed/inp_birth_1year_features_corrected2_level_3.pkl')


# ### 26 Weeks

# In[41]:


pickle_path = '/data/padmalab/preg/data/processed/inp_birth_10months_combined_(with)_GESTATIONAL_WEEKS.pkl'
inp_10MONTHS= load_path_obj(pickle_path)


# In[42]:


inp_10MONTHS = inp_10MONTHS[(inp_10MONTHS['GESTATIONAL_WEEKS'] < weeks)]


# In[48]:


for col in HLTH_DX_CODE:
    inp_10MONTHS[col] = inp_10MONTHS[col].str[:icd_label]

for col in SE_INTRV_CODE:
    inp_10MONTHS[col] = inp_10MONTHS[col].str[:icd_label]


# In[44]:



# In[45]:


all_codes_in_HLTH_DX_CODE = []
for code in HLTH_DX_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_HLTH_DX_CODE = all_codes_in_HLTH_DX_CODE + inp_10MONTHS[code].unique().tolist()
all_codes_in_HLTH_DX_CODE = np.unique(all_codes_in_HLTH_DX_CODE).tolist()
all_codes_in_HLTH_DX_CODE.remove("")


all_codes_in_SE_INTRV_CODE = []
for code in SE_INTRV_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_SE_INTRV_CODE = all_codes_in_SE_INTRV_CODE + inp_10MONTHS[code].unique().tolist()
all_codes_in_SE_INTRV_CODE = np.unique(all_codes_in_SE_INTRV_CODE).tolist()
all_codes_in_SE_INTRV_CODE.remove("")
all_codes_in_all_columns = all_codes_in_HLTH_DX_CODE + all_codes_in_SE_INTRV_CODE

icd_codes_end = [s + '_end' for s in all_codes_in_all_columns]


# In[46]:


birth_registry_baby_unique_id = np.unique(birth_registry['Rcpt_Anon_ID_B']).tolist()
inp_group = inp_10MONTHS.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(inp_10MONTHS['Rcpt_Anon_ID_B'])
inp_10MONTHS_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + all_codes_in_all_columns + icd_codes_end, index=birth_registry_baby_unique_id)


# In[ ]:


with tqdm(total=len(all_codes_in_all_columns)) as pbar:       
    for col in all_codes_in_all_columns:
        inp_10MONTHS_features[col].values[:] = 0
        pbar.update(1)
print('Done 1')
with tqdm(total=len(icd_codes_end)) as pbar:       
    for col in icd_codes_end:
        inp_10MONTHS_features[col].values[:] = -1
        pbar.update(1)

Unique_baby_id_1 = Unique_baby_id#[:213864]


count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id_1.shape[0]) as pbar:
    for id in Unique_baby_id_1:
        pbar.update(1)
        baby_info = inp_group.get_group(id)
        size = baby_info.shape[0]
        for index in range(size):
            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            inp_10MONTHS_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
            inp_10MONTHS_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID']]
            if np.isnan(baby_info.iloc[index]['SEPI_RIW_VALUE'] ):
                print(baby_info.iloc[index]['SEPI_RIW_VALUE'])
                continue

            #####################################
            for c_name in SE_INTRV_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_SE_INTRV_CODE:
                    inp_10MONTHS_features.loc[baby_id, assigned_code]= inp_10MONTHS_features.loc[baby_id, assigned_code] + 1

                    if inp_10MONTHS_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                        inp_10MONTHS_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']

                else:
                    break

            #####################################
            for c_name in HLTH_DX_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_HLTH_DX_CODE:
                    inp_10MONTHS_features.loc[baby_id, assigned_code]= inp_10MONTHS_features.loc[baby_id, assigned_code] + (baby_info.iloc[index]['SEPI_END_DATE']-baby_info.iloc[index]['SEPI_START_DATE']).days

                    if inp_10MONTHS_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                        inp_10MONTHS_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']
                else:
                    break



# In[6]:


# ## AMB

# In[49]:


#Use SEPI_START_DATE SEPI_END_DATE

HLTH_DX_CODE= ['HLTH_DX_CODE_MR',
       'HLTH_DX_CODE_OTH_2', 'HLTH_DX_CODE_OTH_3', 'HLTH_DX_CODE_OTH_4',
       'HLTH_DX_CODE_OTH_5', 'HLTH_DX_CODE_OTH_6', 'HLTH_DX_CODE_OTH_7',
       'HLTH_DX_CODE_OTH_8', 'HLTH_DX_CODE_OTH_9', 'HLTH_DX_CODE_OTH_10']

# COUNT 
SE_INTRV_CODE=[
       'SE_INTRV_CODE_PR', 'SE_INTRV_CODE_2', 'SE_INTRV_CODE_3',
       'SE_INTRV_CODE_4', 'SE_INTRV_CODE_5', 'SE_INTRV_CODE_6',
       'SE_INTRV_CODE_7', 'SE_INTRV_CODE_8', 'SE_INTRV_CODE_9',
       'SE_INTRV_CODE_10']


# ### 1 yr

# In[50]:




# In[51]:


for col in HLTH_DX_CODE:
    amb_birth_1year[col] = amb_birth_1year[col].str[:icd_label]

for col in SE_INTRV_CODE:
    amb_birth_1year[col] = amb_birth_1year[col].str[:icd_label]


# In[53]:


all_codes_in_HLTH_DX_CODE = []
for code in HLTH_DX_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_HLTH_DX_CODE = all_codes_in_HLTH_DX_CODE + amb_birth_1year[code].unique().tolist()
all_codes_in_HLTH_DX_CODE = np.unique(all_codes_in_HLTH_DX_CODE).tolist()
all_codes_in_HLTH_DX_CODE.remove("")


all_codes_in_SE_INTRV_CODE = []
for code in SE_INTRV_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_SE_INTRV_CODE = all_codes_in_SE_INTRV_CODE + amb_birth_1year[code].unique().tolist()
all_codes_in_SE_INTRV_CODE = np.unique(all_codes_in_SE_INTRV_CODE).tolist()
all_codes_in_SE_INTRV_CODE.remove("")
all_codes_in_all_columns = all_codes_in_HLTH_DX_CODE + all_codes_in_SE_INTRV_CODE 



# In[54]:


amb_group = amb_birth_1year.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(amb_birth_1year['Rcpt_Anon_ID_B'])
amb_1year_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + all_codes_in_all_columns + icd_codes_end, index=birth_registry_baby_unique_id)


# In[ ]:


Unique_baby_id_1 = Unique_baby_id#[43000*2:43000*3]
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id_1.shape[0]) as pbar:
    for id in Unique_baby_id_1:
        pbar.update(1)
        baby_info = amb_group.get_group(id)
        size = baby_info.shape[0]
        for index in range(size):
            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            amb_1year_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
            amb_1year_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID']]

            if not pd.isna(baby_info.iloc[index]['SEPI_CACS_RIW_VALUE']):
                amb_1year_features.loc[baby_id, ['SEPI_CACS_RIW_VALUE']]= amb_1year_features.loc[baby_id, ['SEPI_CACS_RIW_VALUE']] + baby_info.iloc[index]['SEPI_CACS_RIW_VALUE']
            #####################################
            for c_name in other_codes:
                assigned_code = baby_info.iloc[index][c_name] + '_O'
                if assigned_code in all_codes_in_other_codes:
                    amb_1year_features.loc[baby_id, [assigned_code]]= amb_1year_features.loc[baby_id, [assigned_code]] + 1 #[(baby_info.iloc[index]['SEPI_END_DATE']-baby_info.iloc[index]['SEPI_START_DATE']).days]
            #####################################
            for c_name in HLTH_DX_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_HLTH_DX_CODE:
                    amb_1year_features.loc[baby_id, [assigned_code]]= amb_1year_features.loc[baby_id, [assigned_code]] + 1 #[(baby_info.iloc[index]['SEPI_END_DATE']-baby_info.iloc[index]['SEPI_START_DATE']).days]
                else:
                    break
            #####################################
            for c_name in SE_INTRV_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_SE_INTRV_CODE:
                    amb_1year_features.loc[baby_id, [assigned_code]]= amb_1year_features.loc[baby_id, [assigned_code]] + 1
                else:
                    break


# In[7]:



# In[57]:


amb_birth_10months = amb_birth_10months[(amb_birth_10months['GESTATIONAL_WEEKS'] < weeks)]


# In[58]:


for col in HLTH_DX_CODE:
    amb_birth_10months[col] = amb_birth_10months[col].str[:icd_label]

for col in SE_INTRV_CODE:
    amb_birth_10months[col] = amb_birth_10months[col].str[:icd_label]


# In[59]:


all_codes_in_HLTH_DX_CODE = []
for code in HLTH_DX_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_HLTH_DX_CODE = all_codes_in_HLTH_DX_CODE + amb_birth_10months[code].unique().tolist()
all_codes_in_HLTH_DX_CODE = np.unique(all_codes_in_HLTH_DX_CODE).tolist()
all_codes_in_HLTH_DX_CODE.remove("")


all_codes_in_SE_INTRV_CODE = []
for code in SE_INTRV_CODE:
    #print(inp_birth_1year[code].unique())
    all_codes_in_SE_INTRV_CODE = all_codes_in_SE_INTRV_CODE + amb_birth_10months[code].unique().tolist()
all_codes_in_SE_INTRV_CODE = np.unique(all_codes_in_SE_INTRV_CODE).tolist()
all_codes_in_SE_INTRV_CODE.remove("")
all_codes_in_all_columns = all_codes_in_HLTH_DX_CODE + all_codes_in_SE_INTRV_CODE 
icd_codes_end = [s + '_end' for s in all_codes_in_all_columns]


# In[60]:


amb_group = amb_birth_10months.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(amb_birth_10months['Rcpt_Anon_ID_B'])
amb_birth_10months_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + all_codes_in_all_columns + icd_codes_end, index=birth_registry_baby_unique_id)


# In[ ]:


with tqdm(total=len(all_codes_in_all_columns)) as pbar:       
    for col in all_codes_in_all_columns:
        amb_birth_10months_features[col].values[:] = 0
        pbar.update(1)
print('Done 1')
with tqdm(total=len(icd_codes_end)) as pbar:       
    for col in icd_codes_end:
        amb_birth_10months_features[col].values[:] = -1
        pbar.update(1)

Unique_baby_id_1 = Unique_baby_id#[:43000]
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id_1.shape[0]) as pbar:
    for id in Unique_baby_id_1:
        pbar.update(1)
        baby_info = amb_group.get_group(id)
        size = baby_info.shape[0]
        for index in range(size):
            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            amb_birth_10months_features.loc[baby_id, 'Rcpt_Anon_ID_B']=baby_id
            amb_birth_10months_features.loc[baby_id, 'Rcpt_Anon_ID']=baby_info.iloc[index]['Rcpt_Anon_ID']

            for c_name in HLTH_DX_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_HLTH_DX_CODE:
                    amb_birth_10months_features.loc[baby_id, assigned_code]= amb_birth_10months_features.loc[baby_id, assigned_code] + 1 #[(baby_info.iloc[index]['SEPI_END_DATE']-baby_info.iloc[index]['SEPI_START_DATE']).days]
                    if amb_birth_10months_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                        amb_birth_10months_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']

                else:
                    break
            #####################################
            for c_name in SE_INTRV_CODE:
                assigned_code = baby_info.iloc[index][c_name]
                if assigned_code in all_codes_in_SE_INTRV_CODE:
                    amb_birth_10months_features.loc[baby_id, assigned_code]= amb_birth_10months_features.loc[baby_id, assigned_code] + 1
                    if amb_birth_10months_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                        amb_birth_10months_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']

                else:
                    break






# ## CLM

# In[64]:


# amount money 
HLTH_DX_ICD9X_CODES=[
       'HLTH_DX_ICD9X_CODE_1','HLTH_DX_ICD9X_CODE_2','HLTH_DX_ICD9X_CODE_3']

#count
Others_CODES=[
        'HLTH_SRVC_CCPX_CODE'
]


# In[ ]:





# ### 1 Yr

# In[65]:



# In[78]:


clm_birth_1year


# In[66]:


all_codes_in_HLTH_DX_ICD9X_CODE = []
for code in HLTH_DX_ICD9X_CODES:
    #print(inp_birth_1year[code].unique())
    all_codes_in_HLTH_DX_ICD9X_CODE = all_codes_in_HLTH_DX_ICD9X_CODE + clm_birth_1year[code].unique().tolist()
all_codes_in_HLTH_DX_ICD9X_CODE = np.unique(all_codes_in_HLTH_DX_ICD9X_CODE).tolist()
all_codes_in_HLTH_DX_ICD9X_CODE.remove("")


all_codes_in_Others_CODES = []
for code in Others_CODES:
    #print(inp_birth_1year[code].unique())
    all_codes_in_Others_CODES = all_codes_in_Others_CODES + clm_birth_1year[code].unique().tolist()
all_codes_in_Others_CODES = np.unique(all_codes_in_Others_CODES).tolist()
#all_codes_in_Others_CODES.remove("")


all_codes_in_all_columns = all_codes_in_HLTH_DX_ICD9X_CODE + all_codes_in_Others_CODES


# In[67]:


clm_group = clm_birth_1year.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(clm_birth_1year['Rcpt_Anon_ID_B'])
clm_birth_1year_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + all_codes_in_all_columns, index=birth_registry_baby_unique_id)


# In[ ]:


for col in clm_birth_1year_features.columns:
    clm_birth_1year_features[col].values[:] = 0

count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id.shape[0]) as pbar:
    for id in Unique_baby_id:
        pbar.update(1)

        baby_info = clm_group.get_group(id)
        size = baby_info.shape[0]

        for index in range(size):
            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            clm_birth_1year_features.loc[baby_id, ['Rcpt_Anon_ID_B']]=[baby_id]
            clm_birth_1year_features.loc[baby_id, ['Rcpt_Anon_ID']]=[baby_info.iloc[index]['Rcpt_Anon_ID']]

            #####################################

            assigned_code = baby_info.iloc[index]['HLTH_SRVC_CCPX_CODE']
            clm_birth_1year_features.loc[baby_id, [assigned_code]]= clm_birth_1year_features.loc[baby_id, [assigned_code]] + baby_info.iloc[index]['CAE_SYS_AMT']

            #####################################

            assigned_code = baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_1']
            if assigned_code =="":
                continue
            clm_birth_1year_features.loc[baby_id, [assigned_code]]= clm_birth_1year_features.loc[baby_id, [assigned_code]] + 1

            assigned_code = baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_2']
            if assigned_code =="":
                continue
            clm_birth_1year_features.loc[baby_id, [assigned_code]]= clm_birth_1year_features.loc[baby_id, [assigned_code]] + 1

            assigned_code = baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_3']
            if assigned_code =="":
                continue
            clm_birth_1year_features.loc[baby_id, [assigned_code]]= clm_birth_1year_features.loc[baby_id, [assigned_code]] + 1



# In[ ]:


atc_level = 3
new_columns={}
new_columns['Rcpt_Anon_ID'] = clm_birth_1year_features['Rcpt_Anon_ID']
new_columns['Rcpt_Anon_ID_B'] = clm_birth_1year_features['Rcpt_Anon_ID_B']
for col in clm_birth_1year_features.columns[2:]:
    prefix = col[:atc_level]
    if prefix not in new_columns:
        new_columns[prefix] = clm_birth_1year_features[col]
    else:
        new_columns[prefix] += clm_birth_1year_features[col]
clm_birth_1year_features_atclevel = pd.DataFrame(new_columns)

atc_level = 3
new_columns={}
new_columns['Rcpt_Anon_ID'] = clm_birth_1year_features['Rcpt_Anon_ID']
new_columns['Rcpt_Anon_ID_B'] = clm_birth_1year_features['Rcpt_Anon_ID_B']
for col in clm_birth_1year_features.columns[2:]:
    if 'end' in col:
        prefix = col[:atc_level] + '_end'
        clm_birth_1year_features[col][clm_birth_1year_features[col] == -1] = 0
    else:
        prefix = col[:atc_level]

    if prefix not in new_columns:
        new_columns[prefix] = clm_birth_1year_features[col]
    else:
        new_columns[prefix] += clm_birth_1year_features[col]
clm_birth_1year_features_atclevel = pd.DataFrame(new_columns)


# In[ ]:


clm_birth_1year_features = clm_birth_1year_features_atclevel


# In[9]:

# ### 26 Weeks

# In[69]:



# In[ ]:





# In[70]:


clm_birth_10MONTHS = clm_birth_10MONTHS[(clm_birth_10MONTHS['GESTATIONAL_WEEKS'] < weeks)]


# In[71]:


birth_registry_baby_unique_id = np.unique(birth_registry['Rcpt_Anon_ID_B']).tolist()


# In[72]:


all_codes_in_HLTH_DX_ICD9X_CODE = []
for code in HLTH_DX_ICD9X_CODES:
    #print(inp_birth_1year[code].unique())
    all_codes_in_HLTH_DX_ICD9X_CODE = all_codes_in_HLTH_DX_ICD9X_CODE + clm_birth_10MONTHS[code].unique().tolist()
all_codes_in_HLTH_DX_ICD9X_CODE = np.unique(all_codes_in_HLTH_DX_ICD9X_CODE).tolist()
all_codes_in_HLTH_DX_ICD9X_CODE.remove("")


all_codes_in_Others_CODES = []
for code in Others_CODES:
    #print(inp_birth_1year[code].unique())
    all_codes_in_Others_CODES = all_codes_in_Others_CODES + clm_birth_10MONTHS[code].unique().tolist()
all_codes_in_Others_CODES = np.unique(all_codes_in_Others_CODES).tolist()
#all_codes_in_Others_CODES.remove("")


all_codes_in_all_columns = all_codes_in_HLTH_DX_ICD9X_CODE + all_codes_in_Others_CODES
#HLTH_DX_ICD9X_CODE_list = list(np.unique([x[:3] for x in all_codes_in_HLTH_DX_ICD9X_CODE]))
icd_codes_end = [s + '_end' for s in all_codes_in_HLTH_DX_ICD9X_CODE]


# In[73]:


clm_group = clm_birth_10MONTHS.groupby('Rcpt_Anon_ID_B', as_index=False)
Unique_baby_id = np.unique(clm_birth_10MONTHS['Rcpt_Anon_ID_B'])
clm_10MONTHS_features = pd.DataFrame(columns=['Rcpt_Anon_ID','Rcpt_Anon_ID_B'] + all_codes_in_all_columns + icd_codes_end, index=birth_registry_baby_unique_id)


# In[74]:


with tqdm(total=len(all_codes_in_all_columns)) as pbar:       
    for col in all_codes_in_all_columns:
        clm_10MONTHS_features[col].values[:] = 0
        pbar.update(1)
print('Done 1')
with tqdm(total=len(icd_codes_end)) as pbar:       
    for col in icd_codes_end:
        clm_10MONTHS_features[col].values[:] = -1
        pbar.update(1)
count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id.shape[0]) as pbar:
    for id in Unique_baby_id:
        pbar.update(1)

        baby_info = clm_group.get_group(id)
        size = baby_info.shape[0]

        for index in range(size):
            baby_id = baby_info.iloc[index]['Rcpt_Anon_ID_B']
            clm_10MONTHS_features.loc[baby_id, 'Rcpt_Anon_ID_B']=baby_id
            clm_10MONTHS_features.loc[baby_id, 'Rcpt_Anon_ID']=baby_info.iloc[index]['Rcpt_Anon_ID']

            #####################################

            assigned_code = baby_info.iloc[index]['HLTH_SRVC_CCPX_CODE']
            clm_10MONTHS_features.loc[baby_id, assigned_code]= clm_10MONTHS_features.loc[baby_id, assigned_code] + baby_info.iloc[index]['CAE_SYS_AMT']

            #####################################

            assigned_code = baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_1']
            if assigned_code =="":
                continue
            clm_10MONTHS_features.loc[baby_id, assigned_code]= clm_10MONTHS_features.loc[baby_id, assigned_code] + 1
            if clm_10MONTHS_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                clm_10MONTHS_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']



            assigned_code = baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_2']
            if assigned_code =="":
                continue
            clm_10MONTHS_features.loc[baby_id, assigned_code]= clm_10MONTHS_features.loc[baby_id, assigned_code] + 1
            if clm_10MONTHS_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                clm_10MONTHS_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']

            assigned_code = baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_3']
            if assigned_code =="":
                continue
            clm_10MONTHS_features.loc[baby_id, assigned_code]= clm_10MONTHS_features.loc[baby_id, assigned_code] + 1
            if clm_10MONTHS_features.loc[baby_id, assigned_code + '_end'] < baby_info.iloc[index]['GESTATIONAL_WEEKS']:
                clm_10MONTHS_features.loc[baby_id, assigned_code + '_end']  = baby_info.iloc[index]['GESTATIONAL_WEEKS']




# In[ ]:


atc_level = 3
new_columns={}
new_columns['Rcpt_Anon_ID'] = clm_10MONTHS_features['Rcpt_Anon_ID']
new_columns['Rcpt_Anon_ID_B'] = clm_10MONTHS_features['Rcpt_Anon_ID_B']
for col in clm_10MONTHS_features.columns[2:]:
    prefix = col[:atc_level]
    if prefix not in new_columns:
        new_columns[prefix] = clm_10MONTHS_features[col]
    else:
        new_columns[prefix] += clm_10MONTHS_features[col]
clm_10MONTHS_features_atclevel = pd.DataFrame(new_columns)

atc_level = 3
new_columns={}
new_columns['Rcpt_Anon_ID'] = clm_10MONTHS_features['Rcpt_Anon_ID']
new_columns['Rcpt_Anon_ID_B'] = clm_10MONTHS_features['Rcpt_Anon_ID_B']
for col in clm_10MONTHS_features.columns[2:]:
    if 'end' in col:
        prefix = col[:atc_level] + '_end'
        clm_10MONTHS_features[col][clm_10MONTHS_features[col] == -1] = 0
    else:
        prefix = col[:atc_level]

    if prefix not in new_columns:
        new_columns[prefix] = clm_10MONTHS_features[col]
    else:
        new_columns[prefix] += clm_10MONTHS_features[col]
clm_10MONTHS_features_atclevel = pd.DataFrame(new_columns)


# In[ ]:




# In[ ]:


clm_10MONTHS_features = clm_10MONTHS_features_atclevel


# In[5]:



# In[75]:


clm_10MONTHS_features.columns


# In[ ]:




#### END of Preprocessing ########

# # Machine Learning

# In[11]:


#importing libraries
import numpy as np
#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import catboost as cb
from tqdm import tqdm
#import cuml
#from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet, LogisticRegressionCV, ElasticNetCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
#from imblearn.combine import SMOTETomek
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from numpy import random
#from mlxtend.classifier import StackingClassifier
#from mlxtend.evaluate import confusion_matrix
import pandas as pd
import pickle
import os
import glob
import time
import copy
#import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
pd.options.mode.chained_assignment = None  # default='warn'


#pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[4]:


# functions to load pickle objects
def load_path_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


#Function to generate performance metrics
def report_proba_ytest(ytest,ypred):

    ytest = np.where(ytest > 0.5, 1, 0)
    roc_auc = roc_auc_score(ytest, ypred)
    ypred = np.where(ypred > 0.5, 1, 0)


    confusion = confusion_matrix(ytest, ypred)
    TP = confusion[1, 1]; TN = confusion[0, 0]; FP = confusion[0, 1];FN = confusion[1, 0]

    f1 = f1_score(ytest, ypred)
    acc = (TP + TN) / float(TP + TN + FP + FN)
    sens = TP / float(TP + FN)
    spec = TN / float(TN + FP)
    fpr = FP / float(TN + FP)
    prec = TP / float(TP + FP)

    return f1, roc_auc, sens,spec, prec,fpr, acc, confusion


#Function to generate performance metrics
def report_proba(ytest,ypred):

    roc_auc = roc_auc_score(ytest, ypred)
    ypred = np.where(ypred > 0.5, 1, 0)

    confusion = confusion_matrix(ytest, ypred)
    TP = confusion[1, 1]; TN = confusion[0, 0]; FP = confusion[0, 1];FN = confusion[1, 0]

    f1 = f1_score(ytest, ypred)
    acc = (TP + TN) / float(TP + TN + FP + FN)
    sens = TP / float(TP + FN)
    spec = TN / float(TN + FP)
    fpr = FP / float(TN + FP)
    prec = TP / float(TP + FP)


    return f1, roc_auc, sens,spec, prec,fpr, acc, confusion



# In[ ]:





# In[12]:


patients_train= load_path_obj('/data/padmalab/preg/data/processed/finetune_TRAIN_2009_2015_combined_CLM_AMB_INP_PIN_clm_no_conditioned.pkl')
patients_valid = load_path_obj('/data/padmalab/preg/data/processed/finetune_VALID_2016_combined_CLM_AMB_INP_PIN_clm_no_conditioned.pkl')
patients_test = load_path_obj('/data/padmalab/preg/data/processed/finetune_TEST_2017_2018_combined_CLM_AMB_INP_PIN_clm_no_conditioned.pkl')
test_ids = list(patients_test.keys())
valid_ids = list(patients_valid.keys())
train_ids = list(patients_train.keys())
del patients_train
del patients_test
del patients_valid


# In[ ]:



# In[13]:


#pickle_path = '/data/padmalab/preg/data/processed/birth_registry/birth_registry_GESTATION_LBW_SINGLE_COHORT_MIGRATION_FINAL_REMOVE_DUPLICATES_FROM_2009_fixed_TERM_values.pkl'
birth_registry_all = birth_registry #load_path_obj(pickle_path)
birth_registry_filter = birth_registry_all[birth_registry_all.Rcpt_Anon_ID_B.isin(train_ids + valid_ids + test_ids)]
pickle_path = '/data/padmalab/preg/data/processed/req04417_nob_2006_2018.pkl'
nob = load_path_obj(pickle_path)

birth_registry_all = birth_registry_filter[birth_registry_filter.Rcpt_Anon_ID_B.isin(nob.Rcpt_Anon_ID_B)]
nob = nob[nob.Rcpt_Anon_ID_B.isin(birth_registry_filter.Rcpt_Anon_ID_B)]
birth_registry_nob = pd.merge(birth_registry_all, nob, on=['Rcpt_Anon_ID_B','Rcpt_Anon_ID_M', 'YEAR'], how='inner')
selected_cols = ['SEX','LIVE','STILL','M_AGE','RES_RHA','MOMLANG','HYPERTENSION','DIABETES','INSULIN','HEPB','MHDEPRESSION','MHONANTI','MHPREVHIST','MHANXIETY','ALCOHOL','SMOKER','SMOKEQUAN','SMOKEQUANN','SMOKEQ','SMOKEQUITPR','SMOKEQUITRI','SECONDSHOME','SECONDSWORK','DRUGUSE','DRUGM','DRUGME','DRUGHE','DRUGCO','DRUGSOL','DRUGCRY','DRUGECS','OTHERDRUG','OTHERDRUGSPEC','DRUGQUITPR','SMOKENICOPROD','SMOKEECIG','SMOKEOTH','SMOKEOTHSP','ALCOFREQ','DRUGOXY','DRUGPRES','DRUGSPRESSPEC','Support','VISIT','PRENATEDUC']
birth_registry = birth_registry_nob


# In[14]:


unique_M_id_nob = np.unique(birth_registry_nob.Rcpt_Anon_ID_M)
birth_all_group = birth_registry_nob.groupby('Rcpt_Anon_ID_M', as_index=False)

birth_registry_preterm_count = pd.DataFrame(columns=['Rcpt_Anon_ID_B','Previous_Preterm', 'Previous_delivary_at_37','Preivous_normal_delivery_from38'], index=np.unique(birth_registry_nob.Rcpt_Anon_ID_B))   
for col in ['Previous_Preterm','Previous_delivary_at_37', 'Preivous_normal_delivery_from38']:
    birth_registry_preterm_count[col].values[:] = 0


# In[15]:


count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=unique_M_id_nob.shape[0]) as pbar:
    for id in unique_M_id_nob:
        pbar.update(1)

        mother_info = birth_all_group.get_group(id)
        mother_info = mother_info.sort_values(['YEAR'])
        size = mother_info.shape[0]

        for b in range(size):
            if b == 0:
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Previous_Preterm'] = 0
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Previous_delivary_at_37'] = 0
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Preivous_normal_delivery_from37'] = 0
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Preivous_normal_delivery_from38'] = 0
            else:
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Previous_Preterm'] = np.sum(mother_info.iloc[:b].GESTATION < 37)
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Previous_delivary_at_37'] = np.sum(mother_info.iloc[:b].GESTATION == 37)
                #birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Preivous_normal_delivery_from37'] = np.sum(mother_info.iloc[:b].GESTATION >= 37)
                birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Preivous_normal_delivery_from38'] = np.sum(mother_info.iloc[:b].GESTATION > 37)

            birth_registry_preterm_count.loc[mother_info.iloc[b].Rcpt_Anon_ID_B, 'Rcpt_Anon_ID_B'] = mother_info.iloc[b].Rcpt_Anon_ID_B


# In[16]:


#pickle_path = '/data/padmalab/preg/data/processed/birth_registry/birth_registry_preterm_delivary_at_37_Preivous_normal_delivery_from37_38_count.pkl'
#birth_registry_preterm_count = load_path_obj(pickle_path)


# In[ ]:





# In[17]:


#Settings
#Data Filtering
preterm_start = 26
preterm_end = 36
normal_start = 37
normal_end = 42
nulliparous = False
birth_weight_filter = True
birth_weight_threshold = 1000

prediction_time = "26Weeks"
preterm_def = 'Preterm <=36'
class_label_probability = False

LAB_features_before={'Featuretools': False, 'Last_LAB': False}
AMB_features_before={'FULL': False, 'Summary': True}
INP_features_before={'FULL': False, 'Summary': True}
PIN_features_before={'FULL': False, 'Summary': True}
CLM_features_before={'FULL': False, 'Summary': True}

LAB_features_during={'Featuretools': False, 'Last_LAB': False}
AMB_features_during={'FULL': False, 'Summary': True}
INP_features_during={'FULL': False, 'Summary': True}
PIN_features_during={'FULL': False, 'Summary': True}
CLM_features_during={'FULL': False, 'Summary': True}

icd10= [['O00', 'O08'],
        ['O10', 'O16'],
        ['O20', 'O29'],
        ['O30', 'O48'],
        ['O60', 'O75'],
        ['O94', 'O99']
       ]
icd9= [['630', '639'],
        ['640', '648'],
        ['650', '659'],
        ['660', '669'],
        ['670', '676']
       ]

w_term = 0.10 
w_preterm =0.90 


# In[18]:


TERM = np.array(birth_registry_nob.TERM)
gestation = np.array(birth_registry_nob.GESTATION)

gestation_normalized=[]
if class_label_probability:

    for x in gestation:
        if x <= preterm_end:
            gestation_normalized.append(1-((x-preterm_start)/(preterm_end-preterm_start))*0.4)
        elif x>=normal_start and x<=normal_end:
            gestation_normalized.append(0.4 - ((x-normal_start)/(normal_end-normal_start))*0.4)
        #elif x>normal_end:
        #    gestation_normalized.append(0.0)
else:
    for x in gestation:
        if x <= preterm_end:
            gestation_normalized.append(1)
        elif x>=normal_start and x<=normal_end:
            gestation_normalized.append(0)
        #elif x>normal_end:
        #    gestation_normalized.append(0.0)
print('Unique Prob. Values')
print(np.unique(gestation_normalized))

Rcpt_Anon_ID_M =  np.array(birth_registry_nob.Rcpt_Anon_ID_M)
#compute gestational age
from collections import defaultdict


def list_duplicates(seq):
    tally = defaultdict(list)
    value = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
        if TERM[i] !=1:
            value[item].append(0.0)
        else:
            value[item].append(1.0)
    return ((key, locs, value[key]) for key, locs in tally.items() if len(locs) >0)

mother_indexing = []
for dup in sorted(list_duplicates(Rcpt_Anon_ID_M)):
    mother_indexing.append(dup)


def list_duplicates(seq):
    tally = defaultdict(list)
    value = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
        value[item].append(gestation_normalized[i])
    return ((key, locs, value[key]) for key, locs in tally.items() if len(locs) >0)

mother_indexing_gestation = []
for dup in sorted(list_duplicates(Rcpt_Anon_ID_M)):
    mother_indexing_gestation.append(dup)

X= np.array(range(len(mother_indexing)))
Y = []
for i in X:
    if 1.0 in mother_indexing[i][2]:
        Y.append(1.0)
    else:
        Y.append(0.0)
Y= np.array(Y)


mother_indexing_arr = np.array(mother_indexing, dtype=object)
mother_indexing_gestation_arr = np.array(mother_indexing_gestation, dtype=object)


# In[19]:


#pickle_path = '/data/padmalab/preg/data/processed/clm_birth_10MONTHS_(with)_GESTATIONAL_WEEKS.pkl'
#clm_birth_10MONTHS = load_path_obj(pickle_path)
clm_birth_10MONTHS = clm_birth_10MONTHS[clm_birth_10MONTHS.Rcpt_Anon_ID_B.isin(birth_registry_nob.Rcpt_Anon_ID_B)]
clm_birth_10MONTHS= clm_birth_10MONTHS[clm_birth_10MONTHS.GESTATIONAL_WEEKS <26]
unique_baby_id = np.unique(clm_birth_10MONTHS.Rcpt_Anon_ID_B)
clm_group = clm_birth_10MONTHS.groupby('Rcpt_Anon_ID_B', as_index=False)


from tqdm import tqdm
#Find the first visit code or week with doctor
Unique_baby_id = np.unique(clm_birth_10MONTHS['Rcpt_Anon_ID_B'])
clm_birth_first_visit = pd.DataFrame(columns=['Rcpt_Anon_ID_B','Initial_prenatal_visit', 'total_prenatal_visits', 'V22', 'total_unique_ICD9Xcodes'], index=np.unique(birth_registry_nob.Rcpt_Anon_ID_B))   

for col in ['Initial_prenatal_visit', 'total_prenatal_visits', 'V22', 'total_unique_ICD9Xcodes']:
    clm_birth_first_visit[col].values[:] = 0


count  = 0
#np.unique(inp_birth['Rcpt_Anon_ID'])
with tqdm(total=Unique_baby_id.shape[0]) as pbar:
    for id in Unique_baby_id:
        pbar.update(1)

        baby_info = clm_group.get_group(id)
        baby_info = baby_info.sort_values(['GESTATIONAL_WEEKS'])
        size = baby_info.shape[0]

        baby_id = baby_info.iloc[0]['Rcpt_Anon_ID_B']
        clm_birth_first_visit.loc[baby_id, 'Rcpt_Anon_ID_B']=baby_id
        clm_birth_first_visit.loc[baby_id, 'total_unique_ICD9Xcodes'] = np.unique(baby_info.HLTH_DX_ICD9X_CODE_1).shape[0]
        for index in range(size):
            #clm_birth_first_visit.loc[baby_id, 'Rcpt_Anon_ID']=baby_info.iloc[index]['Rcpt_Anon_ID']

            #####################################

            if baby_info.iloc[index]['HLTH_SRVC_CCPX_CODE'] == '03.04B':
                #clm_birth_first_visit.loc[baby_id, 'Initial_prenatal_visit']=  baby_info.iloc[index]['GESTATIONAL_WEEKS']
                clm_birth_first_visit.loc[baby_id, 'total_prenatal_visits'] = 1
                #break # no have break because some patients might have 2nd one - 1st one does not have anything confirm pregnancy
            if baby_info.iloc[index]['HLTH_SRVC_CCPX_CODE'] == '03.03B':
                clm_birth_first_visit.loc[baby_id, 'total_prenatal_visits'] +=  1
            if baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_1'][:3] == 'V22':
                clm_birth_first_visit.loc[baby_id, 'V22'] +=  1
            if pd.isnull(clm_birth_first_visit.loc[baby_id, 'Initial_prenatal_visit']) & ((baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_1'][:3] == 'V22') | (baby_info.iloc[index]['HLTH_DX_ICD9X_CODE_1'][:3] == 'V23')):
                clm_birth_first_visit.loc[baby_id, 'Initial_prenatal_visit']=  baby_info.iloc[index]['GESTATIONAL_WEEKS']
                clm_birth_first_visit.loc[baby_id, 'total_prenatal_visits'] += 1


# In[48]:


#For all data
other_features = birth_registry_nob[selected_cols + ['Rcpt_Anon_ID_B']]
other_features_lastlab = other_features #pd.merge(other_features, lastlab, on=['Rcpt_Anon_ID_B'], how='left')
other_features_lastlab = other_features_lastlab.merge(clm_birth_first_visit, on='Rcpt_Anon_ID_B',how='left')
other_features_lastlab = other_features_lastlab.merge(birth_registry_preterm_count, on='Rcpt_Anon_ID_B',how='left')


# In[50]:


other_features_lastlab.RES_RHA = other_features_lastlab.RES_RHA.apply(lambda x: 'Urban' if x in ['R2', 'R4', 'R5', 'R7', 'R8', 'R3', 'R6'] else ('Rural' if x in ['R1', 'R9'] else np.nan))
other_features_lastlab.VISIT = other_features_lastlab.VISIT.replace('0',0)
other_features_lastlab.VISIT = other_features_lastlab.VISIT.replace('1-3',1)
other_features_lastlab.VISIT = other_features_lastlab.VISIT.replace('4-8',2)
other_features_lastlab.VISIT = other_features_lastlab.VISIT.replace('9+',3)
other_features_lastlab.VISIT = other_features_lastlab.VISIT.replace('',np.nan)
other_features_lastlab.VISIT = other_features_lastlab.VISIT.replace('999',np.nan)


other_features_lastlab.STILL = other_features_lastlab.STILL.replace(99.0,np.nan)
other_features_lastlab.M_AGE = other_features_lastlab.M_AGE.replace(-1.0,np.nan)
other_features_lastlab.M_AGE = other_features_lastlab.M_AGE.replace(0.0,np.nan)
other_features_lastlab.M_AGE = other_features_lastlab.M_AGE.replace(6.0,np.nan)
other_features_lastlab.M_AGE = other_features_lastlab.M_AGE.replace(999.0,np.nan)

other_features_lastlab.MOMLANG = other_features_lastlab.MOMLANG.replace(9.0,np.nan)
other_features_lastlab.HYPERTENSION = other_features_lastlab.HYPERTENSION.replace(9.0,np.nan)
other_features_lastlab.DIABETES = other_features_lastlab.DIABETES.replace(9.0,np.nan)
other_features_lastlab.INSULIN = other_features_lastlab.INSULIN.replace(9.0,np.nan)
other_features_lastlab.HEPB = other_features_lastlab.HEPB.replace(9.0,np.nan)
other_features_lastlab.ALCOHOL = other_features_lastlab.ALCOHOL.replace(9.0,np.nan)
other_features_lastlab.SMOKER = other_features_lastlab.SMOKER.replace(9.0,np.nan)
other_features_lastlab.ALCOFREQ = other_features_lastlab.ALCOFREQ.replace(9.0,np.nan)
#other_features_lastlab.MFOLIC = other_features_lastlab.MFOLIC.replace(9.0,np.nan)
other_features_lastlab.PRENATEDUC = other_features_lastlab.PRENATEDUC.replace(9.0,np.nan)


# In[51]:


other_features_numpy = other_features_lastlab.to_numpy()
features_list = []
for index, value in enumerate(list(other_features_lastlab.columns)):
    features_list.append(str(value))


# In[52]:


# SUMMARY ATC/ICD CODES

if PIN_features_before['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/pin_1year_features_atclevel_3.pkl'
    #pin_1year_features = load_path_obj(pickle_path)
    pin_1year_features['Rcpt_Anon_ID_B']=pin_1year_features.index
    pin_1year_features = pin_1year_features[pin_1year_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    pin_1year_features_numpy = pin_1year_features.to_numpy()[:,2:]
    assert sum(np.array(pin_1year_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0
    #del pin_1year_features

if PIN_features_during['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/'+prediction_time+'/pin_ATC3_26Weeks_features.pkl'

    #pin_10MONTHS_features  = load_path_obj(pickle_path) 
    pin_10MONTHS_features['Rcpt_Anon_ID_B']=pin_10MONTHS_features.index
    pin_10MONTHS_features = pin_10MONTHS_features[pin_10MONTHS_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    pin_10MONTHS_features_numpy = pin_10MONTHS_features.to_numpy()[:,2:]
    assert sum(np.array(pin_10MONTHS_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0
    #del pin_10MONTHS_features



################################################################
if CLM_features_before['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/clm_birth_1year_features_atclevel_3.pkl'

    #clm_birth_1year_features  = load_path_obj(pickle_path)
    clm_birth_1year_features['Rcpt_Anon_ID_B']=clm_birth_1year_features.index
    clm_birth_1year_features = clm_birth_1year_features[clm_birth_1year_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    clm_birth_1year_features_numpy = clm_birth_1year_features.to_numpy()[:,2:]

    assert sum(np.array(clm_birth_1year_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0

    new_columns={}
    new_columns['Rcpt_Anon_ID'] = clm_birth_1year_features['Rcpt_Anon_ID']
    new_columns['Rcpt_Anon_ID_B'] = clm_birth_1year_features['Rcpt_Anon_ID_B']

    for index, icds in enumerate(icd9):
        clm_features = clm_birth_1year_features[[col for col in clm_birth_1year_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
        new_columns['clm_before_grp_'+str(index)] = clm_features.sum(axis=1)

    clm_1Y_features_ICDs_grps = pd.DataFrame(new_columns)
    clm_1Y_features_ICDs_grps_numpy = clm_1Y_features_ICDs_grps.to_numpy()[:,2:]
    #del clm_birth_1year_features


if CLM_features_during['Summary']:

    #pickle_path = '/data/padmalab/preg/data/processed/'+prediction_time+'/clm_birth_26Weeks_features_fixed_endcode2_ICD3level.pkl'
    #clm_10MONTHS_features  = load_path_obj(pickle_path)
    clm_10MONTHS_features['Rcpt_Anon_ID_B']=clm_10MONTHS_features.index
    clm_10MONTHS_features = clm_10MONTHS_features[clm_10MONTHS_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    clm_10MONTHS_features_numpy = clm_10MONTHS_features.to_numpy()[:,2:]
    assert sum(np.array(clm_10MONTHS_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0


    pickle_path = '/data/padmalab/preg/data/processed/'+prediction_time+'/HLTH_DX_ICD9X_CODE_list.pkl'
    #pickle_path = '/data/padmalab/preg/data/processed/28Weeks/clm_birth_26Weeks_features_fixed_endcode1.pkl'
    HLTH_DX_ICD9X_CODE_list = load_path_obj(pickle_path)
    #clm_10MONTHS_features[[x for x in list(clm_10MONTHS_features.columns) if '_end' not in x][2:]]
    visit_counts_after = clm_10MONTHS_features[HLTH_DX_ICD9X_CODE_list].sum(axis=1).to_numpy()
    other_features_numpy = np.concatenate( (other_features_numpy, visit_counts_after.reshape(-1,1)), axis =1)
    features_list.append("CLM_preg_visit_counts")

    new_columns={}
    new_columns['Rcpt_Anon_ID'] = clm_10MONTHS_features['Rcpt_Anon_ID']
    new_columns['Rcpt_Anon_ID_B'] = clm_10MONTHS_features['Rcpt_Anon_ID_B']

    for index, icds in enumerate(icd9):
        clm_features = clm_10MONTHS_features[[col for col in clm_10MONTHS_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
        new_columns['clm_during_grp_'+str(index)] = clm_features.sum(axis=1)

    '''
    codes_complications = [['630', '639'], ['640', '648'], 'V23']
    for index, icds in enumerate(codes_complications):
        if index <2:
            temp = clm_10MONTHS_features[[col for col in clm_10MONTHS_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
            new_columns[str(index)] = temp.sum(axis=1)
        else:
            new_columns[str(index)] = clm_10MONTHS_features[icds]
    temp1 = pd.DataFrame(new_columns)
    new_columns['preg_complications_binary'] = (temp1['0']+temp1['1'] + temp1['2'] >= 1).astype(int)
    new_columns['preg_complications_raw'] = (temp1['0']+temp1['1'] + temp1['2'] >= 1).astype(int)
    '''


    clm_10MONTHS_features_ICDs_grps = pd.DataFrame(new_columns)
    clm_10MONTHS_features_ICDs_grps_numpy = clm_10MONTHS_features_ICDs_grps.to_numpy()[:,2:]



######################################################################
if AMB_features_before['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/amb_birth_1year_features_all_level_3.pkl'
    #amb_1year_features  = load_path_obj(pickle_path)
    amb_1year_features['Rcpt_Anon_ID_B']=amb_1year_features.index
    amb_1year_features = amb_1year_features[amb_1year_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    amb_1year_features_numpy = amb_1year_features.to_numpy()[:,2:]
    assert sum(np.array(amb_1year_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0


    new_columns={}
    new_columns['Rcpt_Anon_ID'] = amb_1year_features['Rcpt_Anon_ID']
    new_columns['Rcpt_Anon_ID_B'] = amb_1year_features['Rcpt_Anon_ID_B']

    for index, icds in enumerate(icd10):
        amb_features = amb_1year_features[[col for col in amb_1year_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
        new_columns['amb_before_grp_'+str(index)] = amb_features.sum(axis=1)

    amb_1Y_features_ICDs_grps = pd.DataFrame(new_columns)
    amb_1Y_features_ICDs_grps_numpy = amb_1Y_features_ICDs_grps.to_numpy()[:,2:]
    #del amb_1year_features


if AMB_features_during['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/'+prediction_time+'/amb_birth_26Weeks_features.pkl'

    #amb_birth_10months_features  = load_path_obj(pickle_path)
    amb_birth_10months_features['Rcpt_Anon_ID_B']=amb_birth_10months_features.index
    amb_birth_10months_features = amb_birth_10months_features[amb_birth_10months_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    amb_birth_10months_features_numpy = amb_birth_10months_features.to_numpy()[:,2:]
    assert sum(np.array(amb_birth_10months_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0

    new_columns={}
    new_columns['Rcpt_Anon_ID'] = amb_birth_10months_features['Rcpt_Anon_ID']
    new_columns['Rcpt_Anon_ID_B'] = amb_birth_10months_features['Rcpt_Anon_ID_B']

    for index, icds in enumerate(icd10):
        amb_features = amb_birth_10months_features[[col for col in amb_birth_10months_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
        new_columns['amb_during_grp_'+str(index)] = amb_features.sum(axis=1)

    amb_birth_10months_features_ICDs_grps = pd.DataFrame(new_columns)
    amb_birth_10months_features_ICDs_grps_numpy = amb_birth_10months_features_ICDs_grps.to_numpy()[:,2:]
    #del amb_birth_10months_features


if INP_features_before['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/inp_birth_1year_features_corrected2_level_3.pkl'# incorrect file (calculation wrong) --> inp_birth_1year_features_all.pkl' --> fixed nan value in SWIP value
    #inp_1year_features  = load_path_obj(pickle_path)
    inp_1year_features['Rcpt_Anon_ID_B']=inp_1year_features.index
    inp_1year_features = inp_1year_features[inp_1year_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    inp_1year_features_numpy = inp_1year_features.to_numpy()[:,2:]
    assert sum(np.array(inp_1year_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0

    new_columns={}
    new_columns['Rcpt_Anon_ID'] = inp_1year_features['Rcpt_Anon_ID']
    new_columns['Rcpt_Anon_ID_B'] = inp_1year_features['Rcpt_Anon_ID_B']

    for index, icds in enumerate(icd10):
        inp_features = inp_1year_features[[col for col in inp_1year_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
        new_columns['inp_before_grp_'+str(index)] = inp_features.sum(axis=1)

    inp_1Y_features_ICDs_grps = pd.DataFrame(new_columns)
    inp_1Y_features_ICDs_grps_numpy = inp_1Y_features_ICDs_grps.to_numpy()[:,2:]
    #del inp_1year_features


if INP_features_during['Summary']:
    #pickle_path = '/data/padmalab/preg/data/processed/'+prediction_time+'/inp_birth_26Weeks_features_V2corrected.pkl'
    #inp_10MONTHS_features  = load_path_obj(pickle_path)
    inp_10MONTHS_features['Rcpt_Anon_ID_B']=inp_10MONTHS_features.index
    inp_10MONTHS_features = inp_10MONTHS_features[inp_10MONTHS_features['Rcpt_Anon_ID_B'].isin(birth_registry.Rcpt_Anon_ID_B)]
    inp_10MONTHS_features_numpy = inp_10MONTHS_features.to_numpy()[:,2:]
    assert sum(np.array(inp_10MONTHS_features.Rcpt_Anon_ID_B) != np.array(birth_registry.Rcpt_Anon_ID_B)) == 0

    new_columns={}
    new_columns['Rcpt_Anon_ID'] = inp_10MONTHS_features['Rcpt_Anon_ID']
    new_columns['Rcpt_Anon_ID_B'] = inp_10MONTHS_features['Rcpt_Anon_ID_B']

    for index, icds in enumerate(icd10):
        inp_features = inp_10MONTHS_features[[col for col in inp_10MONTHS_features.columns if icds[0] < col < icds[1] and not col.endswith('_end')]]#col.startswith('O') and not col.endswith('_end')]]
        new_columns['inp_during_grp_'+str(index)] = inp_features.sum(axis=1)

    inp_10MONTHS_features_ICDs_grps = pd.DataFrame(new_columns)
    inp_10MONTHS_features_ICDs_grps_numpy = inp_10MONTHS_features_ICDs_grps.to_numpy()[:,2:]

    #del inp_10MONTHS_features


# In[53]:


def get_pin_10MONTHS_features(X):
    X = X.reshape((X.shape[0]))
    temp  = pin_10MONTHS_features_numpy[X,:]
    #print(temp.shape)
    return temp

def get_pin_last_7MONTHS_features(X):
    X = X.reshape((X.shape[0]))
    temp = pin_last_7MONTHS_features_numpy[X,:]
    #print(temp.shape)
    return temp

def get_pin_1year_features(X):
    X = X.reshape((X.shape[0]))
    temp = pin_1year_features_numpy[X,:]
    #print(temp.shape)
    return temp

def get_clm_10MONTHS_features(X):
    X = X.reshape((X.shape[0]))
    temp = clm_10MONTHS_features_numpy[X,:]
    #print(temp.shape)
    return temp

def get_clm_birth_1year_features(X):
    X = X.reshape((X.shape[0]))
    temp = clm_birth_1year_features_numpy[X,:]
    #print(temp.shape)
    return temp


def get_amb_birth_10months_features(X):
    X = X.reshape((X.shape[0]))
    temp = amb_birth_10months_features_numpy[X,:]
    #print(temp.shape)
    return temp

def get_amb_1year_features(X):
    X = X.reshape((X.shape[0]))
    temp = amb_1year_features_numpy[X,:]
    return temp


def get_inp_10MONTHS_features(X):
    X = X.reshape((X.shape[0]))
    temp = inp_10MONTHS_features_numpy[X,:]
    #print(temp.shape)

    return temp

def get_inp_1year_features(X):
    X = X.reshape((X.shape[0]))
    temp = inp_1year_features_numpy[X,:]
    return temp

def get_other_features(X):
    X = X.reshape((X.shape[0]))
    temp = other_features_numpy[X,:]
    return temp



# In[54]:


get_data_fun= [get_other_features]
length_arr=[other_features_numpy.shape[1]]

if AMB_features_before['FULL']:
    get_data_fun = get_data_fun + [get_amb_1year_features_full]
    length_arr = length_arr + [amb_1year_features_full_numpy.shape[1]]
if AMB_features_before['Summary']:
    get_data_fun = get_data_fun + [get_amb_1year_features]
    length_arr = length_arr + [amb_1year_features_numpy.shape[1]]

if AMB_features_during['FULL']:
    get_data_fun = get_data_fun + [ get_amb_birth_10months_features_full]
    length_arr = length_arr + [amb_birth_10months_features_full_numpy.shape[1]]
if AMB_features_during['Summary']:
    get_data_fun = get_data_fun + [get_amb_birth_10months_features]
    length_arr = length_arr + [amb_birth_10months_features_numpy.shape[1]]

for index, value in enumerate(list(amb_1year_features.columns[2:])):
    features_list.append('AMB1Y_'+value)
for index, value in enumerate(list(amb_birth_10months_features.columns[2:])):
    features_list.append('AMB26W_'+value)



if INP_features_before['FULL']:
    get_data_fun = get_data_fun + [get_inp_1year_features_full]
    length_arr = length_arr + [inp_1year_features_full_numpy.shape[1]]
if INP_features_before['Summary']:
    get_data_fun = get_data_fun + [get_inp_1year_features]
    length_arr = length_arr + [inp_1year_features_numpy.shape[1]]

if INP_features_during['FULL']:
    get_data_fun = get_data_fun + [ get_inp_10MONTHS_features_full]
    length_arr = length_arr + [inp_10MONTHS_features_full_numpy.shape[1]]
if INP_features_during['Summary']:
    get_data_fun = get_data_fun + [get_inp_10MONTHS_features]
    length_arr = length_arr + [inp_10MONTHS_features_numpy.shape[1]]


for index, value in enumerate(list(inp_1year_features.columns[2:])):
    features_list.append('INP1Y_'+value)
for index, value in enumerate(list(inp_10MONTHS_features.columns[2:])):
    features_list.append('INP26W_'+value)


####################################

if PIN_features_before['FULL']:
    get_data_fun = get_data_fun + [get_pin_1year_features_full]
    length_arr = length_arr + [pin_1year_features_full_numpy.shape[1]]
if PIN_features_before['Summary']:
    get_data_fun = get_data_fun + [get_pin_1year_features]
    length_arr = length_arr + [pin_1year_features_numpy.shape[1]]

if PIN_features_during['FULL']:
    get_data_fun = get_data_fun + [ get_pin_10MONTHS_features_full]
    length_arr = length_arr + [pin_10MONTHS_features_full_numpy.shape[1]]
if PIN_features_during['Summary']:
    get_data_fun = get_data_fun + [get_pin_10MONTHS_features]
    length_arr = length_arr + [pin_10MONTHS_features_numpy.shape[1]]

for index, value in enumerate(list(pin_1year_features.columns[2:])):
    features_list.append('PIN1Y_'+value)
for index, value in enumerate(list(pin_10MONTHS_features.columns[2:])):
    features_list.append('PIN26W_'+value)



if CLM_features_before['Summary']:
    get_data_fun = get_data_fun + [get_clm_birth_1year_features]
    length_arr = length_arr + [clm_birth_1year_features_numpy.shape[1]]
if CLM_features_during['Summary']:
    get_data_fun = get_data_fun + [get_clm_10MONTHS_features]
    length_arr = length_arr + [clm_10MONTHS_features_numpy.shape[1]]

for index, value in enumerate(list(clm_birth_1year_features.columns[2:])):
    features_list.append('CLM1Y_'+value)
for index, value in enumerate(list(clm_10MONTHS_features.columns[2:])):
    features_list.append('CLM26W_'+value)


#get_lab_6M_features_last, get_lab_1year_features, get_lab_6M_features,


# In[74]:


len(features_list)
new_features_list=[]
for f in features_list:
    if f not in ['Rcpt_Anon_ID_B', 'V22', 'pin_26W', 'pin_1y', 'pin_sum_26W_1y','pin_26W_binary','pin_1y_binary','pin_sum_26W_1y_binary','pin_sum_26W_1y_binary_binary','Preivous_normal_delivery_from37','Initial_prenatal_visit','total_unique_ICD9Xcodes','AMB1Y_SEPI_CACS_RIW_VALUE']:
        new_features_list.append(f)
len(new_features_list)


# In[71]:


# In[55]:


get_data_fun


# In[56]:


threshold = 0.01
tf_data_process = []
for index, data_fun in enumerate(get_data_fun):

    tf_name = str(data_fun.__name__)
    print(tf_name)
    temp_tf = FunctionTransformer(func= data_fun, validate=False, accept_sparse=False)
    tf_data_process.append((tf_name, Pipeline(steps=[('get_data', temp_tf)]), slice(0,length_arr[index],1)))

concatenate_pipe = ColumnTransformer(transformers=tf_data_process, verbose=True)


# In[57]:


birth_registry_filter_reset_index = birth_registry_nob.reset_index(drop=True)


# In[58]:


y_test =  np.where( np.array(birth_registry_filter_reset_index[birth_registry_filter_reset_index.YEAR>2016].GESTATION) <37, 1, 0)
X_test = np.array(birth_registry_filter_reset_index[birth_registry_filter_reset_index.YEAR>2016].index)
print((X_test.shape[0]/birth_registry_filter_reset_index.shape[0])*100)
condition = (birth_registry_filter_reset_index.YEAR>2015) &(birth_registry_filter_reset_index.YEAR < 2017)
y_valid =  np.where( np.array(birth_registry_filter_reset_index[condition].GESTATION) <37, 1, 0)
X_valid = np.array(birth_registry_filter_reset_index[condition].index)
print((X_valid.shape[0]/birth_registry_filter_reset_index.shape[0])*100)
y_train =  np.where( np.array(birth_registry_filter_reset_index[birth_registry_filter_reset_index.YEAR<=2015].GESTATION) <37, 1, 0)
X_train = np.array(birth_registry_filter_reset_index[birth_registry_filter_reset_index.YEAR <=2015].index)
print((X_train.shape[0]/birth_registry_filter_reset_index.shape[0])*100)
#y_train = np.array(class_probability(birth_registry_filter_reset_index.GESTATION.iloc[X_train]))


# In[59]:


X_train = np.reshape(X_train,(X_train.size,1))
X_valid = np.reshape(X_valid,(X_valid.size,1))
X_test = np.reshape(X_test,(X_test.size,1))


print(X_train.shape)
#gs.fit(X_train, y_train)
concatenate_pipe.fit(X_train)
X_train_transformed = concatenate_pipe.transform(X_train)
X_valid_transformed = concatenate_pipe.transform(X_valid)
X_test_transformed = concatenate_pipe.transform(X_test)

eval_set_xgboost=[(X_train_transformed, y_train), (X_valid_transformed, y_valid)]


# In[76]:


X_test_transformed_df = pd.DataFrame(data=X_test_transformed, columns=features_list)
X_valid_transformed_df = pd.DataFrame(data=X_valid_transformed, columns=features_list)
X_train_transformed_df = pd.DataFrame(data=X_train_transformed, columns=features_list)

X_test_transformed = X_test_transformed_df[new_features_list].to_numpy()
X_valid_transformed = X_valid_transformed_df[new_features_list].to_numpy()
X_train_transformed = X_train_transformed_df[new_features_list].to_numpy()


# In[77]:


non_float_columns=[]
for col_idx in range(X_train_transformed.shape[1]):
    try:
        _ = X_train_transformed[:, col_idx].astype(float)
    except ValueError:
        non_float_columns.append(col_idx)


for col_idx in range(X_valid_transformed.shape[1]):
    try:
        _ = X_valid_transformed[:, col_idx].astype(float)
    except ValueError:
        if col_idx not in non_float_columns:
            non_float_columns.append(col_idx)

for col_idx in range(X_test_transformed.shape[1]):
    try:
        _ = X_test_transformed[:, col_idx].astype(float)
    except ValueError:
        if col_idx not in non_float_columns:
            non_float_columns.append(col_idx)


# In[78]:


# Categorical values are converted to str
categorical_features_indices = non_float_columns

#[0,4,6] + lab_str_features_indeces + non_float_columns
for index, feature_index in enumerate(non_float_columns):
    X_train_transformed[:, feature_index] = X_train_transformed[:, feature_index].astype(str)
    X_valid_transformed[:, feature_index] = X_valid_transformed[:, feature_index].astype(str)
    X_test_transformed[:, feature_index] = X_test_transformed[:, feature_index].astype(str)


# In[62]:


sum(y_test==1)


# In[79]:

if class_label_probability:
    train_weights=[w_term if y_train[i] < 0.5 else w_preterm for i in range(X_train_transformed.shape[0])]
else:
    train_weights=[w_term if y_train[i] < 1 else w_preterm for i in range(X_train_transformed.shape[0])]

y_train_binary = np.where(y_train > 0.5, 1, 0)

train_pool = cb.Pool(data=X_train_transformed, label=y_train_binary, thread_count = -1, weight = train_weights, cat_features = categorical_features_indices)
#train_pool = cb.Pool(data=X_train_transformed, label=y_train, thread_count = -1, cat_features = categorical_features_indices)
#del X_train_transformed
#del y_train 
if class_label_probability:
    y_valid = np.where(y_valid > 0.5, 1, 0)
valid_pool = cb.Pool(data=X_valid_transformed, label=y_valid, thread_count = -1, cat_features = categorical_features_indices)
#del X_valid_transformed
#del y_valid
if class_label_probability:
    y_test = np.where(y_test > 0.5, 1, 0)
test_pool = cb.Pool(data=X_test_transformed, label=y_test, thread_count = -1, cat_features = categorical_features_indices)
#del X_test_transformed
#del y_test


# In[80]:


#working
model_params={
    
    'task_type': 'GPU',
    'devices':'5:6:7',
    'loss_function': 'CrossEntropy',
    

    'iterations': 3000,
    'early_stopping_rounds': 500,
    'eval_metric' : 'AUC',
    'learning_rate' : 0.009,
    'depth' : 10,#<--
    'bagging_temperature':1.0,'random_strength':1,  
    'max_bin': 512, 'grow_policy': 'SymmetricTree','min_data_in_leaf':3,
    'random_seed': 1.0, 'thread_count': -1, 'verbose': 1,
}

model = cb.CatBoost(model_params)


# In[81]:

model.fit(train_pool, verbose=False, plot=True, eval_set=valid_pool, use_best_model = True)


# In[82]:


len(new_features_list) == X_train_transformed.shape[1]


# In[83]:


feature_importances


# In[86]:
###### VALIDATION PART #####################


import matplotlib.pyplot as plt
feature_importances = model.get_feature_importance(train_pool)
sorted_idx = np.argsort(feature_importances)
fig = plt.figure(figsize=(16, 15))
plt.barh(range(21), feature_importances[sorted_idx[-21:]], tick_label=np.array(new_features_list)[sorted_idx[-21:]])
plt.show()

# In[93]:


from sklearn.metrics import roc_auc_score,classification_report, det_curve,precision_recall_fscore_support, brier_score_loss
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix, roc_curve, auc, RocCurveDisplay 
from sklearn.metrics import DetCurveDisplay, f1_score, ConfusionMatrixDisplay, average_precision_score, precision_recall_curve
import sklearn.metrics as metrics
from sklearn.calibration import calibration_curve

# import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns
from scipy.stats import hmean
import numpy as np
import pandas as pd

from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

#function to plot results    
def evalplots(y_test,y_score,y_pred,labels, creport_dict=None, thrplot = False):
    '''

    '''
    precision, recall, thr = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    f1score = f1_score(y_test, y_pred)
    f1vec = [hmean([precision[i],recall[i]]) for i in range(sum(recall!=0))]

    #plt.plot([i/len(f1vec) for i in range(len(f1vec))],f1vec,color='r',alpha=0.2)
    plt.figure(figsize = (15,7))
    plt.subplot(1, 2, 1)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
    if creport_dict:
        plt.plot([creport_dict['rec'], 0], [creport_dict['prec'], creport_dict['prec']], color = 'blue', linestyle='--')
        plt.plot([creport_dict['rec'], creport_dict['rec']], [creport_dict['prec'], 0], color = 'blue', linestyle='--')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}, F1={1:0.2f}'.format(average_precision,f1score))
    #plt.show()

    # Compute ROC curve
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.title('Receiver Operating Characteristic')
    # print ('creport_dict', creport_dict)
    if creport_dict:
        tfrate = creport_dict['tp']/(creport_dict['tp']+creport_dict['fn'])
        fprate = creport_dict['fp']/(creport_dict['fp']+creport_dict['tn'])
        plt.plot([fprate, 0], [tfrate, tfrate], color = 'blue', linestyle='--')
        plt.plot([fprate, fprate], [0, tfrate], color = 'blue', linestyle='--')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.tight_layout()
    plt.show()

    if thrplot:
        plt.step( thr[recall[:-1]!=0],f1vec,color='r',alpha=0.2,where='post')
        plt.fill_between(thr[recall[:-1]!=0],f1vec,step='post', alpha=0.2,color='r')
        plt.xlabel('Threshold')
        plt.ylabel('Estimated F1-Scores')
        plt.ylim([0.0, 1.0])
        plt.axvline(x=0.5,color ='r')
        plt.title('Threshold Vs F1-Score: Max F1 ={0:0.2f}, Reported F1={1:0.2f}'.format(np.max(f1vec),f1score))
        plt.show()        

        #plt.rcParams["figure.figsize"] = (5, 5)
        plt.step(precision[:-1], thr, color='b', alpha=0.2, where='post')
        plt.fill_between(precision[:-1], thr, alpha=0.2, color='b', step='post')
        plt.xlabel('precision')
        plt.ylabel('Threshold')
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.grid()
        plt.show()

        #plt.rcParams["figure.figsize"] = (5, 5)
        plt.step(recall[:-1], thr, color='b', alpha=0.2, where='post')
        plt.fill_between(recall[:-1], thr, alpha=0.2, color='b', step='post')
        plt.xlabel('Recall')
        plt.ylabel('Threshold')
        plt.xticks(np.arange(0, 1, step=0.1))
        plt.yticks(np.arange(0, 1, step=0.1))
        plt.grid()
        plt.show()


#classification report
def class_report(y_test, y_pred, y_score, verbose=True):
    acc = (y_pred == y_test).mean()
    roc = roc_auc_score(y_test, y_score)
    f1 = f1_score(y_test, y_pred, average='binary')
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) 

    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred)
    auprc = metrics.auc(lr_recall, lr_precision)

    aprec = average_precision_score(y_test, y_score)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    br_score = brier_score_loss(y_test, y_score, pos_label=y_test.max())
    spec = tn / (tn+fp)

    if (verbose):
        print("AUROC score: {0:,.4f}".format(roc))
        print("AUPRC score: {0:,.4f}".format(auprc))     #TODO AUPRC
        print('Average precision-recall score: {0:,.4f}'.format(aprec))
        print("Accuracy score: {0:,.4f}".format(acc))
        print("Sensitivity / Recall score: {0:,.4f}".format(rec))
        print("Specificity score: {0:,.4f}".format(spec))
        print("Positive predictive value / Precision score: {0:,.4f}".format(prec))
        print("f1 score: {0:,.4f}".format(f1))
        print("Brier score: {0:,.4f}".format(br_score))
        print()

    return {
        'accuracy': acc,
        'auroc': roc,
        'auprc': auprc,
        'f1_score': f1,
        'prec': prec,
        'rec': rec,
        'spec': spec,
        'aprec': aprec,
        'br_score': br_score,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    } 

#plot the calibration curve
def plot_calib_curve(y_test,y_score, nbins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_score, n_bins = nbins)
    fig = plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "o-", label="Catboost", linewidth=3)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Model", linewidth=1.5)
    plt.title('Calibration Plot')

    plt.xlabel('Mean Predicted Score')
    plt.ylabel('Fraction of True Positives')
    plt.legend()
    plt.show()


def get_optimal_cutoff_original(y_test,y_prob,text_labels):
    # get optimal cutoff based on tpr - fpr should be max : Youden's J-Score
    n_classes = len(text_labels)
    y_test = y_test
    y_score = y_prob

    # Compute ROC curve and ROC area for each class
    roc = dict()
    roc_auc = dict()
    roc['tpr'] = dict()
    roc['fpr'] = dict()
    roc['thr'] = dict()
    roc_df = dict()
    roc_tf_thr = dict()
    roc_j_thr = dict()
    roc_j_thr_dict = dict()

    for i in range(n_classes):
        roc['fpr'][i], roc['tpr'][i], roc['thr'][i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[text_labels[i]] = auc(roc['fpr'][i], roc['tpr'][i])
        j = np.arange(len(roc['tpr'][i])) 
        roc_df[text_labels[i]] = pd.DataFrame({'tf' : pd.Series(roc['tpr'][i]-(1- roc['fpr'][i]), index=j),'j_score' : pd.Series((roc['tpr'][i]- roc['fpr'][i]), index=j), 'threshold' : pd.Series(roc['thr'][i], index=j)})
        roc_tf_thr[text_labels[i]] = roc_df[text_labels[i]].iloc[(roc_df[text_labels[i]].tf-0).abs().argsort()[:1]]
        roc_j_thr[text_labels[i]] = roc_df[text_labels[i]].iloc[(roc_df[text_labels[i]].j_score).argsort()[-1:]]
        roc_j_thr_dict[text_labels[i]] =  roc_j_thr[text_labels[i]]['threshold'].values[0]
    return roc_auc,roc_df, roc_tf_thr, roc_j_thr, roc_j_thr_dict

def get_optimal_cutoff(y_test,y_prob,text_labels):
    # get optimal cutoff based on tpr - fpr should be max : Youden's J-Score
    n_classes = len(text_labels)
    y_test = y_test
    y_score = y_prob

    # Compute ROC curve and ROC area for each class
    roc = dict()
    roc_auc = dict()
    roc['tpr'] = dict()
    roc['fpr'] = dict()
    roc['thr'] = dict()
    roc_df = dict()
    roc_tf_thr = dict()
    roc_j_thr = dict()
    roc_j_thr_dict = dict()

    for i in range(n_classes):
        roc['fpr'][i], roc['tpr'][i], roc['thr'][i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[text_labels[i]] = auc(roc['fpr'][i], roc['tpr'][i])
        j = np.arange(len(roc['tpr'][i])) 
        roc_df[text_labels[i]] = pd.DataFrame({'tf' : pd.Series(roc['tpr'][i]-(1- roc['fpr'][i]), index=j),'j_score' : pd.Series((roc['tpr'][i]- roc['fpr'][i]), index=j), 'threshold' : pd.Series(roc['thr'][i], index=j)})
        roc_tf_thr[text_labels[i]] = roc_df[text_labels[i]].iloc[(roc_df[text_labels[i]].tf-0).abs().argsort()[:1]]
        roc_j_thr[text_labels[i]] = roc_df[text_labels[i]].iloc[(roc_df[text_labels[i]].j_score).argsort()[-1:]]
        roc_j_thr_dict[text_labels[i]] =  roc_j_thr[text_labels[i]]['threshold'].values[0]
    return roc_auc,roc_df, roc_tf_thr, roc_j_thr, roc_j_thr_dict


def get_pred_report(y_test,y_prob,text_labels,roc_j_thr_dict, verbose = True):
    n_classes = len(text_labels)
    y_test = y_test
    y_score = y_prob
    class_df_dict = dict()
    creport_dict = dict()

    for i in range(n_classes):
        j = np.arange(len(y_score[:, i])) 
        class_df_dict[text_labels[i]] = pd.DataFrame({'true' : pd.Series(y_test[:, i], index=j),'pred_proba' : pd.Series(y_score[:, i], index=j)})
        class_df_dict[text_labels[i]]['pred'] = class_df_dict[text_labels[i]]['pred_proba'].map(lambda x: 1 if x > roc_j_thr_dict[text_labels[i]] else 0)
        #printmd('**' + text_labels[i] + '**')
        #print(confusion_matrix(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred']))
        #print('Cutoff Probability based on Training ROC: ', roc_j_thr_dict[text_labels[i]])
        if verbose:
            print (classification_report(y_test[:, i], class_df_dict[text_labels[i]]['pred'], target_names=['0','1']))

        creport_dict[text_labels[i]] = class_report(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred'], class_df_dict[text_labels[i]]['pred_proba'], verbose)
        if verbose:
            cm = confusion_matrix(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred'])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6,6))
            disp.plot(ax=ax) 
            evalplots(class_df_dict[text_labels[i]]['true'], class_df_dict[text_labels[i]]['pred_proba'], 
                      class_df_dict[text_labels[i]]['pred'], [0, 1], creport_dict[text_labels[i]])
            plot_calib_curve(class_df_dict[text_labels[i]]['true'],class_df_dict[text_labels[i]]['pred_proba'], nbins=10)

        #print('----------------------------------------')

    return class_df_dict, creport_dict



# In[94]:


from venn_abers import VennAbersCalibrator
import sys

train_y_true = np.squeeze(np.eye(2)[np.where(y_train > 0.5, 1, 0).astype(int).reshape(-1)]) #y_train#data['train_y_true']
test_y_prob = model.predict(data = test_pool, prediction_type='Probability')#[:,1]#model.predict_proba(X_test_transformed)#data['test_y_prob']


VAC = VennAbersCalibrator()

valid_y_true = y_valid#np.squeeze(np.eye(2)[np.where(y_valid > 0.5, 1, 0).astype(int).reshape(-1)]) #y_train#data['train_y_true']
valid_y_prob = model.predict(data = valid_pool, prediction_type='Probability')#[:,1]#clf.predict_proba(X_train_transformed)#data['train_y_prob']


train_y_prob = model.predict(data = train_pool, prediction_type='Probability')#[:,1]#clf.predict_proba(X_train_transformed)#data['train_y_prob']
train_y_prob_cali = VAC.predict_proba(p_cal=valid_y_prob, y_cal=valid_y_true, p_test=train_y_prob)
train_y_prob = train_y_prob_cali


results_names = ['accuracy', 'auroc', 'auprc', 'f1_score', 'prec', 'rec', 'spec', 'aprec', 'br_score', 'tn', 'fp', 'fn', 'tp']

results={'accuracy': [], 
         'auroc': [], 
         'auprc': [], 'f1_score': [], 'prec': [], 'rec': [], 'spec': [], 'aprec': [], 'br_score': [], 'tn': [], 'fp': [], 'fn': [], 'tp': []
}


#sys.path.append('generate_reports')
text_labels = ['Term', 'Preterm']

# Use the train_y to generate the optimal cutoff point
_,_, _, _, roc_j_thr_dict = get_optimal_cutoff(train_y_true,train_y_prob, text_labels)


testing_samples_num = []
from tqdm import tqdm
with tqdm(total=1000) as pbar:   
    for num_bootstraps in range(1000):
        rnumber = [random.randint(0,y_test.shape[0]) for x in range(y_test.shape[0])]
        #print(np.unique(rnumber).shape)
        test_y_true = np.squeeze(np.eye(2)[y_test[rnumber].astype(int).reshape(-1)])
        test_y_prob_bootstrap = VAC.predict_proba(p_cal=valid_y_prob, y_cal=valid_y_true, p_test=test_y_prob[rnumber])

        # Use the test_y to generate report
        class_df_dict, creport_dict = get_pred_report(test_y_true,test_y_prob_bootstrap,text_labels,roc_j_thr_dict, verbose=False) # verbose = False
        #print(creport_dict['Preterm'])

        for metric in results_names:
            results[metric].append(creport_dict['Preterm'][metric])

        pbar.update(1)

from scipy.stats import sem
results_mean_std={}
for metric in results_names:
    results_mean_std[metric]=(np.mean(results[metric]), sem(results[metric]))
    print(metric, np.mean(results[metric]), sem(results[metric]))
'''
save_obj(results_mean_std, 'catboost_results_mean_std')
save_obj(results, 'catboost_raw_results')
save_obj(testing_samples_num, 'catboost_bootstrapping_test_sample_distribution')
'''


# In[ ]:





---
title: Digging through some Folding@Home data 
date: 2020-05-06
permalink: /posts/2020/05/covid_moonshot/
tags:
    - personal
    - datascience
    - molecularmodeling
---
# Learning cheminformatics from some Folding@Home data


![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_75_0.png)

Top 10 (based on Hybrid2 docking score) small molecules


2020-05-06 - 2020-05-11

I have no formal training in cheminformatics, so I am going to be stumbling and learning as I wade through this dataset.
I welcome any learning lessons from experts.

This will be an ongoing foray

Source: https://github.com/FoldingAtHome/covid-moonshot

## Introduction
Folding@Home is a distributed computing project - allowing molecular simulations to be run in parallel across thousands of different computers with minimal communication.
This, combined with other molecular modeling methods, has yielded a lot of open data for others to examine.
In particular, I'm interested in the docking screens and compounds targeted by the F@H and postera collaborations


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
pd.options.display.max_columns = 999

moonshot_df = pd.read_csv('moonshot-submissions/covid_submissions_all_info.csv')
```


```python
moonshot_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>CID</th>
      <th>creator</th>
      <th>fragments</th>
      <th>link</th>
      <th>real_space</th>
      <th>SCR</th>
      <th>BB</th>
      <th>extended_real_space</th>
      <th>in_molport_or_mcule</th>
      <th>in_ultimate_mcule</th>
      <th>in_emolecules</th>
      <th>covalent_frag</th>
      <th>covalent_warhead</th>
      <th>acrylamide</th>
      <th>acrylamide_adduct</th>
      <th>chloroacetamide</th>
      <th>chloroacetamide_adduct</th>
      <th>vinylsulfonamide</th>
      <th>vinylsulfonamide_adduct</th>
      <th>nitrile</th>
      <th>nitrile_adduct</th>
      <th>MW</th>
      <th>cLogP</th>
      <th>HBD</th>
      <th>HBA</th>
      <th>TPSA</th>
      <th>num_criterion_violations</th>
      <th>BMS</th>
      <th>Dundee</th>
      <th>Glaxo</th>
      <th>Inpharmatica</th>
      <th>LINT</th>
      <th>MLSMR</th>
      <th>PAINS</th>
      <th>SureChEMBL</th>
      <th>PostEra</th>
      <th>ORDERED</th>
      <th>MADE</th>
      <th>ASSAYED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CCN(Cc1cccc(-c2ccncc2)c1)C(=O)Cn1nnc2ccccc21</td>
      <td>AAR-POS-8a4e0f60-1</td>
      <td>Aaron Morris, PostEra</td>
      <td>x0072</td>
      <td>https://covid.postera.ai/covid/submissions/AAR...</td>
      <td>Z1260533612</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>371.444</td>
      <td>3.5420</td>
      <td>0</td>
      <td>5</td>
      <td>63.91</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>O=C(Cn1nnc2ccccc21)NCc1ccc(Oc2cccnc2)c(F)c1</td>
      <td>AAR-POS-8a4e0f60-10</td>
      <td>Aaron Morris, PostEra</td>
      <td>x0072</td>
      <td>https://covid.postera.ai/covid/submissions/AAR...</td>
      <td>Z826180044</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>s_22____1723102____13206668</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>377.379</td>
      <td>3.0741</td>
      <td>1</td>
      <td>6</td>
      <td>81.93</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CN(Cc1nnc2ccccn12)C(=O)N(Cc1cccs1)c1ccc(Br)cc1</td>
      <td>AAR-POS-8a4e0f60-11</td>
      <td>Aaron Morris, PostEra</td>
      <td>x0072</td>
      <td>https://covid.postera.ai/covid/submissions/AAR...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>456.369</td>
      <td>4.8119</td>
      <td>0</td>
      <td>5</td>
      <td>53.74</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Filter9_metal</td>
      <td>aryl bromide</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CCN(Cc1cccc(-c2ccncc2)c1)C(=O)Cc1noc2ccccc12</td>
      <td>AAR-POS-8a4e0f60-2</td>
      <td>Aaron Morris, PostEra</td>
      <td>x0072</td>
      <td>https://covid.postera.ai/covid/submissions/AAR...</td>
      <td>Z1260535907</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>371.440</td>
      <td>4.4810</td>
      <td>0</td>
      <td>4</td>
      <td>59.23</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>O=C(NCc1noc2ccccc12)N(Cc1cccs1)c1ccc(F)cc1</td>
      <td>AAR-POS-8a4e0f60-3</td>
      <td>Aaron Morris, PostEra</td>
      <td>x0072</td>
      <td>https://covid.postera.ai/covid/submissions/AAR...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>s_272164____9388766____17338746</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>381.432</td>
      <td>4.9448</td>
      <td>1</td>
      <td>4</td>
      <td>58.37</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The moonshot data has a lot of logging/metadata information, some one-hot-encoding information about functional groups, and some additional columns about Glaxo, Dundee, BMS, Lint, PAINS, SureChEMBL - I'm not sure what those additional coluns mean, but the values are binary values, possibly the results of some other test or availability in another databases.

I'm going to focus on the molecular properties: MW, cLogP, HBD, HBA, TPSA


* MW: Molecular Weight
* cLogP: The logarithm of the partition coefficient (ratio of concentrations in octanol vs water, 
$\log{\frac{c_{octanol}}{c_{water}}}$)
* HBD: Hydrogen bond donors
* HBA: Hydrogen bond acceptors
* TPSA: Topological polar surface area

Some of the correlations make some chemical sense - heavier molecules have more heavy atoms (O, N, F, etc.), but these heavier atoms are also the hydrogen bond acceptors.
By that logic, more heavy atoms also coincides with more electronegative atoms, increasing your TPSA.
It's a little convoluted because TPSA looks at the surface, not necessarily the volume of the compound; geometry/shape will influence TPSA.
There don't appear to be any strong correlations with cLogP.
Partition coefficients are a complex function of polarity, size/sterics, and shape - a 1:1 correlation with a singular, other variable will be hard to pinpoint

This csv file doesn't have much other numerical data, but maybe some of those true/false, pass/fail data might be relevant...but I definitely need more context here



```python
fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=100)
cols = ['MW', 'cLogP', 'HBD', 'HBA', 'TPSA']
ax.matshow(moonshot_df[cols].corr(), cmap='RdBu')

ax.set_xticks([i for i,_ in enumerate(cols)])
ax.set_xticklabels(cols)

ax.set_yticks([i for i,_ in enumerate(cols)])
ax.set_yticklabels(cols)

for i, (rowname, row) in enumerate(moonshot_df[cols].corr().iterrows()):
    for j, (key, val) in enumerate(row.iteritems()):
        ax.annotate(f"{val:0.2f}", xy=(i,j), xytext=(-10, -5), textcoords="offset points")

```


![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_4_0.png)


## Some docking results

Okay here's a couple other CSVs I found, these include some docking scores

* Repurposing scores: "The Drug Repurposing Hub is a curated and annotated collection of FDA-approved drugs, clinical trial drugs, and pre-clinical tool compounds with a companion information resource" [source here](https://clue.io/repurposing), so a public dataset of some drugs
* Redock scores: "This directory contains experiments in redocking all screened fragments into the entire ensemble of X-ray structures." Taking fragments and re-docking them


```python
repurposing_df = pd.read_csv('repurposing-screen/drugset-docked.csv')
redock_df = pd.read_csv('redock-fragments/all-screened-fragments-docked.csv')
```

SMILES strings, names, docking scores


```python
repurposing_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-_dock</th>
      <th>site</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C[C@@H](c1ccc-2c(c1)Cc3c2cccc3)C(=O)[O-]</td>
      <td>CHEMBL2104122</td>
      <td>-11.519580</td>
      <td>x0749</td>
      <td>0.509349</td>
      <td>active-covalent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@]2(C#C)O)CCC4...</td>
      <td>CHEMBL1387</td>
      <td>-10.580162</td>
      <td>x0749</td>
      <td>2.706928</td>
      <td>active-covalent</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CC(C)(C)c1cc(cc(c1O)C(C)(C)C)/C=C\2/C(=O)NC(=[...</td>
      <td>CHEMBL275835</td>
      <td>-10.557229</td>
      <td>x0107</td>
      <td>1.801830</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C[C@]12CC[C@@H]3[C@H]4CCCCC4=CC[C@H]3[C@@H]1CC...</td>
      <td>CHEMBL2104104</td>
      <td>-10.480992</td>
      <td>x0749</td>
      <td>3.791700</td>
      <td>active-covalent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CC(=O)[C@]1(CC[C@@H]2[C@@]1(CCC3=C4CCC(=O)C=C4...</td>
      <td>CHEMBL2104231</td>
      <td>-10.430775</td>
      <td>x0749</td>
      <td>4.230903</td>
      <td>active-covalent</td>
    </tr>
  </tbody>
</table>
</div>



[Hybrid2](https://docs.eyesopen.com/toolkits/java/dockingtk/docking.html) looks like a docking method provided via OpenEye. 
Mpro likely refers to COVID-19 main protease.
I'm not entirely sure what the receptor for "Hybrid2" is, but there seem to be multiple "sites" or "fragments" for docking.
There are lots of different fragments, but very few sites.
For each site-fragment combination, multiple small molecules may have been tested.


```python
repurposing_df['docked_fragment'].value_counts()
```




    x0195    114
    x0749     69
    x0678     58
    x0397     45
    x0104     24
    x0161     21
    x1077     19
    x0072     14
    x0874     13
    x0354     13
    x0689     10
    x1382      7
    x0708      4
    x0434      4
    x1093      3
    x1392      2
    x0395      2
    x1402      2
    x0831      2
    x0107      2
    x1385      2
    x1418      2
    x0387      2
    x0830      2
    x1478      1
    x0786      1
    x1187      1
    x0692      1
    x0967      1
    x0426      1
    x0305      1
    x0946      1
    x1386      1
    x0759      1
    Name: docked_fragment, dtype: int64




```python
repurposing_df['site'].value_counts()
```




    active-noncovalent    338
    active-covalent       107
    dimer-interface         1
    Name: site, dtype: int64




```python
repurposing_df.groupby(["docked_fragment", "site"]).count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>SMILES</th>
      <th>TITLE</th>
      <th>Hybrid2</th>
      <th>Mpro-_dock</th>
    </tr>
    <tr>
      <th>docked_fragment</th>
      <th>site</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x0072</th>
      <th>active-noncovalent</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>x0104</th>
      <th>active-noncovalent</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>x0107</th>
      <th>active-noncovalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x0161</th>
      <th>active-noncovalent</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>x0195</th>
      <th>active-noncovalent</th>
      <td>114</td>
      <td>114</td>
      <td>114</td>
      <td>114</td>
    </tr>
    <tr>
      <th>x0305</th>
      <th>active-noncovalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x0354</th>
      <th>active-noncovalent</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>x0387</th>
      <th>active-noncovalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x0395</th>
      <th>active-noncovalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x0397</th>
      <th>active-noncovalent</th>
      <td>45</td>
      <td>45</td>
      <td>45</td>
      <td>45</td>
    </tr>
    <tr>
      <th>x0426</th>
      <th>active-noncovalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x0434</th>
      <th>active-noncovalent</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>x0678</th>
      <th>active-noncovalent</th>
      <td>58</td>
      <td>58</td>
      <td>58</td>
      <td>58</td>
    </tr>
    <tr>
      <th>x0689</th>
      <th>active-covalent</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>x0692</th>
      <th>active-covalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x0708</th>
      <th>active-covalent</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>x0749</th>
      <th>active-covalent</th>
      <td>69</td>
      <td>69</td>
      <td>69</td>
      <td>69</td>
    </tr>
    <tr>
      <th>x0759</th>
      <th>active-covalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x0786</th>
      <th>active-covalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x0830</th>
      <th>active-covalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x0831</th>
      <th>active-covalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x0874</th>
      <th>active-noncovalent</th>
      <td>13</td>
      <td>13</td>
      <td>13</td>
      <td>13</td>
    </tr>
    <tr>
      <th>x0946</th>
      <th>active-noncovalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x0967</th>
      <th>active-noncovalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x1077</th>
      <th>active-noncovalent</th>
      <td>19</td>
      <td>19</td>
      <td>19</td>
      <td>19</td>
    </tr>
    <tr>
      <th>x1093</th>
      <th>active-noncovalent</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>x1187</th>
      <th>dimer-interface</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x1382</th>
      <th>active-covalent</th>
      <td>7</td>
      <td>7</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>x1385</th>
      <th>active-covalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x1386</th>
      <th>active-covalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>x1392</th>
      <th>active-covalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x1402</th>
      <th>active-covalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x1418</th>
      <th>active-covalent</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>x1478</th>
      <th>active-covalent</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Some molecules show up multiple times - why?
Upon further investigation, this is mainly due to the molecule's presence in multiple databases


```python
repurposing_df.groupby(['SMILES']).count().sort_values("TITLE")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TITLE</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-_dock</th>
      <th>site</th>
    </tr>
    <tr>
      <th>SMILES</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>B(CCCC)(O)O</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CCCc1ccccc1N</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CCCc1cc(=O)[nH]c(=S)[nH]1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CCC[N@@H+]1CCO[C@H]2[C@H]1CCc3c2cc(cc3)O</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CCC[N@@H+]1CCC[C@H]2[C@H]1Cc3c[nH]nc3C2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@]2(C#C)O)CCC4=CC(=O)CC[C@H]34</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>C[C@]12CC[C@H]3[C@H]([C@@H]1CCC2=O)CC(=C)C4=CC(=O)C=C[C@]34C</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CC(C)C[C@@H](C1(CCC1)c2ccc(cc2)Cl)[NH+](C)C</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CC[C@](/C=C/Cl)(C#C)O</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CC[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@]2(C#C)O)CCC4=CC(=O)CC[C@H]34</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>432 rows × 5 columns</p>
</div>




```python
repurposing_df[repurposing_df['SMILES']=="CC[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@]2(C#C)O)CCC4=CC(=O)CC[C@H]34"]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-_dock</th>
      <th>site</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>82</th>
      <td>CC[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@]2(C#C)O)CC...</td>
      <td>CHEMBL2107797</td>
      <td>-9.002963</td>
      <td>x0749</td>
      <td>2.616094</td>
      <td>active-covalent</td>
    </tr>
    <tr>
      <th>105</th>
      <td>CC[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@]2(C#C)O)CC...</td>
      <td>EDRUG178</td>
      <td>-8.705896</td>
      <td>x0104</td>
      <td>2.248707</td>
      <td>active-noncovalent</td>
    </tr>
  </tbody>
</table>
</div>



There doesn't seem to be a very good correlation between the two docking scores - if these are docking scores to different receptors, that would help explain things.
It's worth noting that we're not seeing if the two numbers agree for each molecule, but if the trends persist (both scores go up for this molecule, but go down for this other molecule).
The weak correlation suggests the trends do not persist between the two docking measures


```python
repurposing_df[['Hybrid2', 'Mpro-_dock']].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hybrid2</th>
      <th>Mpro-_dock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hybrid2</th>
      <td>1.000000</td>
      <td>0.581966</td>
    </tr>
    <tr>
      <th>Mpro-_dock</th>
      <td>0.581966</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Redocking dataframe: SMILES, names, data collection information, docking scores


```python
redock_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE</th>
      <th>fragments</th>
      <th>CompoundCode</th>
      <th>Unnamed: 4</th>
      <th>covalent_warhead</th>
      <th>MountingResult</th>
      <th>DataCollectionOutcome</th>
      <th>DataProcessingResolutionHigh</th>
      <th>RefinementOutcome</th>
      <th>Deposition_PDB_ID</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-x0500_dock</th>
      <th>site</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c1ccc(c(c1)NCc2ccn[nH]2)F</td>
      <td>x0500</td>
      <td>x0500</td>
      <td>Z1545196403</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>2.19</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-11.881923</td>
      <td>x0678</td>
      <td>-2.501554</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cc1ccccc1OCC(=O)Nc2ncccn2</td>
      <td>x0415</td>
      <td>x0415</td>
      <td>Z53834613</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.62</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-11.622278</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cc1csc(n1)CNC(=O)c2ccn[nH]2</td>
      <td>x0356</td>
      <td>x0356</td>
      <td>Z466628048</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>3.25</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-11.435024</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cc1csc(n1)CNC(=O)c2ccn[nH]2</td>
      <td>x1113</td>
      <td>x1113</td>
      <td>Z466628048</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.57</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-11.435024</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c1cc(cnc1)NC(=O)CC2CCCCC2</td>
      <td>x0678</td>
      <td>x0678</td>
      <td>Z31792168</td>
      <td>NaN</td>
      <td>False</td>
      <td>Mounted_Clear</td>
      <td>success</td>
      <td>1.83</td>
      <td>6 - Deposited</td>
      <td>5R84</td>
      <td>-11.355046</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
  </tbody>
</table>
</div>



There don't seem to be many Mpro docking scores in this dataset (only one molecule has a non-null Mpro docking score)


```python
redock_df[redock_df['Mpro-x0500_dock'].isnull()].count()
```




    SMILES                          1452
    TITLE                           1452
    fragments                       1452
    CompoundCode                    1452
    Unnamed: 4                         0
    covalent_warhead                1452
    MountingResult                  1452
    DataCollectionOutcome           1452
    DataProcessingResolutionHigh    1357
    RefinementOutcome               1306
    Deposition_PDB_ID                 78
    Hybrid2                         1452
    docked_fragment                 1452
    Mpro-x0500_dock                    0
    site                            1452
    dtype: int64




```python
redock_df[~redock_df['Mpro-x0500_dock'].isnull()].count()
```




    SMILES                          1
    TITLE                           1
    fragments                       1
    CompoundCode                    1
    Unnamed: 4                      0
    covalent_warhead                1
    MountingResult                  1
    DataCollectionOutcome           1
    DataProcessingResolutionHigh    1
    RefinementOutcome               1
    Deposition_PDB_ID               0
    Hybrid2                         1
    docked_fragment                 1
    Mpro-x0500_dock                 1
    site                            1
    dtype: int64



Are there overlaps in the molecules in each of these datasets?


```python
repurpose_redock = repurposing_df.merge(redock_df, on='SMILES', how='inner',suffixes=("_L", "_R"))
```


```python
moonshot_redock = moonshot_df.merge(redock_df, on='SMILES', how='inner',suffixes=("_L", "_R"))
```


```python
repurpose_redock
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE_L</th>
      <th>Hybrid2_L</th>
      <th>docked_fragment_L</th>
      <th>Mpro-_dock</th>
      <th>site_L</th>
      <th>TITLE_R</th>
      <th>fragments</th>
      <th>CompoundCode</th>
      <th>Unnamed: 4</th>
      <th>covalent_warhead</th>
      <th>MountingResult</th>
      <th>DataCollectionOutcome</th>
      <th>DataProcessingResolutionHigh</th>
      <th>RefinementOutcome</th>
      <th>Deposition_PDB_ID</th>
      <th>Hybrid2_R</th>
      <th>docked_fragment_R</th>
      <th>Mpro-x0500_dock</th>
      <th>site_R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cc1cc(=O)n([nH]1)c2ccccc2</td>
      <td>CHEMBL290916</td>
      <td>-7.889587</td>
      <td>x0195</td>
      <td>-2.068452</td>
      <td>active-noncovalent</td>
      <td>x0297</td>
      <td>x0297</td>
      <td>Z50145861</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.98</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-7.889587</td>
      <td>x0195</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC(C)Nc1ncccn1</td>
      <td>CHEMBL1740513</td>
      <td>-7.178702</td>
      <td>x0072</td>
      <td>-1.248482</td>
      <td>active-noncovalent</td>
      <td>x0583</td>
      <td>x0583</td>
      <td>Z31190928</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>3.08</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-7.293537</td>
      <td>x1093</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CC(C)Nc1ncccn1</td>
      <td>CHEMBL1740513</td>
      <td>-7.178702</td>
      <td>x0072</td>
      <td>-1.248482</td>
      <td>active-noncovalent</td>
      <td>x1102</td>
      <td>x1102</td>
      <td>Z31190928</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.46</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-7.293537</td>
      <td>x1093</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C[C@H](C(=O)[O-])O</td>
      <td>CHEMBL1200559</td>
      <td>-5.675188</td>
      <td>x0397</td>
      <td>-0.179049</td>
      <td>active-noncovalent</td>
      <td>x1035</td>
      <td>x1035</td>
      <td>Z1741982441</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>Failed - no diffraction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-6.505556</td>
      <td>x0397</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CC(=O)C(=O)[O-]</td>
      <td>DB00119</td>
      <td>-5.448891</td>
      <td>x0689</td>
      <td>-0.494791</td>
      <td>active-covalent</td>
      <td>x1037</td>
      <td>x1037</td>
      <td>Z1741977082</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>Failed - no diffraction</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-5.448891</td>
      <td>x0689</td>
      <td>NaN</td>
      <td>active-covalent</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CCC(=O)[O-]</td>
      <td>CHEMBL14021</td>
      <td>-5.374838</td>
      <td>x0397</td>
      <td>-0.555688</td>
      <td>active-noncovalent</td>
      <td>x1029</td>
      <td>x1029</td>
      <td>Z955123616</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.73</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-5.135675</td>
      <td>x0689</td>
      <td>NaN</td>
      <td>active-covalent</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C1CNCC[NH2+]1</td>
      <td>CHEMBL1412</td>
      <td>-5.079155</td>
      <td>x0354</td>
      <td>1.716032</td>
      <td>active-noncovalent</td>
      <td>x0996</td>
      <td>x0996</td>
      <td>Z1245537944</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.96</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-4.675085</td>
      <td>x0354</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
  </tbody>
</table>
</div>



We joined on SMILES string, and now we can compare the docking scores between the repurposing and redocking datasets.

Some `Hybrid2` scores look quantitatively similar, but for those that don't, the ranking is still there.
Looking at the COVID-19 main protease (Mpro I believe?), the docking scores don't follow similar rankings - docking scores aren't transferable to different receptors (this might be a fairly obvious observation)


```python
repurpose_redock[['SMILES', "TITLE_L", "TITLE_R", "Hybrid2_L", "Hybrid2_R", 'Mpro-_dock', 'Mpro-x0500_dock']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE_L</th>
      <th>TITLE_R</th>
      <th>Hybrid2_L</th>
      <th>Hybrid2_R</th>
      <th>Mpro-_dock</th>
      <th>Mpro-x0500_dock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cc1cc(=O)n([nH]1)c2ccccc2</td>
      <td>CHEMBL290916</td>
      <td>x0297</td>
      <td>-7.889587</td>
      <td>-7.889587</td>
      <td>-2.068452</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC(C)Nc1ncccn1</td>
      <td>CHEMBL1740513</td>
      <td>x0583</td>
      <td>-7.178702</td>
      <td>-7.293537</td>
      <td>-1.248482</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CC(C)Nc1ncccn1</td>
      <td>CHEMBL1740513</td>
      <td>x1102</td>
      <td>-7.178702</td>
      <td>-7.293537</td>
      <td>-1.248482</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>C[C@H](C(=O)[O-])O</td>
      <td>CHEMBL1200559</td>
      <td>x1035</td>
      <td>-5.675188</td>
      <td>-6.505556</td>
      <td>-0.179049</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CC(=O)C(=O)[O-]</td>
      <td>DB00119</td>
      <td>x1037</td>
      <td>-5.448891</td>
      <td>-5.448891</td>
      <td>-0.494791</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CCC(=O)[O-]</td>
      <td>CHEMBL14021</td>
      <td>x1029</td>
      <td>-5.374838</td>
      <td>-5.135675</td>
      <td>-0.555688</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>C1CNCC[NH2+]1</td>
      <td>CHEMBL1412</td>
      <td>x0996</td>
      <td>-5.079155</td>
      <td>-4.675085</td>
      <td>1.716032</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Joining the moonshot submission and redocking datasets does not yield too many overlapping molecules


```python
moonshot_redock
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>CID</th>
      <th>creator</th>
      <th>fragments_L</th>
      <th>link</th>
      <th>real_space</th>
      <th>SCR</th>
      <th>BB</th>
      <th>extended_real_space</th>
      <th>in_molport_or_mcule</th>
      <th>in_ultimate_mcule</th>
      <th>in_emolecules</th>
      <th>covalent_frag</th>
      <th>covalent_warhead_L</th>
      <th>acrylamide</th>
      <th>acrylamide_adduct</th>
      <th>chloroacetamide</th>
      <th>chloroacetamide_adduct</th>
      <th>vinylsulfonamide</th>
      <th>vinylsulfonamide_adduct</th>
      <th>nitrile</th>
      <th>nitrile_adduct</th>
      <th>MW</th>
      <th>cLogP</th>
      <th>HBD</th>
      <th>HBA</th>
      <th>TPSA</th>
      <th>num_criterion_violations</th>
      <th>BMS</th>
      <th>Dundee</th>
      <th>Glaxo</th>
      <th>Inpharmatica</th>
      <th>LINT</th>
      <th>MLSMR</th>
      <th>PAINS</th>
      <th>SureChEMBL</th>
      <th>PostEra</th>
      <th>ORDERED</th>
      <th>MADE</th>
      <th>ASSAYED</th>
      <th>TITLE</th>
      <th>fragments_R</th>
      <th>CompoundCode</th>
      <th>Unnamed: 4</th>
      <th>covalent_warhead_R</th>
      <th>MountingResult</th>
      <th>DataCollectionOutcome</th>
      <th>DataProcessingResolutionHigh</th>
      <th>RefinementOutcome</th>
      <th>Deposition_PDB_ID</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-x0500_dock</th>
      <th>site</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CC(C)Nc1cccnc1</td>
      <td>MAK-UNK-2c1752f0-4</td>
      <td>Maksym Voznyy</td>
      <td>x1093</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>Z2574930241</td>
      <td>EN300-56005</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>136.198</td>
      <td>1.9019</td>
      <td>1</td>
      <td>2</td>
      <td>24.92</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>x1098</td>
      <td>x1098</td>
      <td>Z1259341037</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.66</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-7.474369</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CC(C)Nc1cccnc1</td>
      <td>MAK-UNK-2c1752f0-4</td>
      <td>Maksym Voznyy</td>
      <td>x1093</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>Z2574930241</td>
      <td>EN300-56005</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>136.198</td>
      <td>1.9019</td>
      <td>1</td>
      <td>2</td>
      <td>24.92</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>x0572</td>
      <td>x0572</td>
      <td>Z1259341037</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>2.98</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-7.474369</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CCS(=O)(=O)Nc1ccccc1F</td>
      <td>MAK-UNK-2c1752f0-5</td>
      <td>Maksym Voznyy</td>
      <td>x1093</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>Z53825177</td>
      <td>EN300-116204</td>
      <td>FALSE</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>203.238</td>
      <td>1.5873</td>
      <td>1</td>
      <td>2</td>
      <td>46.17</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Hetero_hetero</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>x0247</td>
      <td>x0247</td>
      <td>Z53825177</td>
      <td>NaN</td>
      <td>False</td>
      <td>OK: No comment:No comment</td>
      <td>success</td>
      <td>1.83</td>
      <td>7 - Analysed &amp; Rejected</td>
      <td>NaN</td>
      <td>-7.413380</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
    </tr>
  </tbody>
</table>
</div>



## Comparing other databases

CHEMBL, DrugBank, and "EDrug"(?) look to be the 3 prefixes in the "TITLE" column


```python
from chembl_webresource_client.new_client import new_client
molecule = new_client.molecule
res = molecule.search('CHEMBL1387')
```


```python
res_df = pd.DataFrame.from_dict(res)
```


```python
res_df.columns
```




    Index(['atc_classifications', 'availability_type', 'biotherapeutic',
           'black_box_warning', 'chebi_par_id', 'chirality', 'cross_references',
           'dosed_ingredient', 'first_approval', 'first_in_class', 'helm_notation',
           'indication_class', 'inorganic_flag', 'max_phase', 'molecule_chembl_id',
           'molecule_hierarchy', 'molecule_properties', 'molecule_structures',
           'molecule_synonyms', 'molecule_type', 'natural_product', 'oral',
           'parenteral', 'polymer_flag', 'pref_name', 'prodrug', 'score',
           'structure_type', 'therapeutic_flag', 'topical', 'usan_stem',
           'usan_stem_definition', 'usan_substem', 'usan_year', 'withdrawn_class',
           'withdrawn_country', 'withdrawn_flag', 'withdrawn_reason',
           'withdrawn_year'],
          dtype='object')




```python
res_df[['chirality', 'molecule_properties', 'molecule_structures', 'score']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>chirality</th>
      <th>molecule_properties</th>
      <th>molecule_structures</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>{'alogp': '3.64', 'aromatic_rings': 0, 'cx_log...</td>
      <td>{'canonical_smiles': 'C#C[C@]1(O)CC[C@H]2[C@@H...</td>
      <td>17.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
res_df[['molecule_properties']].values[0]
```




    array([{'alogp': '3.64', 'aromatic_rings': 0, 'cx_logd': '2.81', 'cx_logp': '2.81', 'cx_most_apka': None, 'cx_most_bpka': None, 'full_molformula': 'C20H26O2', 'full_mwt': '298.43', 'hba': 2, 'hba_lipinski': 2, 'hbd': 1, 'hbd_lipinski': 1, 'heavy_atoms': 22, 'molecular_species': None, 'mw_freebase': '298.43', 'mw_monoisotopic': '298.1933', 'num_lipinski_ro5_violations': 0, 'num_ro5_violations': 0, 'psa': '37.30', 'qed_weighted': '0.55', 'ro3_pass': 'N', 'rtb': 0}],
          dtype=object)




```python
res_df['molecule_properties'].apply(pd.Series)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alogp</th>
      <th>aromatic_rings</th>
      <th>cx_logd</th>
      <th>cx_logp</th>
      <th>cx_most_apka</th>
      <th>cx_most_bpka</th>
      <th>full_molformula</th>
      <th>full_mwt</th>
      <th>hba</th>
      <th>hba_lipinski</th>
      <th>hbd</th>
      <th>hbd_lipinski</th>
      <th>heavy_atoms</th>
      <th>molecular_species</th>
      <th>mw_freebase</th>
      <th>mw_monoisotopic</th>
      <th>num_lipinski_ro5_violations</th>
      <th>num_ro5_violations</th>
      <th>psa</th>
      <th>qed_weighted</th>
      <th>ro3_pass</th>
      <th>rtb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.64</td>
      <td>0</td>
      <td>2.81</td>
      <td>2.81</td>
      <td>None</td>
      <td>None</td>
      <td>C20H26O2</td>
      <td>298.43</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>None</td>
      <td>298.43</td>
      <td>298.1933</td>
      <td>0</td>
      <td>0</td>
      <td>37.30</td>
      <td>0.55</td>
      <td>N</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_results = [molecule.search(a) for a in repurposing_df['TITLE']]
```

Here's a big Python function tangent.

For each chembl molecule, we've searched for it within the chembl, returning us a list (of length 1) containing a dictionary of properties. 

All molecules have been compiled into a list, so we have a list of lists of dicionatires.

For sanity, we can use a Python `filter` to only retain the non-None results.

We can chain that with a Python `map` function to parse the first item from each molecule's list. 
Recall, each molecule was a list with just one element, a dictionary.
We can boil this down to only returning the dictionary (eliminating the list wrapper).

For validation, I've called `next` to look at the results


```python
filtered = map(lambda x: x[0], filter(lambda x: x is not None, all_results))
```


```python
next(filtered)
```




    {'atc_classifications': [],
     'availability_type': -1,
     'biotherapeutic': None,
     'black_box_warning': 0,
     'chebi_par_id': None,
     'chirality': 0,
     'cross_references': [],
     'dosed_ingredient': False,
     'first_approval': None,
     'first_in_class': 0,
     'helm_notation': None,
     'indication_class': 'Anti-Inflammatory',
     'inorganic_flag': 0,
     'max_phase': 0,
     'molecule_chembl_id': 'CHEMBL2104122',
     'molecule_hierarchy': {'molecule_chembl_id': 'CHEMBL2104122',
      'parent_chembl_id': 'CHEMBL2104122'},
     'molecule_properties': {'alogp': '3.45',
      'aromatic_rings': 2,
      'cx_logd': '1.26',
      'cx_logp': '3.92',
      'cx_most_apka': '4.68',
      'cx_most_bpka': None,
      'full_molformula': 'C16H14O2',
      'full_mwt': '238.29',
      'hba': 1,
      'hba_lipinski': 2,
      'hbd': 1,
      'hbd_lipinski': 1,
      'heavy_atoms': 18,
      'molecular_species': 'ACID',
      'mw_freebase': '238.29',
      'mw_monoisotopic': '238.0994',
      'num_lipinski_ro5_violations': 0,
      'num_ro5_violations': 0,
      'psa': '37.30',
      'qed_weighted': '0.74',
      'ro3_pass': 'N',
      'rtb': 2},
     'molecule_structures': {'canonical_smiles': 'CC(C(=O)O)c1ccc2c(c1)Cc1ccccc1-2',
      'molfile': '\n     RDKit          2D\n\n 18 20  0  0  0  0  0  0  0  0999 V2000\n   -0.5375    0.0250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -0.5375    1.1083    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.4458    1.1083    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -2.4458    0.0250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.3625    0.0250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -1.4875   -0.5125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.4125   -0.5125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.3292    0.0250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    0.4125    1.6500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.3417   -0.5292    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    1.3625    1.1083    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.3500    1.1958    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    4.2167   -0.6292    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   -3.3958    1.6500    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -3.3958   -0.5125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.3417   -1.6417    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.3458    1.1083    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   -4.3458    0.0250    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n  2  1  2  0\n  3  2  1  0\n  4  6  1  0\n  5  7  2  0\n  6  1  1  0\n  7  1  1  0\n  8 10  1  0\n  9  2  1  0\n 10  5  1  0\n 11  5  1  0\n 12  8  2  0\n 13  8  1  0\n 14  3  1  0\n 15  4  1  0\n 16 10  1  0\n 17 14  2  0\n 18 15  2  0\n  3  4  2  0\n  9 11  2  0\n 17 18  1  0\nM  END\n\n> <chembl_id>\nCHEMBL2104122\n\n> <chembl_pref_name>\nCICLOPROFEN\n\n',
      'standard_inchi': 'InChI=1S/C16H14O2/c1-10(16(17)18)11-6-7-15-13(8-11)9-12-4-2-3-5-14(12)15/h2-8,10H,9H2,1H3,(H,17,18)',
      'standard_inchi_key': 'LRXFKKPEBXIPMW-UHFFFAOYSA-N'},
     'molecule_synonyms': [{'molecule_synonym': 'Cicloprofen',
       'syn_type': 'BAN',
       'synonyms': 'CICLOPROFEN'},
      {'molecule_synonym': 'Cicloprofen',
       'syn_type': 'INN',
       'synonyms': 'CICLOPROFEN'},
      {'molecule_synonym': 'Cicloprofen',
       'syn_type': 'USAN',
       'synonyms': 'CICLOPROFEN'},
      {'molecule_synonym': 'SQ-20824',
       'syn_type': 'RESEARCH_CODE',
       'synonyms': 'SQ 20824'}],
     'molecule_type': 'Small molecule',
     'natural_product': 0,
     'oral': False,
     'parenteral': False,
     'polymer_flag': False,
     'pref_name': 'CICLOPROFEN',
     'prodrug': 0,
     'score': 16.0,
     'structure_type': 'MOL',
     'therapeutic_flag': False,
     'topical': False,
     'usan_stem': '-profen',
     'usan_stem_definition': 'anti-inflammatory/analgesic agents (ibuprofen type)',
     'usan_substem': '-profen',
     'usan_year': 1974,
     'withdrawn_class': None,
     'withdrawn_country': None,
     'withdrawn_flag': False,
     'withdrawn_reason': None,
     'withdrawn_year': None}



For now, I'm only really interested in the `molecule_properties` dictionary


```python
filtered = [a[0]['molecule_properties'] for a in all_results if len(a) > 0]
```


```python
chembl_df = pd.DataFrame(filtered)
chembl_df['TITLE'] = repurposing_df['TITLE']
```

## Molecular properties contained in the chembl database

Here are the definitions I can dig up 
* alogp: (lipophilicity) partition coefficient
* aromatic_rings: number of aromatic rings
* cx_logd: distribution coefficient taking into account ionized and non-ionized forms
* cx_most_apka: acidic pka
* cx_most_bpka: basic pka
* full_mwt: molecular weight (and also free base and monoisotopic masses)
* hba: hydrogen bond acceptors (and hba_lipinski for lipinski definitiosn)
* hbd: hydrogen bond donors (and hbd_lipinski)
* heavy_atoms: number of heavy atoms
* num_lipinski_ro5_violations: how many times this molecule violated [Lipinski's rule of five](https://en.wikipedia.org/wiki/Lipinski%27s_rule_of_five)
* num_ro5_violations: not sure, seems similar to lipinski rule of 5
* psa: protein sequence alignment
* qed_weighted: "quantitative estimate of druglikeness" (ranges between 0 and 1, with 1 being more favorable). This is based on a [quantitatve mean of drugability functions](https://www.nature.com/articles/nchem.1243)
* ro3_pass: [rule of three](https://caz.lab.uic.edu/discovery/Medicinal-Chemistry-2018-Barcelona.pdf)
* rtb: number of rotatable bonds



```python
chembl_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alogp</th>
      <th>aromatic_rings</th>
      <th>cx_logd</th>
      <th>cx_logp</th>
      <th>cx_most_apka</th>
      <th>cx_most_bpka</th>
      <th>full_molformula</th>
      <th>full_mwt</th>
      <th>hba</th>
      <th>hba_lipinski</th>
      <th>hbd</th>
      <th>hbd_lipinski</th>
      <th>heavy_atoms</th>
      <th>molecular_species</th>
      <th>mw_freebase</th>
      <th>mw_monoisotopic</th>
      <th>num_lipinski_ro5_violations</th>
      <th>num_ro5_violations</th>
      <th>psa</th>
      <th>qed_weighted</th>
      <th>ro3_pass</th>
      <th>rtb</th>
      <th>TITLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.45</td>
      <td>2.0</td>
      <td>1.26</td>
      <td>3.92</td>
      <td>4.68</td>
      <td>None</td>
      <td>C16H14O2</td>
      <td>238.29</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>ACID</td>
      <td>238.29</td>
      <td>238.0994</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.30</td>
      <td>0.74</td>
      <td>N</td>
      <td>2.0</td>
      <td>CHEMBL2104122</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.64</td>
      <td>0.0</td>
      <td>2.81</td>
      <td>2.81</td>
      <td>None</td>
      <td>None</td>
      <td>C20H26O2</td>
      <td>298.43</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>None</td>
      <td>298.43</td>
      <td>298.1933</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.30</td>
      <td>0.55</td>
      <td>N</td>
      <td>0.0</td>
      <td>CHEMBL1387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.92</td>
      <td>1.0</td>
      <td>4.25</td>
      <td>4.25</td>
      <td>10.15</td>
      <td>2.86</td>
      <td>C18H24N2O2S</td>
      <td>332.47</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>23.0</td>
      <td>NEUTRAL</td>
      <td>332.47</td>
      <td>332.1558</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>75.68</td>
      <td>0.76</td>
      <td>N</td>
      <td>1.0</td>
      <td>CHEMBL275835</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.31</td>
      <td>0.0</td>
      <td>4.04</td>
      <td>4.04</td>
      <td>None</td>
      <td>None</td>
      <td>C20H28O</td>
      <td>284.44</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>None</td>
      <td>284.44</td>
      <td>284.2140</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20.23</td>
      <td>0.52</td>
      <td>N</td>
      <td>0.0</td>
      <td>CHEMBL2104104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.79</td>
      <td>0.0</td>
      <td>3.96</td>
      <td>3.96</td>
      <td>None</td>
      <td>None</td>
      <td>C21H28O2</td>
      <td>312.45</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>None</td>
      <td>312.45</td>
      <td>312.2089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>34.14</td>
      <td>0.70</td>
      <td>N</td>
      <td>1.0</td>
      <td>CHEMBL2104231</td>
    </tr>
  </tbody>
</table>
</div>




```python
chembl_df.columns
```




    Index(['alogp', 'aromatic_rings', 'cx_logd', 'cx_logp', 'cx_most_apka',
           'cx_most_bpka', 'full_molformula', 'full_mwt', 'hba', 'hba_lipinski',
           'hbd', 'hbd_lipinski', 'heavy_atoms', 'molecular_species',
           'mw_freebase', 'mw_monoisotopic', 'num_lipinski_ro5_violations',
           'num_ro5_violations', 'psa', 'qed_weighted', 'ro3_pass', 'rtb',
           'TITLE'],
          dtype='object')




```python
chembl_df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aromatic_rings</th>
      <th>hba</th>
      <th>hba_lipinski</th>
      <th>hbd</th>
      <th>hbd_lipinski</th>
      <th>heavy_atoms</th>
      <th>num_lipinski_ro5_violations</th>
      <th>num_ro5_violations</th>
      <th>rtb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>aromatic_rings</th>
      <td>1.000000</td>
      <td>0.192569</td>
      <td>0.178507</td>
      <td>0.014928</td>
      <td>0.036106</td>
      <td>0.249022</td>
      <td>0.031094</td>
      <td>0.031094</td>
      <td>0.229124</td>
    </tr>
    <tr>
      <th>hba</th>
      <td>0.192569</td>
      <td>1.000000</td>
      <td>0.868859</td>
      <td>0.084553</td>
      <td>0.054409</td>
      <td>0.451560</td>
      <td>-0.047705</td>
      <td>-0.047705</td>
      <td>-0.023690</td>
    </tr>
    <tr>
      <th>hba_lipinski</th>
      <td>0.178507</td>
      <td>0.868859</td>
      <td>1.000000</td>
      <td>0.348600</td>
      <td>0.294276</td>
      <td>0.295864</td>
      <td>-0.070783</td>
      <td>-0.070783</td>
      <td>0.021812</td>
    </tr>
    <tr>
      <th>hbd</th>
      <td>0.014928</td>
      <td>0.084553</td>
      <td>0.348600</td>
      <td>1.000000</td>
      <td>0.935710</td>
      <td>-0.172866</td>
      <td>-0.060462</td>
      <td>-0.060462</td>
      <td>0.040505</td>
    </tr>
    <tr>
      <th>hbd_lipinski</th>
      <td>0.036106</td>
      <td>0.054409</td>
      <td>0.294276</td>
      <td>0.935710</td>
      <td>1.000000</td>
      <td>-0.211899</td>
      <td>-0.085660</td>
      <td>-0.085660</td>
      <td>0.084225</td>
    </tr>
    <tr>
      <th>heavy_atoms</th>
      <td>0.249022</td>
      <td>0.451560</td>
      <td>0.295864</td>
      <td>-0.172866</td>
      <td>-0.211899</td>
      <td>1.000000</td>
      <td>0.397240</td>
      <td>0.397240</td>
      <td>0.259011</td>
    </tr>
    <tr>
      <th>num_lipinski_ro5_violations</th>
      <td>0.031094</td>
      <td>-0.047705</td>
      <td>-0.070783</td>
      <td>-0.060462</td>
      <td>-0.085660</td>
      <td>0.397240</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.345308</td>
    </tr>
    <tr>
      <th>num_ro5_violations</th>
      <td>0.031094</td>
      <td>-0.047705</td>
      <td>-0.070783</td>
      <td>-0.060462</td>
      <td>-0.085660</td>
      <td>0.397240</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.345308</td>
    </tr>
    <tr>
      <th>rtb</th>
      <td>0.229124</td>
      <td>-0.023690</td>
      <td>0.021812</td>
      <td>0.040505</td>
      <td>0.084225</td>
      <td>0.259011</td>
      <td>0.345308</td>
      <td>0.345308</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



At a glance, no definite linear correlations among this crowd besides pKas, partition coefficients, mwt/hba


```python
corr_df = chembl_df.corr()
cols = chembl_df.columns

fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=100)

ax.imshow(chembl_df.corr(), cmap='RdBu')

ax.set_xticklabels(['']+cols)
ax.tick_params(axis='x', rotation=90)

ax.set_yticklabels(cols)

for i, (rowname, row) in enumerate(corr_df.iterrows()):
    for j, (key, val) in enumerate(row.iteritems()):
        ax.annotate(f"{val:0.2f}", xy=(i,j), xytext=(-10, -5), textcoords="offset points")

```


![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_50_0.png)


Maybe there are higher-order correlations and relationship more appropriate for clustering and decomposition


```python
cols = ['aromatic_rings', 'cx_logp',  'full_mwt', 'hba']
cleaned = (chembl_df[~chembl_df[cols]
                     .isnull()
                     .all(axis='columns', skipna=False)][cols]
           .astype('float')
           .fillna(0, axis='columns'))
```


```python
from sklearn import preprocessing

normalized = preprocessing.scale(cleaned)
```

Appears to be maybe 4 clusters of these compounds examined by the covid-moonshot group


```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne_analysis = TSNE(n_components=2)
output = tsne_analysis.fit_transform(normalized)
fig,ax = plt.subplots(1,1)
ax.scatter(output[:,0], output[:,1])
ax.set_title("Aromatic rings, cx_logp, mwt, hba")
```




    Text(0.5, 1.0, 'Aromatic rings, cx_logp, mwt, hba')




![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_55_1.png)


By taking turns leaving out some features, it looks like leaving out aromatic rings or hydrogen bond acceptors will diminish the cluster distinction.

Aromatic rings are huge and bulky components to small molecules, it makes sense that a chunk of the behavior corresponds to the aromatic rings.
Similarly, hydrogen bond acceptors (heavy molecules) also induce van der Waals and electrostatics influences on small molecules.
Left with only weight and partition coefficient, there's mainly a continous behavior



```python
def clean_df(cols):
    cleaned = (chembl_df[~chembl_df[cols]
                     .isnull()
                     .all(axis='columns', skipna=False)][cols]
           .astype('float')
           .fillna(0, axis='columns'))

    normalized = preprocessing.scale(cleaned)
    
    return normalized

cols = ['cx_logp',  'full_mwt', 'hba']
normalized = clean_df(cols)
tsne_analysis = TSNE(n_components=2)
output = tsne_analysis.fit_transform(normalized)
fig,ax = plt.subplots(3,1, figsize=(8,8))
ax[0].scatter(output[:,0], output[:,1])
ax[0].set_title("cx_logp, mwt, hba")

cols = ['cx_logp',  'full_mwt', 'aromatic_rings']
normalized = clean_df(cols)

tsne_analysis = TSNE(n_components=2)
output = tsne_analysis.fit_transform(normalized)

ax[1].scatter(output[:,0], output[:,1])
ax[1].set_title("aromatic_rings, cx_logp, mwt")

cols = ['cx_logp',  'full_mwt']
normalized = clean_df(cols)

ax[2].scatter(normalized[:,0], normalized[:,1])
ax[2].set_title("cx_logp, mwt")

fig.tight_layout()
```


![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_57_0.png)


DrugBank

I found someone had already [downloaded the database](https://github.com/choderalab/nano-drugbank/blob/master/df_drugbank_smiles.csv).
I may double-over these dataframes, but query the drugbank dataset rather than chembl

## Some docking data

We have some smiles strings, molecular properties, docking scores, and information about the docking fragments


```python
moonshot = pd.read_csv('moonshot-submissions/covid_submissions_all_info-docked-overlap.csv')
```


```python
moonshot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE</th>
      <th>creator</th>
      <th>fragments</th>
      <th>link</th>
      <th>real_space</th>
      <th>SCR</th>
      <th>BB</th>
      <th>extended_real_space</th>
      <th>in_molport_or_mcule</th>
      <th>in_ultimate_mcule</th>
      <th>in_emolecules</th>
      <th>covalent_frag</th>
      <th>covalent_warhead</th>
      <th>acrylamide</th>
      <th>acrylamide_adduct</th>
      <th>chloroacetamide</th>
      <th>chloroacetamide_adduct</th>
      <th>vinylsulfonamide</th>
      <th>vinylsulfonamide_adduct</th>
      <th>nitrile</th>
      <th>nitrile_adduct</th>
      <th>MW</th>
      <th>cLogP</th>
      <th>HBD</th>
      <th>HBA</th>
      <th>TPSA</th>
      <th>num_criterion_violations</th>
      <th>BMS</th>
      <th>Dundee</th>
      <th>Glaxo</th>
      <th>Inpharmatica</th>
      <th>LINT</th>
      <th>MLSMR</th>
      <th>PAINS</th>
      <th>SureChEMBL</th>
      <th>PostEra</th>
      <th>ORDERED</th>
      <th>MADE</th>
      <th>ASSAYED</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-x1418_dock</th>
      <th>site</th>
      <th>number_of_overlapping_fragments</th>
      <th>overlapping_fragments</th>
      <th>overlap_score</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c1ccc(cc1)n2c3cc(c(cc3c(=O)c(c2[O-])c4cccnc4)F)Cl</td>
      <td>MAK-UNK-9e4a73aa-2</td>
      <td>Maksym Voznyy</td>
      <td>x1418</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>366.779</td>
      <td>4.51890</td>
      <td>0</td>
      <td>3</td>
      <td>50.27</td>
      <td>0</td>
      <td>PASS</td>
      <td>beta-keto/anhydride</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Ketone, Dye 11</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-11.881256</td>
      <td>x1418</td>
      <td>1.206534</td>
      <td>active-covalent</td>
      <td>3</td>
      <td>x0434,x0678,x0830</td>
      <td>3.208124</td>
      <td>271.986084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cc1ccncc1n2c(=O)ccc3c2CCCN3CC(=[NH2+])N</td>
      <td>KIM-UNI-60f168f5-7</td>
      <td>Kim Tai Tran, University of Copenhagen</td>
      <td>x0107,x0991</td>
      <td>https://covid.postera.ai/covid/submissions/KIM...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>297.362</td>
      <td>1.22949</td>
      <td>2</td>
      <td>5</td>
      <td>88.00</td>
      <td>0</td>
      <td>PASS</td>
      <td>imine, imine</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>acyclic C=N-H</td>
      <td>Imine 3</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-11.654112</td>
      <td>x0107</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0107,x1412,x1392</td>
      <td>4.753475</td>
      <td>232.815506</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c1ccc(cc1)n2c3cc(c(cc3c(=O)n(c2=O)c4cnccn4)F)Cl</td>
      <td>MAK-UNK-9e4a73aa-14</td>
      <td>Maksym Voznyy</td>
      <td>x1418</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>368.755</td>
      <td>2.72410</td>
      <td>0</td>
      <td>6</td>
      <td>69.78</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-10.460650</td>
      <td>x0678</td>
      <td>2.716276</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0678,x1412,x1392</td>
      <td>5.520980</td>
      <td>266.688721</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cc1ccncc1N(C=C)[C@H]([C@@H](C)[C@@H]2CN=Cc3c2c...</td>
      <td>AUS-WAB-916db9c0-1</td>
      <td>Austin D. Chivington, Wabash College</td>
      <td>x0107,x1077,x1374</td>
      <td>https://covid.postera.ai/covid/submissions/AUS...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>351.450</td>
      <td>3.51932</td>
      <td>1</td>
      <td>5</td>
      <td>57.95</td>
      <td>0</td>
      <td>non_ring_acetal</td>
      <td>het-C-het not in ring</td>
      <td>PASS</td>
      <td>Filter10_Terminal_vinyl</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-9.516450</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0434,x0831,x0678</td>
      <td>3.446572</td>
      <td>284.195312</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c1ccc2c(c1)ncc(n2)/C=C/C(=O)c3cccc(c3)O</td>
      <td>DRV-DNY-ae159ed1-12</td>
      <td>Dr. Vidya Desai, Dnyanprassarak Mandals Colleg...</td>
      <td>x1249</td>
      <td>https://covid.postera.ai/covid/submissions/DRV...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>276.295</td>
      <td>3.23150</td>
      <td>1</td>
      <td>4</td>
      <td>63.08</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Filter44_michael_acceptor2</td>
      <td>PASS</td>
      <td>Ketone, Dye 9, vinyl michael acceptor1</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-9.243208</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0434,x0678,x0830</td>
      <td>2.865147</td>
      <td>220.275421</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4630</th>
      <td>C[C@H]([C@@H](C(=O)N[C@H](Cc1ccccc1)C(=O)N[C@@...</td>
      <td>PAU-UNI-6d15a9f5-4</td>
      <td>paul brear, University of cambridge</td>
      <td>x1086</td>
      <td>https://covid.postera.ai/covid/submissions/PAU...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>714.821</td>
      <td>-0.91270</td>
      <td>8</td>
      <td>11</td>
      <td>256.10</td>
      <td>4</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Long aliphatic chain, Dipeptide</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3.175111</td>
      <td>x0305</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>0</td>
      <td>NaN</td>
      <td>5.297134</td>
      <td>548.583191</td>
    </tr>
    <tr>
      <th>4631</th>
      <td>c1cc2cc(c(cc2c(c1)S(=O)(=O)N3CC[NH+](CC3)Cc4cc...</td>
      <td>MAK-UNK-e05327b2-2</td>
      <td>Maksym Voznyy</td>
      <td>x1402</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>837.964</td>
      <td>6.63190</td>
      <td>0</td>
      <td>9</td>
      <td>98.31</td>
      <td>2</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Hetero_hetero</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>3.561681</td>
      <td>x1392</td>
      <td>NaN</td>
      <td>active-covalent</td>
      <td>0</td>
      <td>NaN</td>
      <td>3.297014</td>
      <td>591.877563</td>
    </tr>
    <tr>
      <th>4632</th>
      <td>Cc1cccc(c1)C[NH+]2CCN(CC2)C(=O)c3ccc(cc3)C#Cc4...</td>
      <td>MAK-UNK-e4a48a85-16</td>
      <td>Maksym Voznyy</td>
      <td>x0387,x0692</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>574.794</td>
      <td>6.18892</td>
      <td>0</td>
      <td>5</td>
      <td>39.68</td>
      <td>2</td>
      <td>PASS</td>
      <td>triple bond</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4.056698</td>
      <td>x0978</td>
      <td>NaN</td>
      <td>active-covalent</td>
      <td>0</td>
      <td>NaN</td>
      <td>4.360606</td>
      <td>470.944824</td>
    </tr>
    <tr>
      <th>4633</th>
      <td>c1cc2cc(c(cc2c(c1)S(=O)(=O)N3CC[NH+](CC3)Cc4cc...</td>
      <td>MAK-UNK-e05327b2-6</td>
      <td>Maksym Voznyy</td>
      <td>x1402</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>990.183</td>
      <td>5.19160</td>
      <td>0</td>
      <td>12</td>
      <td>138.93</td>
      <td>3</td>
      <td>alpha_halo_heteroatom, secondary_halide_sulfate</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Hetero_hetero</td>
      <td>PASS</td>
      <td>Dithiomethylene_acetal</td>
      <td>Alkyl Halide</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>4.242827</td>
      <td>x0731</td>
      <td>NaN</td>
      <td>active-covalent</td>
      <td>0</td>
      <td>NaN</td>
      <td>4.193186</td>
      <td>694.333069</td>
    </tr>
    <tr>
      <th>4634</th>
      <td>Cc1cccc(c1)C[NH+]2CCN(CC2)c3cc(c(c(c3)Cl)c4cc5...</td>
      <td>MAK-UNK-e4a48a85-15</td>
      <td>Maksym Voznyy</td>
      <td>x0387,x0692</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>659.687</td>
      <td>7.36362</td>
      <td>1</td>
      <td>7</td>
      <td>68.36</td>
      <td>2</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>5.966927</td>
      <td>x0705</td>
      <td>NaN</td>
      <td>active-covalent</td>
      <td>0</td>
      <td>NaN</td>
      <td>1.473711</td>
      <td>503.583801</td>
    </tr>
  </tbody>
</table>
<p>4635 rows × 48 columns</p>
</div>




```python
moonshot.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SMILES</th>
      <th>TITLE</th>
      <th>creator</th>
      <th>fragments</th>
      <th>link</th>
      <th>real_space</th>
      <th>SCR</th>
      <th>BB</th>
      <th>extended_real_space</th>
      <th>in_molport_or_mcule</th>
      <th>in_ultimate_mcule</th>
      <th>in_emolecules</th>
      <th>covalent_frag</th>
      <th>covalent_warhead</th>
      <th>acrylamide</th>
      <th>acrylamide_adduct</th>
      <th>chloroacetamide</th>
      <th>chloroacetamide_adduct</th>
      <th>vinylsulfonamide</th>
      <th>vinylsulfonamide_adduct</th>
      <th>nitrile</th>
      <th>nitrile_adduct</th>
      <th>MW</th>
      <th>cLogP</th>
      <th>HBD</th>
      <th>HBA</th>
      <th>TPSA</th>
      <th>num_criterion_violations</th>
      <th>BMS</th>
      <th>Dundee</th>
      <th>Glaxo</th>
      <th>Inpharmatica</th>
      <th>LINT</th>
      <th>MLSMR</th>
      <th>PAINS</th>
      <th>SureChEMBL</th>
      <th>PostEra</th>
      <th>ORDERED</th>
      <th>MADE</th>
      <th>ASSAYED</th>
      <th>Hybrid2</th>
      <th>docked_fragment</th>
      <th>Mpro-x1418_dock</th>
      <th>site</th>
      <th>number_of_overlapping_fragments</th>
      <th>overlapping_fragments</th>
      <th>overlap_score</th>
      <th>volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>c1ccc(cc1)n2c3cc(c(cc3c(=O)c(c2[O-])c4cccnc4)F)Cl</td>
      <td>MAK-UNK-9e4a73aa-2</td>
      <td>Maksym Voznyy</td>
      <td>x1418</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>366.779</td>
      <td>4.51890</td>
      <td>0</td>
      <td>3</td>
      <td>50.27</td>
      <td>0</td>
      <td>PASS</td>
      <td>beta-keto/anhydride</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Ketone, Dye 11</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-11.881256</td>
      <td>x1418</td>
      <td>1.206534</td>
      <td>active-covalent</td>
      <td>3</td>
      <td>x0434,x0678,x0830</td>
      <td>3.208124</td>
      <td>271.986084</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cc1ccncc1n2c(=O)ccc3c2CCCN3CC(=[NH2+])N</td>
      <td>KIM-UNI-60f168f5-7</td>
      <td>Kim Tai Tran, University of Copenhagen</td>
      <td>x0107,x0991</td>
      <td>https://covid.postera.ai/covid/submissions/KIM...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>297.362</td>
      <td>1.22949</td>
      <td>2</td>
      <td>5</td>
      <td>88.00</td>
      <td>0</td>
      <td>PASS</td>
      <td>imine, imine</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>acyclic C=N-H</td>
      <td>Imine 3</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-11.654112</td>
      <td>x0107</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0107,x1412,x1392</td>
      <td>4.753475</td>
      <td>232.815506</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c1ccc(cc1)n2c3cc(c(cc3c(=O)n(c2=O)c4cnccn4)F)Cl</td>
      <td>MAK-UNK-9e4a73aa-14</td>
      <td>Maksym Voznyy</td>
      <td>x1418</td>
      <td>https://covid.postera.ai/covid/submissions/MAK...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>368.755</td>
      <td>2.72410</td>
      <td>0</td>
      <td>6</td>
      <td>69.78</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-10.460650</td>
      <td>x0678</td>
      <td>2.716276</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0678,x1412,x1392</td>
      <td>5.520980</td>
      <td>266.688721</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cc1ccncc1N(C=C)[C@H]([C@@H](C)[C@@H]2CN=Cc3c2c...</td>
      <td>AUS-WAB-916db9c0-1</td>
      <td>Austin D. Chivington, Wabash College</td>
      <td>x0107,x1077,x1374</td>
      <td>https://covid.postera.ai/covid/submissions/AUS...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>351.450</td>
      <td>3.51932</td>
      <td>1</td>
      <td>5</td>
      <td>57.95</td>
      <td>0</td>
      <td>non_ring_acetal</td>
      <td>het-C-het not in ring</td>
      <td>PASS</td>
      <td>Filter10_Terminal_vinyl</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-9.516450</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0434,x0831,x0678</td>
      <td>3.446572</td>
      <td>284.195312</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c1ccc2c(c1)ncc(n2)/C=C/C(=O)c3cccc(c3)O</td>
      <td>DRV-DNY-ae159ed1-12</td>
      <td>Dr. Vidya Desai, Dnyanprassarak Mandals Colleg...</td>
      <td>x1249</td>
      <td>https://covid.postera.ai/covid/submissions/DRV...</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>FALSE</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>276.295</td>
      <td>3.23150</td>
      <td>1</td>
      <td>4</td>
      <td>63.08</td>
      <td>0</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>Filter44_michael_acceptor2</td>
      <td>PASS</td>
      <td>Ketone, Dye 9, vinyl michael acceptor1</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>PASS</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>-9.243208</td>
      <td>x0678</td>
      <td>NaN</td>
      <td>active-noncovalent</td>
      <td>3</td>
      <td>x0434,x0678,x0830</td>
      <td>2.865147</td>
      <td>220.275421</td>
    </tr>
  </tbody>
</table>
</div>




```python
moonshot['Mpro-x1418_dock'].isnull().sum() # Lots of missing Mpro dock scores
```




    4586



While there are a lot of different fragments to which the small molecule can bind, there are two "classes", active-covalent and active-noncovalent (possibly referring to sites that covalently bond?)

This presents a way to logically bisect the data based on some fundamental chemistry of the binding pocket.


```python
moonshot['docked_fragment'].value_counts()
```




    x0678    940
    x0749    771
    x0104    347
    x0831    283
    x0830    281
    x0195    269
    x0161    252
    x0107    201
    x0072    172
    x1077    127
    x1392    107
    x1093    107
    x0434    105
    x0874     81
    x1385     69
    x1418     58
    x1334     50
    x0967     46
    x0397     42
    x0946     38
    x0692     37
    x0759     37
    x1386     35
    x0395     29
    x0305     24
    x1311     16
    x0708     13
    x0774     12
    x1380     10
    x1412      7
    x1374      7
    x1348      6
    x0770      5
    x1249      5
    x0387      5
    x0736      4
    x0705      4
    x1358      3
    x0426      3
    x1375      3
    x0734      3
    x0540      3
    x0354      3
    x1382      3
    x0755      1
    x1458      1
    x0689      1
    x0769      1
    x0981      1
    x0978      1
    x0731      1
    x1493      1
    x0771      1
    x1478      1
    x1384      1
    x1351      1
    Name: docked_fragment, dtype: int64




```python
moonshot['site'].value_counts()
```




    active-noncovalent    2799
    active-covalent       1836
    Name: site, dtype: int64



We can examine the same correlations, but now for each type of site, and look at the hybrid docking score correlations.

The biggest trend differences appear with the partition coefficient and number of hydrogen bond donors, but still the correlations are extremely weak


```python
site_type = 'active-noncovalent'
fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=100)
cols = ['MW', 'cLogP', 'HBD', 'HBA', 'TPSA', 'Hybrid2']
ax.matshow(moonshot[moonshot['site']==site_type][cols].corr(), cmap='RdBu')

ax.set_xticks([i for i,_ in enumerate(cols)])
ax.set_xticklabels(cols)

ax.set_yticks([i for i,_ in enumerate(cols)])
ax.set_yticklabels(cols)

for i, (rowname, row) in enumerate(moonshot[moonshot['site']==site_type][cols].corr().iterrows()):
    for j, (key, val) in enumerate(row.iteritems()):
        ax.annotate(f"{val:0.2f}", xy=(i,j), xytext=(-10, -5), textcoords="offset points")
ax.set_title(f"Docking to {site_type}")
```




    Text(0.5, 1.05, 'Docking to active-noncovalent')




![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_68_1.png)



```python
site_type = 'active-covalent'
fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=100)
cols = ['MW', 'cLogP', 'HBD', 'HBA', 'TPSA', 'Hybrid2']
ax.matshow(moonshot[moonshot['site']==site_type][cols].corr(), cmap='RdBu')

ax.set_xticks([i for i,_ in enumerate(cols)])
ax.set_xticklabels(cols)

ax.set_yticks([i for i,_ in enumerate(cols)])
ax.set_yticklabels(cols)

for i, (rowname, row) in enumerate(moonshot[moonshot['site']==site_type][cols].corr().iterrows()):
    for j, (key, val) in enumerate(row.iteritems()):
        ax.annotate(f"{val:0.2f}", xy=(i,j), xytext=(-10, -5), textcoords="offset points")
ax.set_title(f"Docking to {site_type}")
```




    Text(0.5, 1.05, 'Docking to active-covalent')




![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_69_1.png)


In general, lower docking score seem better, so the noncovalent sites might present more optimal binding locations (see histogram below).
This seems non-intuitive because, if active-covalent really means sites that bond covalently, then covalent bonds would seem more energetically favorable than non-covalent interactions.
Alternatively, forming covalent bonds might suggest an unstable region of the complex that could be shielded from the surroundings, inhibiting any sort of small molecule from binding the pocket?
Expert opinion would be much appreciated here


```python

fig, ax = plt.subplots(1,1, figsize=(8,6), dpi=100)
covalent_mean = moonshot[moonshot['site']=='active-covalent']['Hybrid2'].mean()
noncovalent_mean = moonshot[moonshot['site']=='active-noncovalent']['Hybrid2'].mean()

ax.hist(moonshot[moonshot['site']=='active-covalent']['Hybrid2'], alpha=0.5, 
        label=f'active-covalent (mean={covalent_mean:.3f})')
ax.hist(moonshot[moonshot['site']=='active-noncovalent']['Hybrid2'], alpha=0.5, 
        label=f'active-noncovalent (mean={noncovalent_mean:.3f})')

ax.set_title(f"Hybrid2 histogram")
ax.set_xlabel("Hybrid2 score")
ax.legend()
```




    <matplotlib.legend.Legend at 0x7fac6b459850>




![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_71_1.png)



```python
from rdkit import Chem
```


```python
rdkit_smiles = [Chem.MolFromSmiles(a) for a in moonshot.sort_values('Hybrid2', ascending=True)['SMILES'].head(10)]
scores = [f"{a:.3f}" for a in moonshot.sort_values('Hybrid2', ascending=True)['Hybrid2'].head(10)]
```


```python
img=Chem.Draw.MolsToGridImage(rdkit_smiles,molsPerRow=5,subImgSize=(200,200),
                             legends=scores)
```


```python
img
```




![png](images/2020-05-06-study_covid_moonshot_files/2020-05-06-study_covid_moonshot_75_0.png)




```python

```

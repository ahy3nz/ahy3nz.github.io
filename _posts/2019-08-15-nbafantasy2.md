---
title: 'Fantasy NBA 2'
date: 2019-08-15
permalink: /posts/2019/08/nbafantasy2/
tags:
  - personal
---
Part 2 of evaluating fantasy NBA draft picks - modeling and sampling
for expected fantasy output.


```python
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
import scipy
from scipy.stats import expon, skewnorm, norm

import nba_api
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import shotchartdetail, playercareerstats, playergamelog

import ballDontLie 
from ballDontLie.util.api_nba import find_player_id
from ballDontLie.util.fantasy import compute_fantasy_points
```


```python
seasons_range = ['2018-19', '2017-18', '2016-17', '2015-16']
players_range = ['Anthony Davis', 'James Harden', 'Stephen Curry', 'Giannis Antetokounmpo', 'Karl-Anthony Towns',
                'Nikola Jokic', 'Joel Embiid', 'Paul George', 'Kawhi Leonard', 'Damian Lillard', 'Jimmy Butler',
                'LeBron James', "Bradley Beal"]
player_id_map = {a: find_player_id(a) for a in players_range}
```

For the various players and the various seasons, let's look at the distributions of some of their box stats


```python
for player, player_id in player_id_map.items():
    fig, ax = plt.subplots(1,1)
    df = pd.read_csv('data/{}.csv'.format(player.replace(" ","")))
    df.hist(column=['FGM', 'FGA', 'FTM', 'FTA', "REB", 'AST',
                     'STL', 'BLK', "PTS"], ax=ax)
    fig.suptitle(player)
                     
```

    /Users/ayang41/anaconda3/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:3296: UserWarning: To output multiple subplots, the figure containing the passed axes is being cleared
      exec(code_obj, self.user_global_ns, self.user_ns)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_1.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_2.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_3.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_4.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_5.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_6.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_7.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_8.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_9.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_10.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_11.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_12.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_3_13.png)


I'm going off by what these distributions sort of look like over all the players:
* AST: Skewed normal
* BLK: Exponential
* FGA: Normal
* FGM: Normal
* FTA: Skewed normal
* FTM: Skewed normal
* PTS: Normal
* REB: Skewed normal
* STL: Skewed normal

For all players, I'm going to model each box stat as such. Given the gamelog data (blue), fit the model to that data, 
generate some values with that model (orange), and compare to the actual gamelog data.

Some comments:

For the "bigger" numbers like PTS, FGA, FGM, REB, the model distributions fit pretty well. 

For the "smaller" numbers like BLK or STL (a player will usually have 0, 1, 2, 3, or maybe 4 of that stat) - these numbers are more discrete than the "bigger numbers". If you can score points between 0 and 40, each actually reported points behaves more continuously since there is more variety.

From earlier work with PyMC for Bayesian probability modeling, I could have tried using PyMC to sample parameters for each stat-distribution, rather than just do a singular fitting. While that could help report a variety of parameters for each stat-distribution in addition to a sense of variation or uncertainty, I don't think it's super necessary to really venture into exploring the different distributions and their parameters that could fit each box stat; the fitting schemes via scipy seem to work well.

It's possible there are better models to fit some of the data - I can't say my brain-database of statistical models is extensive, so I just kinda perused through `scipy.stats`.

Fitting a distribution helps formalize how much a player's game can vary (is he consistently a 20ppg player? Or are is he hot and cold between 10 and 30 ppg?) Furthermore, if a player is out (injured or some other reason), that implicitly gets captured by a gamelog of 0pts, 0reb, etc. This is definitely important in fantasy because some may value a more reliable/consistent player who will show up to 80/82 games rather than a glass weapon who could drop 50 points, but will only play 40-50/82 games

These distributions assume we can ignore: coaching changes, team roster changes, and maybe player development. For player development, a younger player between 2015-2019 will demonstrate huge variance in two ways - young players are inconsistent game-to-game, but young players can also develop rapidly season-by-season. At the very least, these distributions try to describe variance, which shows room where a young player could go off or bust on a given night. Factoring season-by-season improvement will be hard - one would need to try to forecast a player's future stats rather than draw samples from a "fixed" distribution based on previous stats


```python
stat_model_map = {"AST": skewnorm, "BLK": expon, "FGA": norm, "FGM": norm,
                 "FTA": skewnorm, "FTM": skewnorm, "PTS": norm, "REB": skewnorm,
                 "STL": skewnorm}
```


```python
for player, player_id in player_id_map.items():
    fig, axarray = plt.subplots(3,3)
    df = pd.read_csv('data/{}.csv'.format(player.replace(" ","")))
    for i, (stat, model) in enumerate(stat_model_map.items()):
        row = i // 3
        col = i % 3
        axarray[row, col].hist(df[stat], alpha=0.3)
        axarray[row, col].set_title(stat)
        params = model.fit(df[stat])
        axarray[row, col].hist(model.rvs(*params, size=len(df[stat])), alpha=0.3)

    
    
    fig.suptitle(player)
    fig.tight_layout()
                     
```


![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_0.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_1.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_2.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_3.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_4.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_5.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_6.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_7.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_8.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_9.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_10.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_11.png)



![png](/images/2019-08-15-nbafantasy2_files/2019-08-15-nbafantasy2_6_12.png)


At this point, for each player and box stat, we have a distribution that can describe their game-by-game performance. Maybe we can sample from this distribution 82 times (82 games per season) to get an idea of the fantasy points they'll yield (the fantasy points will depend on the league settings and how each league weights the box stats).

To simulate a season for a player, we will model the distribution for each box stat, and sample from it 82 times. This is our simulated season.


```python
simulated_season = pd.DataFrame()
for player, player_id in player_id_map.items():
    df = pd.read_csv('data/{}.csv'.format(player.replace(" ","")))
    simulated_player_log = {}
    for stat, model in stat_model_map.items():
        params = model.fit(df[stat])
        sample = model.rvs(*params, size=82)
        simulated_player_log[stat] = sample
    simulated_player_log_series = pd.Series(data=simulated_player_log, name=player)
    simulated_season = simulated_season.append(simulated_player_log_series)
```

In addition to getting an 82-list of ast, blk, fga, etc. We can compute an 82-list of fantasy points (point values will depend on the league, but the default args for `compute_fantasy_points` are pulled from ESPN head-to-head points league default categories


```python
simulated_season = compute_fantasy_points(simulated_season)
simulated_season
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
      <th>AST</th>
      <th>BLK</th>
      <th>FGA</th>
      <th>FGM</th>
      <th>FTA</th>
      <th>FTM</th>
      <th>PTS</th>
      <th>REB</th>
      <th>STL</th>
      <th>FP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Anthony Davis</th>
      <td>[5.9597699533362665, 3.218551371084111, 1.1022...</td>
      <td>[8.578394640988105, 0.026198088049728254, 1.84...</td>
      <td>[20.21599540596214, 19.100646432695086, 26.528...</td>
      <td>[13.872698337473729, 7.918853515554012, 9.5048...</td>
      <td>[14.873567185676178, 9.685624594656804, 10.241...</td>
      <td>[1.0995020882840587, 7.294623211426735, 7.2877...</td>
      <td>[30.67865219506473, 15.260082139070928, 17.769...</td>
      <td>[13.498196895282199, 5.04028872591627, 16.2989...</td>
      <td>[1.1347148361527453, 1.7312487922029391, 1.068...</td>
      <td>[39.732366354943515, 11.703574815952832, 18.10...</td>
    </tr>
    <tr>
      <th>James Harden</th>
      <td>[6.725125502991835, 16.337635303737578, 8.6225...</td>
      <td>[0.05211498826964095, 0.5114333703840529, 0.08...</td>
      <td>[31.451667457570714, 15.992361996925819, 19.37...</td>
      <td>[9.307080381863589, 8.033940758354536, 11.1817...</td>
      <td>[12.808604943217352, 12.428544803014987, 11.82...</td>
      <td>[-0.06498946030802255, 8.515864278527303, 3.23...</td>
      <td>[20.97714144699512, 32.02286792578204, 22.5610...</td>
      <td>[6.9093167207067765, 7.988699867679405, 5.5192...</td>
      <td>[0.32632320180968244, 0.6531429770923043, 2.89...</td>
      <td>[-0.0281596184594477, 45.64267768161641, 22.89...</td>
    </tr>
    <tr>
      <th>Stephen Curry</th>
      <td>[12.279924280581394, 7.879981071011731, 5.9023...</td>
      <td>[0.06292945230558766, 0.09033987679766506, 0.1...</td>
      <td>[28.850986709915357, 24.43691542042258, 17.400...</td>
      <td>[12.472086572331825, 9.555215666190444, 8.2599...</td>
      <td>[12.977887030692326, 3.1789263709957845, -0.86...</td>
      <td>[-0.010198935847337554, 3.516792829956679, 6.6...</td>
      <td>[32.67407066270164, 24.97462588841426, 29.3427...</td>
      <td>[3.9327364866510104, 7.6111446721103615, 4.726...</td>
      <td>[3.590539393208442, 1.7898067047882282, 2.1206...</td>
      <td>[23.173214171324872, 27.802064917851006, 40.60...</td>
    </tr>
    <tr>
      <th>Giannis Antetokounmpo</th>
      <td>[0.028588769334945585, 3.262013551923701, 3.40...</td>
      <td>[1.9174837837589362, 6.695156990323455, 0.2378...</td>
      <td>[7.172387181137829, 15.12244530666314, 20.4210...</td>
      <td>[10.4727740339432, 16.131669525478472, 9.71417...</td>
      <td>[11.746787857935951, 5.523726797994956, 15.646...</td>
      <td>[8.191304596222707, 1.8995774627238957, 9.6062...</td>
      <td>[38.321060006893674, 17.906593672693283, 7.885...</td>
      <td>[19.219901699676136, 9.067580346050558, 9.3800...</td>
      <td>[0.19042746270131458, 0.8734312501637576, 1.68...</td>
      <td>[59.422365313457135, 35.189850694699025, 5.842...</td>
    </tr>
    <tr>
      <th>Karl-Anthony Towns</th>
      <td>[4.2665586948098095, 2.239352382015603, 2.8486...</td>
      <td>[1.5142034413304508, 0.5455258265042157, 0.238...</td>
      <td>[16.282437483033767, 16.64803799271481, 15.635...</td>
      <td>[6.438364846075592, 9.772816680970095, 12.7305...</td>
      <td>[-0.7312367428381779, 8.855766357633657, 7.839...</td>
      <td>[12.274313450703058, 3.407774443197603, 1.1792...</td>
      <td>[39.024821580722396, 18.827116373850785, 6.703...</td>
      <td>[13.494959487083559, 6.618974662978514, 13.515...</td>
      <td>[1.1278422771833472, 0.969599487392581, 1.1214...</td>
      <td>[62.589863037712625, 16.87735550656093, 14.862...</td>
    </tr>
    <tr>
      <th>Nikola Jokic</th>
      <td>[2.2495283329958697, 4.313187987247313, 13.591...</td>
      <td>[0.12814150793734586, 1.0322534945133215, 0.22...</td>
      <td>[15.985739566133859, 3.729397368467904, 13.122...</td>
      <td>[2.2695563355573927, 4.6162142202699465, 12.37...</td>
      <td>[1.140250956051838, 0.7396592852362885, 1.0218...</td>
      <td>[1.728722802038216, 4.521343806420914, 2.80787...</td>
      <td>[18.321586322668246, 14.595347285061017, 36.97...</td>
      <td>[12.219234056488448, 6.129650356878898, 10.120...</td>
      <td>[0.46503541814884397, 2.3095400747975847, 4.42...</td>
      <td>[20.255814253648666, 33.0484805714848, 66.3739...</td>
    </tr>
    <tr>
      <th>Joel Embiid</th>
      <td>[-0.5626100721097174, 6.31861993131192, 2.4004...</td>
      <td>[3.745530342633801, 1.2329342353392367, 0.6703...</td>
      <td>[23.858670698530716, 14.171117481078998, 22.38...</td>
      <td>[8.531705694178644, 6.592226230945353, 5.78683...</td>
      <td>[3.636279316429121, 10.099346265562726, 5.7510...</td>
      <td>[5.614085976697099, 10.835254446544083, 10.530...</td>
      <td>[21.316896006823455, 37.63500403364662, 15.459...</td>
      <td>[11.773784732117047, 12.398188556975562, 7.601...</td>
      <td>[0.5668458333098912, 0.5312890917217526, 0.310...</td>
      <td>[23.491288498690384, 51.2730527798428, 14.6226...</td>
    </tr>
    <tr>
      <th>Paul George</th>
      <td>[3.747746441549796, 1.4181358595970992, 5.9393...</td>
      <td>[0.1381904316955113, 1.0488620581494552, 0.056...</td>
      <td>[11.961150191031587, 21.844008115116335, 28.35...</td>
      <td>[8.270970012797578, 10.471683200512018, 8.6575...</td>
      <td>[7.461807193390171, 4.502386500618187, 3.01446...</td>
      <td>[2.330402296133858, 7.49604672023695, 4.815137...</td>
      <td>[24.937871476385926, 24.159147380580418, 16.18...</td>
      <td>[8.39563811879939, 10.92290776474599, 6.403018...</td>
      <td>[0.09488761613910184, 2.6952464170520942, 5.14...</td>
      <td>[28.492749009079404, 31.865634785139502, 15.83...</td>
    </tr>
    <tr>
      <th>Kawhi Leonard</th>
      <td>[1.9111587130339496, 3.861952310135943, 5.2338...</td>
      <td>[1.2732246842101056, 0.3281317197316312, 0.826...</td>
      <td>[11.360972973682683, 9.585929657228244, 21.373...</td>
      <td>[12.958518216160943, 6.903991550195954, 7.6377...</td>
      <td>[16.432022800206976, 2.660269738496524, 7.0656...</td>
      <td>[7.797850653453996, 4.33192367872911, 8.529788...</td>
      <td>[10.856905074154483, 20.15734528602171, 24.835...</td>
      <td>[2.7909724123148116, 5.850300462819424, 10.213...</td>
      <td>[0.42294253595485454, 1.7846431134663236, 1.04...</td>
      <td>[10.218576515393485, 30.97208872537533, 29.882...</td>
    </tr>
    <tr>
      <th>Damian Lillard</th>
      <td>[10.677268995445747, 4.180050225294141, 9.8551...</td>
      <td>[0.15727721227445404, 0.9580314076148106, 0.00...</td>
      <td>[19.284257429878746, 17.31204360854523, 15.118...</td>
      <td>[5.286410181032442, 6.34510375863316, 12.39764...</td>
      <td>[14.714450025225243, 17.748107775039408, 3.093...</td>
      <td>[0.8363791899940345, 3.588830309255546, 6.4196...</td>
      <td>[37.76255224153543, 22.075858810385594, 27.925...</td>
      <td>[4.773627696731243, 4.621992467769962, 7.50805...</td>
      <td>[1.3504753152495277, 0.08428523535351211, 1.64...</td>
      <td>[26.845283377158893, 6.794000830722091, 47.545...</td>
    </tr>
    <tr>
      <th>Jimmy Butler</th>
      <td>[4.209873691051277, 14.197422029412843, 5.9746...</td>
      <td>[0.21814011560469518, 0.30447250932287995, 0.9...</td>
      <td>[7.725284038601737, 10.845475416658621, 13.048...</td>
      <td>[8.892692053613146, 7.059212392324561, 10.9128...</td>
      <td>[3.190729877814753, 7.330081264685362, 12.8000...</td>
      <td>[5.125910874417352, 12.424270820405049, 6.3800...</td>
      <td>[36.69969739697747, 11.83426127688215, 25.6908...</td>
      <td>[7.963523560116641, 7.136457070141159, 7.17463...</td>
      <td>[0.9233590447096361, 1.9610296115104289, 0.402...</td>
      <td>[53.11718282007372, 36.74156902865509, 31.6143...</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>[4.383784518789513, 11.422636546717118, 1.8432...</td>
      <td>[0.49669613448309097, 0.22661116243441118, 0.8...</td>
      <td>[16.273532104553603, 18.89206066775093, 20.779...</td>
      <td>[11.305028513244693, 16.344920059482988, 9.013...</td>
      <td>[11.481284754437317, 7.998971744828838, 7.7999...</td>
      <td>[1.894279478363282, 8.134466938477937, 5.16229...</td>
      <td>[22.984748887841146, 29.481915659477433, 34.58...</td>
      <td>[10.903870207177395, 6.4820198722838445, 8.954...</td>
      <td>[1.42399775829285, 3.7503728208149565, 0.23432...</td>
      <td>[25.637588639201052, 48.951910647108924, 32.10...</td>
    </tr>
    <tr>
      <th>Bradley Beal</th>
      <td>[2.0746630930670698, 6.754432736863302, 8.1447...</td>
      <td>[0.14781341009400328, 0.1456013540011799, 0.25...</td>
      <td>[14.470299375421758, 17.38438420787558, 8.2879...</td>
      <td>[12.937693524852126, 7.286039336111235, 3.5386...</td>
      <td>[8.81124846003997, 5.690489995458582, 8.479262...</td>
      <td>[1.582589186451698, 1.4114941956513458, 1.5218...</td>
      <td>[9.943219061144877, 31.48297458824507, 12.0875...</td>
      <td>[6.252901453057193, 3.3518717200947448, 6.4345...</td>
      <td>[1.2849268221556598, 1.965109099251958, 1.7432...</td>
      <td>[10.942258715360898, 29.322648826884677, 16.96...</td>
    </tr>
  </tbody>
</table>
</div>



To make things simpler to read, we will compress the dataframe into totals for the entire season, including
the total fantasy points for that season


```python
simulated_totals = simulated_season.copy()
for col in simulated_totals.columns:
    simulated_totals[col] = [sum(a) for a in simulated_totals[col]]
simulated_totals.sort_values('FP', ascending=False)
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
      <th>AST</th>
      <th>BLK</th>
      <th>FGA</th>
      <th>FGM</th>
      <th>FTA</th>
      <th>FTM</th>
      <th>PTS</th>
      <th>REB</th>
      <th>STL</th>
      <th>FP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>James Harden</th>
      <td>747.830702</td>
      <td>50.407450</td>
      <td>1762.218494</td>
      <td>767.004764</td>
      <td>864.891220</td>
      <td>728.555527</td>
      <td>2578.629298</td>
      <td>554.455532</td>
      <td>138.349447</td>
      <td>2938.123005</td>
    </tr>
    <tr>
      <th>LeBron James</th>
      <td>644.902311</td>
      <td>53.848741</td>
      <td>1528.397672</td>
      <td>867.848033</td>
      <td>535.034764</td>
      <td>397.555538</td>
      <td>2299.442187</td>
      <td>621.522299</td>
      <td>111.886953</td>
      <td>2933.573626</td>
    </tr>
    <tr>
      <th>Giannis Antetokounmpo</th>
      <td>411.042856</td>
      <td>113.488274</td>
      <td>1307.959350</td>
      <td>724.625241</td>
      <td>567.077934</td>
      <td>473.292758</td>
      <td>2117.891129</td>
      <td>777.568640</td>
      <td>99.125458</td>
      <td>2841.997073</td>
    </tr>
    <tr>
      <th>Stephen Curry</th>
      <td>506.145110</td>
      <td>24.894564</td>
      <td>1409.398642</td>
      <td>773.939875</td>
      <td>380.741190</td>
      <td>329.111538</td>
      <td>2362.073947</td>
      <td>420.829387</td>
      <td>137.321349</td>
      <td>2764.175938</td>
    </tr>
    <tr>
      <th>Karl-Anthony Towns</th>
      <td>205.572031</td>
      <td>131.434058</td>
      <td>1226.963594</td>
      <td>778.868996</td>
      <td>426.204892</td>
      <td>330.018614</td>
      <td>1801.931600</td>
      <td>946.156019</td>
      <td>79.285543</td>
      <td>2620.098373</td>
    </tr>
    <tr>
      <th>Anthony Davis</th>
      <td>196.578860</td>
      <td>175.150824</td>
      <td>1622.873347</td>
      <td>816.723251</td>
      <td>696.171419</td>
      <td>456.367971</td>
      <td>2260.519240</td>
      <td>900.291765</td>
      <td>124.250201</td>
      <td>2610.837347</td>
    </tr>
    <tr>
      <th>Joel Embiid</th>
      <td>234.186200</td>
      <td>163.287594</td>
      <td>1435.194243</td>
      <td>660.241425</td>
      <td>734.381531</td>
      <td>582.971875</td>
      <td>2023.816505</td>
      <td>1017.135889</td>
      <td>65.729671</td>
      <td>2577.793384</td>
    </tr>
    <tr>
      <th>Kawhi Leonard</th>
      <td>254.481192</td>
      <td>58.379376</td>
      <td>1429.088809</td>
      <td>694.413700</td>
      <td>508.884724</td>
      <td>464.766005</td>
      <td>1988.136387</td>
      <td>495.686852</td>
      <td>164.975375</td>
      <td>2182.865355</td>
    </tr>
    <tr>
      <th>Damian Lillard</th>
      <td>517.704615</td>
      <td>22.305801</td>
      <td>1532.940857</td>
      <td>714.316068</td>
      <td>628.611949</td>
      <td>519.754053</td>
      <td>2133.813013</td>
      <td>339.602730</td>
      <td>96.813527</td>
      <td>2182.757000</td>
    </tr>
    <tr>
      <th>Jimmy Butler</th>
      <td>413.035473</td>
      <td>38.031837</td>
      <td>1281.235832</td>
      <td>606.988700</td>
      <td>589.880982</td>
      <td>492.461756</td>
      <td>1824.943569</td>
      <td>467.421917</td>
      <td>151.978400</td>
      <td>2123.744838</td>
    </tr>
    <tr>
      <th>Nikola Jokic</th>
      <td>366.181073</td>
      <td>64.952508</td>
      <td>953.642637</td>
      <td>511.123020</td>
      <td>260.587381</td>
      <td>218.712031</td>
      <td>1287.338234</td>
      <td>741.543901</td>
      <td>101.564244</td>
      <td>2077.184995</td>
    </tr>
    <tr>
      <th>Paul George</th>
      <td>311.926094</td>
      <td>33.618702</td>
      <td>1485.702959</td>
      <td>679.384305</td>
      <td>533.333046</td>
      <td>431.228670</td>
      <td>1877.436210</td>
      <td>558.674019</td>
      <td>164.005182</td>
      <td>2037.237177</td>
    </tr>
    <tr>
      <th>Bradley Beal</th>
      <td>347.030138</td>
      <td>39.924951</td>
      <td>1495.795748</td>
      <td>668.972669</td>
      <td>400.151491</td>
      <td>350.540788</td>
      <td>1754.362308</td>
      <td>325.133155</td>
      <td>93.936173</td>
      <td>1683.952942</td>
    </tr>
  </tbody>
</table>
</div>



Generally speaking, this method is in-line with many other fantasy predictions. James Harden, Anthony Davis, 
LeBron James, Karl-Anthony Towns, Steph Curry, Giannis, and Joel Embiid all top the list. 

In this "simulation" our sample size was 82 to match a season. We could repeat this simulation multiple times (so 82 * n times). That effectively increases our sample size from 82 to much larger.

Sampling enough is always a question, so we'll address that by simulating multiple seasons. Discussion of the approach will follow later


```python
def simulate_n_seasons(player_id_map, stat_model_map, n=5):
    # For a season, we just want the player, FP, and the rank
    # Initialize dictionary of dictionary of lists to store this information across "epochs"
    epoch_results = {}
    for player in player_id_map:
        epoch_results[player] = {'FP':[], 'rank':[]}
        
    for i in range(n):
        # Just copy-pasted code for convenience in a notebook
        # If this were a python script, I would probably put these functions in a module/library somewhere
        # Model the distribution of a player's box stats, simulate 82 times, compute fantasy points
        simulated_season = pd.DataFrame()
        for player, player_id in player_id_map.items():
            df = pd.read_csv('data/{}.csv'.format(player.replace(" ","")))
            simulated_player_log = {}
            for stat, model in stat_model_map.items():
                params = model.fit(df[stat])
                sample = model.rvs(*params, size=82)
                simulated_player_log[stat] = sample
            simulated_player_log_series = pd.Series(data=simulated_player_log, name=player)
            simulated_season = simulated_season.append(simulated_player_log_series)
        simulated_season = compute_fantasy_points(simulated_season)
        simulated_totals = simulated_season.copy()
        for col in simulated_totals.columns:
            simulated_totals[col] = [sum(a) for a in simulated_totals[col]]
        simulated_totals = simulated_totals.sort_values('FP', ascending=False)


        # Store the fantasy points and player rank for that simulated season
        for player in player_id_map:
            epoch_results[player]['FP'].append(simulated_totals[simulated_totals.index==player]['FP'].values[0])
            epoch_results[player]['rank'].append(simulated_totals.index.get_loc(player))
    return epoch_results

epoch_results = simulate_n_seasons(player_id_map, stat_model_map, n=10)
```


```python
pprint(epoch_results)
```

    {'Anthony Davis': {'FP': [2600.2718925173745,
                              2845.699762026843,
                              2732.8142372134657,
                              2841.1189014237507,
                              2971.6018500231144,
                              2513.4197141216027,
                              2671.907808833641,
                              2771.33344794354,
                              2642.4401320506413,
                              2668.8391100382087],
                       'rank': [3, 0, 2, 1, 0, 5, 2, 2, 2, 3]},
     'Bradley Beal': {'FP': [2017.9614666080413,
                             1898.010517178849,
                             1804.7941022033058,
                             1763.042431335895,
                             1730.5132495765397,
                             1715.734094440131,
                             1838.502289868321,
                             1867.0678145772936,
                             1718.2489707303305,
                             1669.67185265485],
                      'rank': [11, 12, 12, 12, 12, 12, 12, 12, 12, 12]},
     'Damian Lillard': {'FP': [2382.8646505353163,
                               2048.068798160208,
                               2477.586060738951,
                               2175.7697065125562,
                               2118.5140365357565,
                               2247.094879780937,
                               2072.756774030209,
                               2139.1192107972674,
                               2370.629467489499,
                               2257.211660962338],
                        'rank': [7, 10, 6, 9, 9, 8, 10, 10, 7, 7]},
     'Giannis Antetokounmpo': {'FP': [2576.219466934943,
                                      2686.76587859472,
                                      2493.6482163098117,
                                      2702.8549155043497,
                                      2569.2853027544083,
                                      2551.6813152859013,
                                      2414.5128757456737,
                                      2561.3702273681874,
                                      2499.1103752312756,
                                      2510.8640790442428],
                               'rank': [4, 3, 5, 2, 4, 3, 6, 3, 5, 6]},
     'James Harden': {'FP': [2887.670929514508,
                             2843.389071489731,
                             3116.7281411190593,
                             2857.6232728678133,
                             2804.0017687844643,
                             2737.302937823686,
                             3007.392134417434,
                             2919.5661859360953,
                             2967.3026370340576,
                             2964.2404529023775],
                      'rank': [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]},
     'Jimmy Butler': {'FP': [2161.113361146545,
                             1909.6945703788665,
                             1986.2703953730904,
                             2136.973720154527,
                             2232.480783438934,
                             2166.1726145299494,
                             1962.9186078450955,
                             2004.7241270670554,
                             2039.3267955068004,
                             1973.2565162804035],
                      'rank': [9, 11, 11, 11, 8, 11, 11, 11, 11, 11]},
     'Joel Embiid': {'FP': [2800.8172929600287,
                            2558.645668264587,
                            2517.7715107689955,
                            2382.7918864392273,
                            2575.564309480633,
                            2513.594633760708,
                            2639.2018786988106,
                            2536.501391112381,
                            2502.800851773036,
                            2546.098737416667],
                     'rank': [1, 5, 4, 6, 3, 4, 3, 4, 4, 5]},
     'Karl-Anthony Towns': {'FP': [2438.6012883732774,
                                   2597.9460772777898,
                                   2437.4608371185327,
                                   2559.1161865080608,
                                   2562.0978448642904,
                                   2655.419027730967,
                                   2479.0296092681992,
                                   2469.6417151916476,
                                   2552.6318141655534,
                                   2706.9979324697306],
                            'rank': [6, 4, 7, 4, 5, 2, 4, 5, 3, 2]},
     'Kawhi Leonard': {'FP': [2192.2405107386744,
                              2416.815987294494,
                              2274.141300566951,
                              2137.2178749787718,
                              2234.4505712447212,
                              2213.2129013592594,
                              2249.1630270595037,
                              2255.1592921722886,
                              2220.8331127315014,
                              2193.058252470087],
                       'rank': [8, 6, 8, 10, 7, 9, 7, 8, 9, 8]},
     'LeBron James': {'FP': [2718.7719659019,
                             2830.2185612255066,
                             2796.158077485558,
                             2679.1235682729816,
                             2793.4255223009245,
                             2876.54356690619,
                             2712.108129400297,
                             2785.7145304012624,
                             2789.5298777236117,
                             2929.5895036201873],
                      'rank': [2, 2, 1, 3, 2, 0, 1, 1, 1, 1]},
     'Nikola Jokic': {'FP': [2065.5848606919335,
                             2211.6735032023817,
                             2211.9020404775197,
                             2257.4205116181893,
                             2103.9926989504497,
                             2279.1497412203903,
                             2204.2763112313123,
                             2424.980161680876,
                             2294.3977539747652,
                             2097.2016861034704],
                      'rank': [10, 8, 9, 7, 10, 7, 8, 7, 8, 10]},
     'Paul George': {'FP': [1954.6580000665872,
                            2067.468713136873,
                            2048.682238913504,
                            2201.483917859578,
                            1949.90816918642,
                            2169.3138128730848,
                            2140.273072194092,
                            2242.3376266501905,
                            2052.403480766016,
                            2142.499394706088],
                     'rank': [12, 9, 10, 8, 11, 10, 9, 9, 10, 9]},
     'Stephen Curry': {'FP': [2505.2751490181836,
                              2316.5751250140206,
                              2567.5576627793876,
                              2546.082326528174,
                              2560.3301729832456,
                              2391.130817270027,
                              2446.09455104389,
                              2428.6777503086655,
                              2412.173845766393,
                              2577.680897739675],
                       'rank': [5, 7, 3, 5, 6, 6, 5, 6, 6, 4]}}


To make things prettier, we can just summarize the player ranks over all the simulated seasons, providing us an estimated average rank and error


```python
def summarize_epoch_results(epoch_results):
    summary_stats = {}
    for player in epoch_results:
        summary_stats[player] = {}
        avg_rank = np.mean(epoch_results[player]['rank'])
        std_rank = np.std(epoch_results[player]['rank'])
        summary_stats[player]['rank'] = avg_rank
        summary_stats[player]['err'] = std_rank
    return summary_stats

summary_stats = summarize_epoch_results(epoch_results)
sorted(summary_stats.items(), key=lambda v: v[1]['rank'])
```




    [('James Harden', {'rank': 0.3, 'err': 0.45825756949558394}),
     ('LeBron James', {'rank': 1.4, 'err': 0.8}),
     ('Anthony Davis', {'rank': 2.0, 'err': 1.4142135623730951}),
     ('Joel Embiid', {'rank': 3.9, 'err': 1.3}),
     ('Giannis Antetokounmpo', {'rank': 4.1, 'err': 1.3}),
     ('Karl-Anthony Towns', {'rank': 4.2, 'err': 1.5362291495737217}),
     ('Stephen Curry', {'rank': 5.3, 'err': 1.1}),
     ('Kawhi Leonard', {'rank': 8.0, 'err': 1.0954451150103321}),
     ('Damian Lillard', {'rank': 8.3, 'err': 1.4177446878757827}),
     ('Nikola Jokic', {'rank': 8.4, 'err': 1.2}),
     ('Paul George', {'rank': 9.7, 'err': 1.1}),
     ('Jimmy Butler', {'rank': 10.5, 'err': 1.02469507659596}),
     ('Bradley Beal', {'rank': 11.9, 'err': 0.3})]



# Observations (based on this approach)
Harden, LeBron, and AD are a cut above the rest. Beal is not looking too hot

# Room for improvement
* Is building a distribution from year 2015-onward a good idea?
* Pick better models to represent the distribution of a player's box stats?
* How do we account for player development? Forecasting player stats, not just modeling
* How do we account for roster/team changes?
* Can we account for hot streaks for a player?
* Is there a more robust way to deal with player injury rather than hoping for 0/0/0 in the gamelogs?
* Correlation between stats? If a player is on, they might end up playing better overall
* Can we try to time schedules? I.e. some NBA players will have 4-game weeks, can a corresponding fantasy player use that based on the fantasy schedule and truly trying to beat your fantasy opponent? 
* Is there a need to draft a player in reaction to other fantasy player draftpicks? This may depend on how specific your team roles have to be. If team roles are lax, then choose the best fantasy option. If you need to fill out a roster, then you have to start weighing your roster choices vs what opponents may end up drafting


```python

```

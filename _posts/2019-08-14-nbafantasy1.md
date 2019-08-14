---
title: 'Fantasy NBA 1'
date: 2019-08-14
permalink: /posts/2019/08/nbafantasy1/
tags:
  - personal
---
Part 1 of evaluating fantasy NBA draft picks - first gathering the relevant
data.

# Step 1) Scraping NBA Stats
I'm using [nba_api](https://github.com/swar/nba_api) to parse the nba stats website. 
While it'd be nice to put everything in one notebook, I've had to split the scraping step into a separate notebook.
Too many and too frequent URL requests lead to connection errors and data limits/throttles (even with a sleep call).
So, we'll parse the information we want and save it to a csv for future recall


```python
import time

import numpy as np
import pandas as pd
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

import nba_api
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import shotchartdetail, playercareerstats, playergamelog

import ballDontLie 
from ballDontLie.util.api_nba import *
```


```python
seasons_range = ['2018-19', '2017-18', '2016-17', '2015-16']
players_range = ['Anthony Davis', 'James Harden', 'Stephen Curry', 'Giannis Antetokounmpo', 'Karl-Anthony Towns',
                'Nikola Jokic', 'Joel Embiid', 'Paul George', 'Kawhi Leonard', 'Damian Lillard', 'Jimmy Butler',
                'LeBron James', "Bradley Beal"]
```


```python
player_id_map = {a: find_player_id(a) for a in players_range}
```


```python
player_id_map
```




    {'Anthony Davis': [203076],
     'James Harden': [201935],
     'Stephen Curry': [201939],
     'Giannis Antetokounmpo': [203507],
     'Karl-Anthony Towns': [1626157],
     'Nikola Jokic': [203999],
     'Joel Embiid': [203954],
     'Paul George': [202331],
     'Kawhi Leonard': [202695],
     'Damian Lillard': [203081],
     'Jimmy Butler': [202710],
     'LeBron James': [2544],
     'Bradley Beal': [203078]}




```python
for player, player_id in player_id_map.items():
    compiled_log = compile_player_gamelog(player_id, seasons_range)
    compiled_log.to_csv("data/{}.csv".format(player.replace(" ","")))
    time.sleep(10)
```


```python

```

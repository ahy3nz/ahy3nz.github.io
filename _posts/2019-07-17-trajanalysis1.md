---
title: 'Molecular Modeling Software: MDTraj'
date: 2019-07-17
permalink: /posts/2019/07/trajanalysis1/
tags:
  - scientificComputing
  - molecularmodeling
---

# Some anecdotes with analyzing simulations
Let's say you've conducted a simulation. 
Everything up to that point (parametrization, initialization, actually running the simulation) will be assumed
and probably discussed another day.
What you have from a simulation is a **trajectory** (*timeseries of coordinates*), and now we have to derive some 
meaningful properties from this trajectory.

Many meaningful properties can be derived from these coordinates, be it how atomic coordinates are related to each other, the sorts of geometries or larger structures we see, or how these coordinates are correlated over time.
Whatever it is you're interested in, it all starts with the coordinates

There are many analysis packages:
* [**MDTraj**](http://mdtraj.org/1.9.3/)
* [**MDAnalysis**](https://www.mdanalysis.org/docs/)
* [**Freud**](https://freud.readthedocs.io/en/stable/)
* [**Pytraj**](https://amber-md.github.io/pytraj/latest/index.html)
* [**Cpptraj**](https://amber-md.github.io/cpptraj/CPPTRAJ.xhtml)
* and many, many others (this is what happens when a open-source software goes rampant with different desired functionality and starts from independent research groups)

While each has a variety of different built-in/common analysis routines, some are more common 
(like radial distribution functions).
What EVERY modeler will use, though, is the coordinates.
The *most important* function in these analysis packages is the ability to turn 
a large trajectory file, written to disk, and read it into memory as a data structure whose XYZ coordinates we can access. 

Every simulation engine has different file formats and data encodings, but many of these analysis packages can support a wide range of file formats and pump out the same `Trajectory` data structure core to each package.

For example, we can use **MDtraj** to read in some simulation files from GROMACS.
We obtain information about the XYZ coordinates and *molecular topology* 
(atoms, elements, atom names/types, residues, chemical bonding)

In general, there's a sort of hierarchy/classification to groups of atoms. 
At the base, you have an *atom*, which is as it sounds, or a coarse-grained particle depending on your simulation.
Groups of atoms can form a *chain*, which is pretty much just a bonded network of atoms.
Groups of atoms and chains form a *residue*. This derives from protein amino acid residues, where each monomer was a residue. In other applications, this can also refer to a closed-loop bonded network of atoms (a singular molecule).
All of these different entities/groupings form your *topology*


```python
import mdtraj
traj = mdtraj.load('trajectory.xtc', top='em.gro')
traj
```




    <mdtraj.Trajectory with 1501 frames, 18546 atoms, 2688 residues, and unitcells at 0x10dab9ba8>



Most analysis packages have some way to access each *atom* in your topology


```python
traj.topology.atom(0)
```




    DSPC1-N



If designed well, you can access *residue information* from each *atom*


```python
traj.topology.atom(0).residue
```




    DSPC1



Or, you could acess each *residue* in your topology


```python
traj.topology.residue(0)
```




    DSPC1



And then access each *atom* from within that *residue*


```python
traj.topology.residue(0).atom(2)
```




    DSPC1-H13A



Every *atom* has an *index*, which is often used for accessing the different arrays


```python
traj.topology.atom(100).index
```




    100



Some analysis packages also have an *atom-selection language*, which returns various atom indices


```python
traj.topology.select("element N")
```




    array([    0,   142,   284,   426,   568,   710,   852,   994,  1136,
            1278,  1420,  1562,  1704,  1846,  1988,  2130,  2272,  2414,
            2556,  2698,  2840,  9273,  9415,  9557,  9699,  9841,  9983,
           10125, 10267, 10409, 10551, 10693, 10835, 10977, 11119, 11261,
           11403, 11545, 11687, 11829, 11971, 12113])



Now we can get to the important numbers, the coordinates


```python
traj.xyz
```




    array([[[3.75500011e+00, 2.16800022e+00, 6.84000015e+00],
            [3.64100027e+00, 2.17200017e+00, 6.94400024e+00],
            [3.68000007e+00, 2.21600008e+00, 7.03500032e+00],
            ...,
            [1.38600004e+00, 2.20000014e-01, 8.43700027e+00],
            [1.31000006e+00, 2.01000005e-01, 8.49200058e+00],
            [1.39500010e+00, 1.43000007e-01, 8.38100052e+00]],
    
           [[3.92900014e+00, 2.18300009e+00, 6.83200026e+00],
            [3.85500026e+00, 2.24800014e+00, 6.94500017e+00],
            [3.92200017e+00, 2.28600001e+00, 7.02000046e+00],
            ...,
            [7.00000003e-02, 3.23000014e-01, 1.30000010e-01],
            [6.00000005e-03, 3.45000029e-01, 6.30000010e-02],
            [8.50000009e-02, 4.05000031e-01, 1.77000001e-01]],
    
           [[3.79200029e+00, 2.13300014e+00, 6.90600014e+00],
            [3.75500011e+00, 2.17000008e+00, 7.04500055e+00],
            [3.69800019e+00, 2.26100016e+00, 7.03900051e+00],
            ...,
            [3.31000030e-01, 8.42000067e-01, 8.24400043e+00],
            [2.39000008e-01, 8.64000022e-01, 8.26000023e+00],
            [3.51000011e-01, 7.74000049e-01, 8.30900002e+00]],
    
           ...,
    
           [[5.35700035e+00, 3.21400023e+00, 6.66500044e+00],
            [9.00000054e-03, 3.30200005e+00, 6.77700043e+00],
            [5.32400036e+00, 3.37800026e+00, 6.80000019e+00],
            ...,
            [5.89000046e-01, 2.46400023e+00, 6.28700018e+00],
            [5.42000055e-01, 2.38100004e+00, 6.27500010e+00],
            [6.04000032e-01, 2.49500012e+00, 6.19800043e+00]],
    
           [[9.20000076e-02, 3.15200019e+00, 7.07700014e+00],
            [1.93000004e-01, 3.26800013e+00, 7.08700037e+00],
            [1.33000001e-01, 3.35700011e+00, 7.09800053e+00],
            ...,
            [8.20000052e-01, 2.19400001e+00, 6.70000029e+00],
            [8.31000030e-01, 2.10200000e+00, 6.67600012e+00],
            [7.78000057e-01, 2.23400021e+00, 6.62400055e+00]],
    
           [[1.24000005e-01, 2.99600005e+00, 6.71500015e+00],
            [9.60000008e-02, 3.05700016e+00, 6.84500027e+00],
            [4.00000019e-03, 3.11200023e+00, 6.83600044e+00],
            ...,
            [5.77000022e-01, 2.19900012e+00, 6.82900047e+00],
            [6.62000060e-01, 2.23500013e+00, 6.80300045e+00],
            [5.80000043e-01, 2.10800004e+00, 6.80000019e+00]]], dtype=float32)



This is a multi-dimensional array, but off the bat you can start seeing these 3-tuples for XYZ. 

This is a `numpy array`, though, so we can use some numpy functions


```python
traj.xyz.shape
```




    (1501, 18546, 3)



1501 frames, 18546 atoms, 3 spatial coordinates. In `numpy array` terms,
the frames are the first dimension, atoms the second dimension, 
and spatial coordinates the third.

We can also snip out a frame to get all of the coordinates for all the atoms in that one frame


```python
traj.xyz[0].shape
```




    (18546, 3)



Snip out an atom (or collection of atoms) - based on index - to get all frames and all the coordinates of that collection of atoms


```python
traj.xyz[:, [1,2,3],:].shape
```




    (1501, 3, 3)



Snip out just one dimension to get all frames and all atoms and just one dimension


```python
traj.xyz[:,:,0].shape
```




    (1501, 18546)



Since a trajectory is just a collection of frames, one after another, you can also snip out frames from a trajectory


```python
traj[0]
```




    <mdtraj.Trajectory with 1 frames, 18546 atoms, 2688 residues, and unitcells at 0x1257af908>



This is still a `Trajectory` object, just 1 frame. XYZ coordinates are still accessible as earlier

All simulations occur within a unitcell to define the boundaries of the simulation. 


```python
traj.unitcell_vectors
```




    array([[[5.11195  , 0.       , 0.       ],
            [0.       , 3.74324  , 0.       ],
            [0.       , 0.       , 8.80772  ]],
    
           [[5.116633 , 0.       , 0.       ],
            [0.       , 3.7466693, 0.       ],
            [0.       , 0.       , 8.806476 ]],
    
           [[5.0943184, 0.       , 0.       ],
            [0.       , 3.7303293, 0.       ],
            [0.       , 0.       , 8.894873 ]],
    
           ...,
    
           [[5.3887806, 0.       , 0.       ],
            [0.       , 3.9459503, 0.       ],
            [0.       , 0.       , 8.189841 ]],
    
           [[5.3347273, 0.       , 0.       ],
            [0.       , 3.9063694, 0.       ],
            [0.       , 0.       , 8.352207 ]],
    
           [[5.3583217, 0.       , 0.       ],
            [0.       , 3.9236465, 0.       ],
            [0.       , 0.       , 8.363432 ]]], dtype=float32)




```python
traj.unitcell_vectors.shape
```




    (1501, 3, 3)



For each frame, there is a 3x3 array to describe the simulation box vectors

I won't go into how you should analyze a trajectory, but every molecular modeler should be familiar with what analysis routines exist in which packages, and which analysis routines you should design yourself

## Comments
There is a whole zoo of trajectory file formats that simulation engines produce - each analysis package can accommodate a subset of those file formats, each analysis package has different built-in analysis routines. Sometimes it's a mix-and-match game where you need to use package A to read a trajectory, and convert to package B representation because it has some particular analysis routine you need.

You could use an analysis package to read in one file format but write in another file format, or use an analysis package to manipulate coordinates/toplogy. Because these packages are designed intuitively and very similar to other structures in the SciPy ecosystem, there is a lot of room for creativity

Recent developments in the SciPy ecosystem look at out-of-memory or GPU representations of `numpy array` or `pandas DataFrame`, and this is a growing issue in our field - sometimes loading an entire Trajectory into memory is just not possible, so *chunking* is necessary to break the whole Trajectory into memory-manageable data

## Summary
There are a variety of analysis packages out there, but they all start out the same way: read a simulation trajectory file and create an in-memory data representation that contains trajectory (coordinates) and topology (atoms, bonds) information.



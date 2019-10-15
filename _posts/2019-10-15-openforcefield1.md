# Putting together open-source molecular modelling software

The open-source molecular modelling community is a small (but growing) trend within academics. 
A lot of academics (professors, lab scientists, grad students) are putting together libraries and API that help fulfill a small task or purpose, and 21st century software engineering standards make them usable by others.
Even in these early stages of open-source molecular modelling, these libraries are striving for interoperability, where two independently-developed API have gotten to the point where now they want to interact and communicate with each other.

This is a very interesting point in time, where molecular modellers are now tasked with the effort of making many different libraries and API work together to successfully run simulations and complete research projects. 
Usually, scientists work within a singular software package that was designed by some core developers, and those scientists didn't need to venture outside that single software package, the license they paid for it, and the manual.

With the release of [OpenForceField 1.0](https://openforcefield.org/news/introducing-openforcefield-1.0/), I was curious to use their SMRINOFF force field. 
To my understanding (don't quote me on this), the idea behind SMIRNOFF is to simplify molecular mechanics force fields, cut down on redundant atom types/parameters, and parametrize molecules based on "chemical perception" (chemical context and local bonding environment). 
Armed with these simplified, context-based force field methodologies, the frustrating, in-the-weeds obstacles associated with force field development might be ameliorated - the kinds of obstacles of which molecular modellers are painfully aware.

Beyond the SMIRNOFF force field, I was curious how parameters might compare to the older OPLS all-atom force fields. As a personal challenge, I wanted to see how much "modern computational science" I could use, specifically trying to exercise the interoperability between different open-source molecular modelling packages.

## Building the our molecular system and model
We begin with some imports. We can already see a variety of packages being used: mBuild, Foyer, ParmEd, OpenForceField, Simtk, OpenMM, MDTraj, and NGLView. 

Take note of all the different data structure interconversions happening. There are *a lot*. This is good that we can get these API working together this often, but maybe not-so-good that we have to do these interconversions so often

Note: OpenForceField also utilizes RDKit and OpenEyeToolkit. mBuild also utilizes OpenBabel


```python
# MoSDeF tools for initializing and parametrizing systems
import mbuild
from mbuild.examples import Ethane
import foyer

# ParmEd for interconverting data structures
import parmed

# Omnia suite of molecular modelling tools
from openforcefield.topology import Topology, Molecule
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm, unit

# For post-simulation analysis and visualization
import mdtraj # Also Omnia
import nglview
```

    Warning: Unable to load toolkit 'OpenEye Toolkit'. The Open Force Field Toolkit does not require the OpenEye Toolkits, and can use RDKit/AmberTools instead. However, if you have a valid license for the OpenEye Toolkits, consider installing them for faster performance and additional file format support: https://docs.eyesopen.com/toolkits/python/quickstart-python/linuxosx.html OpenEye offers free Toolkit licenses for academics: https://www.eyesopen.com/academic-licensing



    _ColormakerRegistry()


    /Users/ayang41/anaconda3/envs/mosdef37/lib/python3.7/site-packages/nglview/widget.py:162: DeprecationWarning: Traits should be given as instances, not types (for example, `Int()`, not `Int`). Passing types is deprecated in traitlets 4.1.
      _ngl_view_id = List(Unicode).tag(sync=True)


We will use mBuild to create a generic Ethane ($C_2H_6$) molecule. 
While this is imported from the examples, mBuild functionality allows users to construct chemical systems in a lego-like fashion by declaring particles and bonding them. 
Under the hood, rigid transformations are performed to orient particles-to-be-bonded


```python
mbuild_compound = Ethane()
mbuild_compound.visualize(backend='nglview')
```

    /Users/ayang41/Programs/mbuild/mbuild/utils/io.py:120: DeprecationWarning: openbabel 2.0 detected and will be dropped in a future release. Consider upgrading to 3.x.
      warnings.warn(msg, DeprecationWarning)
    /Users/ayang41/Programs/mbuild/mbuild/utils/io.py:120: DeprecationWarning: openbabel 2.0 detected and will be dropped in a future release. Consider upgrading to 3.x.
      warnings.warn(msg, DeprecationWarning)



    NGLWidget()


Another operation we can do within mBuild is to take this compound, convert it to an `openbabel.Molecule` object,
and obtain the SMILES string for it.


```python
ethane_obmol = mbuild_compound.to_pybel()
ethane_obmol.write("smi", 'out.smi', overwrite=True)
smiles_string = open('out.smi', 'r').readlines()
print(smiles_string)
```

    ['CC\tEthane\n']


    /Users/ayang41/Programs/mbuild/mbuild/utils/io.py:120: DeprecationWarning: openbabel 2.0 detected and will be dropped in a future release. Consider upgrading to 3.x.
      warnings.warn(msg, DeprecationWarning)


Using foyer, we can convert an `mbuild.Compound` object to an `openmm.Topology` object. 
`openmm.Topology` objects don't actually know positions, they just know certain atomic and bonding information, but no coordinates/velocities/force field information.
This foyer function helps recover the positions in a simple array of `simtk.Quantity`


```python
omm_topology, xyz = foyer.forcefield.generate_topology(mbuild_compound, residues='Ethane')
print(omm_topology)
print(xyz)
```

    <Topology; 1 chains, 1 residues, 8 atoms, 7 bonds>
    [Vec3(x=2.819666989525475e-16, y=-1.4, z=-1.4644271506889933e-16), Vec3(x=-1.0699999332427972, y=-1.4000000000000001, z=-6.273541601638111e-17), Vec3(x=0.3570000827312472, y=-2.169000053405761, z=0.6530000269412993), Vec3(x=0.3570000827312474, y=-1.5810000836849212, z=-0.9929999709129338), Vec3(x=0.0, y=0.0, z=0.0), Vec3(x=1.0699999332427979, y=0.0, z=0.0), Vec3(x=-0.35700008273124695, y=0.7690000534057617, z=0.6530000269412994), Vec3(x=-0.35700008273124695, y=0.18100008368492126, z=-0.9929999709129333)] A


To translate these objects into `openforcefield.Topology` objects, we need to identify the unique molecules, which helps identify the isolated subgraphs - individual molecules that don't bond to anything outside its molecular network. 

Using the SMILES string, we can generate an `openforcefield.Molecule` object, which is this self-enclosed bonding entity (chemically speaking, this is a molecule)


```python
ethane_molecule = Molecule.from_smiles(smiles_string[0].split()[0])
ethane_molecule
```




    Molecule with name '' and SMILES '[H][C]([H])([H])[C]([H])([H])[H]'



Now that we have isolated the unique molecules, we can construct our `openforcefield.Topology` object from our `openmm.Topology` and `openmm.Molecule` objects.


```python
off_topology = Topology.from_openmm(omm_topology, unique_molecules=[ethane_molecule])
off_topology
```




    <openforcefield.topology.topology.Topology at 0x113ad0748>



## Adding in a force field, evaluating energy

Next, we need to create our `openforcefield.Forcefield` object. These are created from `offxml` files, and the OpenForceField group publishes new ones fairly regularly. 

In the comments is an example (but out-of-date) force field within the [main openforcefield package](https://github.com/openforcefield/openforcefield).

The one we are using is the most-recent SMIRNOFF force field (I think this one is Parsley, or maybe just 1.0.0). 
The SMIRNOFF force fields are being housed in [a separate repo](https://github.com/openforcefield/smirnoff99Frosst), but utilize pythonic `entry_points` to help one repo into another.


```python
#off_forcefield = ForceField('test_forcefields/smirnoff99Frosst.offxml')
off_forcefield = ForceField('smirnoff99Frosst-1.1.0.offxml')
off_forcefield
```




    <openforcefield.typing.engines.smirnoff.forcefield.ForceField at 0x114dd72e8>



With the `openforcefield.Topology` and `openforcefield.Forcefield` objects, we can create an `openmm.System`. 
Note the discrepancy/interplay between the objects - `openforcefield` for the molecular mechanics building blocks, but `openmm` is ultimately the workhorse for simulating and representing these systems (although you could opt to simulate with other engines via `parmed`).

Note the use of AM1-BCC methods to identify partial charges.


```python
smirnoff_omm_system = off_forcefield.create_openmm_system(off_topology)
smirnoff_omm_system
```

    Warning: In AmberToolsToolkitwrapper.compute_partial_charges_am1bcc: Molecule '' has more than one conformer, but this function will only generate charges for the first one.





    <simtk.openmm.openmm.System; proxy of <Swig Object of type 'OpenMM::System *' at 0x111a4fde0> >



This is a utility function we will use to evaluate the energy of a molecular system.
Given an `openmm.system` (force field, parameters, topological, atomic information) and atomic coordinates, we can get a potential energy associated with that set of coordinatess


```python
def get_energy(system, positions):
    """
    Return the potential energy.

    Parameters
    ----------
    system : simtk.openmm.System
        The system to check
    positions : simtk.unit.Quantity of dimension (natoms,3) with units of length
        The positions to use
    Returns
    ---------
    energy
    
    Notes
    -----
    Taken from an openforcefield notebook
    """

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().in_units_of(unit.kilocalories_per_mole)
    return energy
```

Next, we will try to calculate the potential energy of our ethane system under the SMIRNOFF force field.
As a (small) obstacle to doing this, we need to change the dimensions of our simulation box because some force fields and simulations use cutoffs, and cutoffs cannot be larger than the simulation box itself.

Okay 45 kcal/mol, cool. Potential energies of single configurations are usually not helpful for any real physical analysis, but can be helpful in comparing force fields.


```python
new_vectors = [[10*unit.nanometer, 0*unit.nanometer, 0*unit.nanometer], 
               [0*unit.nanometer, 10*unit.nanometer, 0*unit.nanometer],
               [0*unit.nanometer, 0* unit.nanometer, 10*unit.nanometer]]
smirnoff_omm_system.setDefaultPeriodicBoxVectors(*new_vectors)
get_energy(smirnoff_omm_system, xyz)
```




    Quantity(value=44.96860291382222, unit=kilocalorie/mole)



## Tangent: Interfacing with other simulation engines
We can use `parmed` to convert the `openmm.Topology`, `openmm.System`, and coordinates into a `parmed.Structure`.
From a `parmed.Structure`, we can spit out files appropriate for different simulation packages. 
Word of caution, while the developers of `parmed` did an excellent job building the conversion tools, please do your due diligence to make sure the output is as you expect


```python
pmd_structure = parmed.openmm.load_topology(omm_topology, system=smirnoff_omm_system, xyz=xyz)
pmd_structure
```




    <Structure 8 atoms; 1 residues; 7 bonds; PBC (orthogonal); NOT parametrized>



## Comparing to the OPLS-AA force field
Let's use a different force field. 
`foyer` ships with an XML of the OPLS-AA force field.
We will use `foyer` (which utilizes some `parmed` and `openmm` api) to build our molecular model of ethane with OPLS

* Create the `foyer.Forcefield` object
* Apply it to our `mbuild.Compound`, get a `parmed.Structure`
* Convert the `parmed.Structure` to an `openmm.System`
* Reset the box vectors to be consistent with the SMIRNOFF example
* Evaluate the energy


```python
foyer_ff = foyer.Forcefield(name='oplsaa')
opls_pmd_structure = foyer_ff.apply(mbuild_compound)
opls_omm_system = opls_pmd_structure.createSystem()
opls_omm_system.setDefaultPeriodicBoxVectors(*new_vectors)
get_energy(opls_omm_system, opls_pmd_structure.positions)
```

    /Users/ayang41/Programs/foyer/foyer/validator.py:132: ValidationWarning: You have empty smart definition(s)
      warn("You have empty smart definition(s)", ValidationWarning)
    /Users/ayang41/Programs/foyer/foyer/forcefield.py:248: UserWarning: Parameters have not been assigned to all impropers. Total system impropers: 8, Parameterized impropers: 0. Note that if your system contains torsions of Ryckaert-Bellemans functional form, all of these torsions are processed as propers
      warnings.warn(msg)





    Quantity(value=37.52734328319192, unit=kilocalorie/mole)



37.5 kcal/mol versus 45.0 kcal/mol, for a *single* ethane. This is a little alarming because this energetic difference stems from how the interactions are quantified and parametrized. 

However, this isn't a deal-breaker since most physically-interesting phenomena depend on changes in (free) energies. So a singular energy isn't important - how it varies when you change configurations or sample configurational space is generally more important

## Cracking open the models and looking at parameters

In this whole process, we've been dealing with data structures and API that are fairly transparent. 
What's great now is that we can look in-depth at these data structures. 
Specifically, we could look at the force field parameters, either within the force field files or molecular models themselves.

We're going to crack open these `openmm.System` objects, looking at how some of these forces are parametrized

Within the SMIRNOFF force field applied to ethane, we have some harmonic bonds, harmonic angles, periodic torsions, and nonbonded forces


```python
smirnoff_omm_system.getForces()
```




    [<simtk.openmm.openmm.NonbondedForce; proxy of <Swig Object of type 'OpenMM::NonbondedForce *' at 0x116245960> >,
     <simtk.openmm.openmm.PeriodicTorsionForce; proxy of <Swig Object of type 'OpenMM::PeriodicTorsionForce *' at 0x116245a50> >,
     <simtk.openmm.openmm.HarmonicAngleForce; proxy of <Swig Object of type 'OpenMM::HarmonicAngleForce *' at 0x116245a80> >,
     <simtk.openmm.openmm.HarmonicBondForce; proxy of <Swig Object of type 'OpenMM::HarmonicBondForce *' at 0x116245ab0> >]



Within the OPLS force field applied to ethane, we have some harmonic bonds, harmonic angles, Ryckaert-Belleman torsions, and nonbonded forces. 
Don't worry about the center of mass motion remover - that's more for running a simulation.


```python
opls_omm_system.getForces()
```




    [<simtk.openmm.openmm.HarmonicBondForce; proxy of <Swig Object of type 'OpenMM::HarmonicBondForce *' at 0x116245930> >,
     <simtk.openmm.openmm.HarmonicAngleForce; proxy of <Swig Object of type 'OpenMM::HarmonicAngleForce *' at 0x116245b40> >,
     <simtk.openmm.openmm.RBTorsionForce; proxy of <Swig Object of type 'OpenMM::RBTorsionForce *' at 0x116245b70> >,
     <simtk.openmm.openmm.NonbondedForce; proxy of <Swig Object of type 'OpenMM::NonbondedForce *' at 0x116245ba0> >,
     <simtk.openmm.openmm.CMMotionRemover; proxy of <Swig Object of type 'OpenMM::CMMotionRemover *' at 0x116245bd0> >]



We are going to compare the *nonbonded parameters* between these `openmm.System` objects. 
For every particle in our system, we're going to look at their charges, LJ sigmas, and LJ epsilsons (both of these systems utilize Coulombic electrostatics and Lennard-Jones potentials)

Based on the charges and frequency-of-appearance, we can see which ones are carbons and which ones are hydrogens. 

The OPLS-system is more-charged, carbons are more negative and hydrogens are more positive. 
The SMIRNOFF-system actually isn't electro-neutral, and that might be consequence of having used AM1-BCC for such a small system.

The sigmas are pretty similar between FF implementations. The hydrogen epsilsons in SMIRNOFF are about half of those in OPLS. The carbon epsilons in SMIRNOFF are almost double those in OPLS. This is kind of interesting, while SMIRNOFF-ethane has weaker electrostatics (weaker charges), the LJ might compensate with the greater carbon-epsilon.


```python
opls_omm_nonbond_force = opls_omm_system.getForce(3)
smirnoff_omm_nonbond_force = smirnoff_omm_system.getForce(0)
for i in range(opls_omm_nonbond_force.getNumParticles()):
    opls_params = opls_omm_nonbond_force.getParticleParameters(i)
    smirnoff_params = smirnoff_omm_nonbond_force.getParticleParameters(i)    
    print(opls_params)
    print(smirnoff_params)
    print('---')
```

    [Quantity(value=-0.18, unit=elementary charge), Quantity(value=0.35000000000000003, unit=nanometer), Quantity(value=0.276144, unit=kilojoule/mole)]
    [Quantity(value=-0.0941, unit=elementary charge), Quantity(value=0.3399669508423535, unit=nanometer), Quantity(value=0.4577296, unit=kilojoule/mole)]
    ---
    [Quantity(value=0.06, unit=elementary charge), Quantity(value=0.25, unit=nanometer), Quantity(value=0.12552, unit=kilojoule/mole)]
    [Quantity(value=0.0317, unit=elementary charge), Quantity(value=0.2649532787749369, unit=nanometer), Quantity(value=0.06568879999999999, unit=kilojoule/mole)]
    ---
    [Quantity(value=0.06, unit=elementary charge), Quantity(value=0.25, unit=nanometer), Quantity(value=0.12552, unit=kilojoule/mole)]
    [Quantity(value=0.0317, unit=elementary charge), Quantity(value=0.2649532787749369, unit=nanometer), Quantity(value=0.06568879999999999, unit=kilojoule/mole)]
    ---
    [Quantity(value=0.06, unit=elementary charge), Quantity(value=0.25, unit=nanometer), Quantity(value=0.12552, unit=kilojoule/mole)]
    [Quantity(value=0.0317, unit=elementary charge), Quantity(value=0.2649532787749369, unit=nanometer), Quantity(value=0.06568879999999999, unit=kilojoule/mole)]
    ---
    [Quantity(value=-0.18, unit=elementary charge), Quantity(value=0.35000000000000003, unit=nanometer), Quantity(value=0.276144, unit=kilojoule/mole)]
    [Quantity(value=-0.0941, unit=elementary charge), Quantity(value=0.3399669508423535, unit=nanometer), Quantity(value=0.4577296, unit=kilojoule/mole)]
    ---
    [Quantity(value=0.06, unit=elementary charge), Quantity(value=0.25, unit=nanometer), Quantity(value=0.12552, unit=kilojoule/mole)]
    [Quantity(value=0.0317, unit=elementary charge), Quantity(value=0.2649532787749369, unit=nanometer), Quantity(value=0.06568879999999999, unit=kilojoule/mole)]
    ---
    [Quantity(value=0.06, unit=elementary charge), Quantity(value=0.25, unit=nanometer), Quantity(value=0.12552, unit=kilojoule/mole)]
    [Quantity(value=0.0317, unit=elementary charge), Quantity(value=0.2649532787749369, unit=nanometer), Quantity(value=0.06568879999999999, unit=kilojoule/mole)]
    ---
    [Quantity(value=0.06, unit=elementary charge), Quantity(value=0.25, unit=nanometer), Quantity(value=0.12552, unit=kilojoule/mole)]
    [Quantity(value=0.0317, unit=elementary charge), Quantity(value=0.2649532787749369, unit=nanometer), Quantity(value=0.06568879999999999, unit=kilojoule/mole)]
    ---


## Running some molecular dynamics simulations
We've come this far in building our model with different force fields, we might as well build up the rest of the simulation.

`openmm` will be used to run our simulation, since we already have an `openmm.System` object. 
We need an integrator that describes our equations of motion, timestep, and temperature behavior.

As a side note, we forcibly made our simulation box really big to address cutoffs, but we can probably go with a smaller box that still fits the bill. The smaller box helps speed up the computation.


```python
integrator = openmm.LangevinIntegrator(323 * unit.kelvin, 1.0/unit.picoseconds, 0.001 * unit.picoseconds)
smallbox_vectors = [[2*unit.nanometer, 0*unit.nanometer, 0*unit.nanometer], 
               [0*unit.nanometer, 2*unit.nanometer, 0*unit.nanometer],
               [0*unit.nanometer, 0* unit.nanometer, 2*unit.nanometer]]
smirnoff_omm_system.setDefaultPeriodicBoxVectors(*smallbox_vectors)
```

We combine our `openmm.Topology`, `openmm.System`, and `openmm.Integrator` to make our `openmm.Simulation`, then set the positions


```python
smirnoff_simulation = openmm.app.Simulation(omm_topology, smirnoff_omm_system, integrator)
smirnoff_simulation.context.setPositions(xyz)
```

Before running the simulation, we need to report some information.
Otherwise, the simulation's going to run and we won't have anything to show for it.
This is handled in `openmm` by creating `openmmm.reporters` and attaching them to your `openmm.Simulation`
We will write out the timeseries of coordinates (trajectory) in a `dcd` format, 
but also a `pdb` format to show a singular configuration. 
In this case, we're printing a `pdb` file that corresponds to the first configuration, before any simulation was run.


```python
smirnoff_simulation.reporters.append(openmm.app.DCDReporter('trajectory.dcd', 10))
pdbreporter = openmm.app.PDBReporter('first_frame.pdb', 5000)
pdbreporter.report(smirnoff_simulation, smirnoff_simulation.context.getState(-1))
```

Now we can run our simulation!


```python
smirnoff_simulation.step(1000)
```

After it's finished, we can load the trajectory files into an `mdtraj.Trajectory` object, and visualize in a jupyter notebook with `nglview`. From this `mdtraj.Trajectory` object, you have pythonic-access to all the coordinates over time, and also access to various analysis libraries within `mdtraj`

The ethane is jumping around the boundaries of the periodic box, but you can see it wiggling. 
Not super interesting, but simulations from open-source software are doable. 
If I had a more powerful computer, maybe I'd try a larger system, but I'll leave it to others to build off of my notebook (you can find this in my website's [git repo](https://github.com/ahy3nz/ahy3nz.github.io/tree/master/files/notebooks))


```python
traj = mdtraj.load('trajectory.dcd', top='first_frame.pdb')
nglview.show_mdtraj(traj)
```


    NGLWidget(max_frame=109)


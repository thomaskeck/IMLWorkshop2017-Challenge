<a class="mk-toclify" id="table-of-contents"></a>

# IML Workshop - Mini Challenge on q/g jet tagging
- [Introduction](#introduction)
- [Challenge Rules](#challenge-rules)
- [The simulation setup](#the-simulation-setup)
    - [Physics process and cuts](#physics-process-and-cuts)
    - [Detector setup](#detector-setup)
        - [Delphes FCC setup](#delphes-fcc-setup)
        - [Jet reconstruction](#jet-reconstruction)
- [The data format](#the-data-format)
    - [Location of the data sets in eos](#location-of-the-data-sets-in-eos)
- [How to develop code for the challenge and examples](#how-to-develop-code-for-the-challenge-and-examples)
    - [Swan and Jupyter notebooks quick start](#swan-and-jupyter-notebooks-quick-start)
    - [The example and QA notebooks](#the-example-and-qa-notebooks)
        - [Basic QA Macro and Notebook (C++)](#basic-qa-macro-and-notebook-c)
        - [Python example: sklearn Naive Bayes](#python-example-sklearn-naive-bayes)
        - [Python example: Keras](#python-example-keras)
        - [ROOT TMVA example](#root-tmva-example)
    - [How to run on lxplus](#how-to-run-on-lxplus)
        - [for the "bleeding edge" version of swan](#for-the-bleeding-edge-version-of-swan)
        - [for the LCG 88 version of swan](#for-the-lcg-88-version-of-swan)
        - [common for all LCG versions](#common-for-all-lcg-versions)
    - [How to submit your results](#how-to-submit-your-results)
- [Chat room](#chat-room)
- [Acknowledgements](#acknowledgements)
- [FAQ and Known Issues](#faq-and-known-issues)


<a class="mk-toclify" id="introduction"></a>
# Introduction

The purpose of this challenge is to develop and test methods to distinguish between quark and gluon jets.

The data sample has been generated simulating with Pythia 8 a Randal-Sundrum graviton, decaying either to a quark or to a gluon pair. 

The data have been then filtered using [Delphes](https://cp3.irmp.ucl.ac.be/projects/delphes). We only provide detector-level information for this challenge.

In order to simulate a more realistic situation (where a Monte Carlo simulation is never perfect), we provide 2 datasets, generated with different versions of Pythia:   Pythia 8.223 for the “standard" samples and 8.180 for the “modified" samples. This leads to a slightly different jet kinematics and substructure. 
The datasets are labelled as "Standard" and "Modified" below.

The training of your method should be based solely on the *Standard* data set, while the score which will be used for the challenge ranking should be based on the *Modified* one.
The idea is that in real-life you would train your method on Monte Carlo, and then apply it to real data (and in general there are small differences between data and MC).
Techniques to minimize the effect of these difference are therefore very welcome.

We realize that the few hours allocated for this challenge are not enough to develop any completely new method. At the same time, we believe that this is an excellent opportunity to compare apples-to-apples methods that you may have been already thinking about, or that are in use in your experiments. This is also the first attempt to create a IML benchmark dataset. Therefore, it will be very important to receive feedback on all aspects of the dataset (simulated process, data format, distribution of the data, examples,... )

Finally, we also believe this also represents an excellent playground for people who are relatively new to machine learning. For instance, people who have just followed the tutorial can test they newly acquired skills on this data.

The provided recipes are intentionally very basic, as their main goal is to show how to access the data and provide a quick start for people new to machine learning.

We hope that you will enjoy this mini challenge!

<a class="mk-toclify" id="challenge-rules"></a>
# Challenge Rules

* The challenge starts during the opening sessions of the IML workshop (Monday March 20, 9:00), and finishes with the last coffee break (Wednsday March 22, 16:00)
* Results should be submitted as merge requests on gitlab, the details are discussed in [How to submit your results](#how-to-submit-your-results)
* Working code should be provided for the result to be considered
* It is allowed to form teams
* You can use any external computing resources you may have at your disposal
* The ranking will be based on the area under the ROC curve (AUC), estimated on the modified sample for a model trained on the standard sample. 
* The modified sample data can be used in the training **as long as the true labels of the modified sample are not used in the training**

<a class="mk-toclify" id="the-simulation-setup"></a>
# The simulation setup

<a class="mk-toclify" id="physics-process-and-cuts"></a>
## Physics process and cuts

The events contain Randall-Sundrum gravitons, inclusively produced from proton-proton collisions at sqrt(s)=13 TeV. The graviton mass is set to 200 GeV, while the kappa_G parameter is set to 0.00001, which results in a negligible width. The graviton is forced to decay to a pair of gluons or light quarks (up, down, or strange). As a reference, the Pythia setting follows for the gg case:

```
ExtraDimensionsG*:all  = on 
5100039:m0 = 200.  ! Mass (in GeV)
ExtraDimensionsG*:kappaMG = 0.00001 !TO SET WIDTH:  kappa m_G* = sqrt(2) x_1 k / Mbar_Pl, where x_1 = 3.83 
5100039:onMode = off  
5100039:onIfAny = 21  
```

and for the qq case:

```
ExtraDimensionsG*:all  = on 
5100039:m0 = 200.  ! Mass (in GeV)
ExtraDimensionsG*:kappaMG = 0.00001 !TO SET WIDTH:  kappa m_G* = sqrt(2) x_1 k / Mbar_Pl, where x_1 = 3.83 
5100039:onMode = off  ! switch off all decays
5100039:onIfAny = 1 2 3 ! switch on the decays to qq (q=u,d,s,c,b)
```

The pT distributions of the quark and gluon jets produced in this process are quite different. 
The classification algorithm should not use directly this difference, as this would defeat the spirit of the challenge. To limit the possibility to leverage on the different pt distribution, only reconstructed jets with pT between 100 and 130 GeV have been selected. We only provide jets which are originating from the initial graviton decay (jets from initial and final state radiation are not included in the sample).



<a class="mk-toclify" id="detector-setup"></a>
## Detector setup

<a class="mk-toclify" id="delphes-fcc-setup"></a>
### Delphes FCC setup 

The Delphes implementation of a possible FCC detector has been used for the simulation.
We used version 3.4.0.
The detector covers |eta| < 6 and includes a tracker and two calorimeters (hadronic and electromagnetic, with different granularities). The efficiency and the calorimeter granularity depend on eta. For more details, see the delphes card: [card](https://github.com/delphes/delphes/blob/3.4.0/cards/FCC/FCChh.tcl) or this [presentation](https://indico.cern.ch/event/550509/contributions/2413234/attachments/1395960/2128279/fccphysics_week_v4.pdf).



<a class="mk-toclify" id="jet-reconstruction"></a>
### Jet reconstruction ###

Jets have been reconstructed using fastjet, with the anti-kt algorithm and a R parameter of 0.4.
The particle-flow algorithm was used, therefore the constituents of the jet are both "tracks" and (calorimeter) "towers". 
As discussed above, there are an electromagnetic and hadronic calorimeters available. The data format (see below) allows you to access the EM and hadronic energy separately. 

<a class="mk-toclify" id="the-data-format"></a>
# The data format

The data are provided as a simple root tree format, with no external dependencies.
Each entry of the tree is a jet. The tree contains a few jet-level variables and an array of constituents (tracks and towers).

The examples discussed below show how to read this tree format using either pyroot or ROOT/C++.

This is the definition of the tree:
```C++
treeOut->Branch("jetPt"  ,&jetPt  , "jetPt/F");
treeOut->Branch("jetEta" ,&jetEta , "jetEta/F");
treeOut->Branch("jetPhi" ,&jetPhi , "jetPhi/F");
treeOut->Branch("jetMass",&jetMass, "jetMass/F");

treeOut->Branch("ntracks",&ntracks,"ntracks/I");
treeOut->Branch("ntowers",&ntowers,"ntowers/I");

treeOut->Branch("trackPt"     , trackPt    ,"trackPt[ntracks]/F");
treeOut->Branch("trackEta"    , trackEta   ,"trackEta[ntracks]/F");
treeOut->Branch("trackPhi"    , trackPhi   ,"trackPhi[ntracks]/F");
treeOut->Branch("trackCharge" , trackCharge,"trackCharge[ntracks]/F");
treeOut->Branch("towerE"      , towerE     ,"towerE[ntowers]/F");
treeOut->Branch("towerEem"    , towerEem   ,"towerEem[ntowers]/F");
treeOut->Branch("towerEhad"   , towerEhad  ,"towerEhad[ntowers]/F");
treeOut->Branch("towerEta"    , towerEta   ,"towerEta[ntowers]/F");
treeOut->Branch("towerPhi"    , towerPhi   ,"towerPhi[ntowers]/F");
```


<a class="mk-toclify" id="location-of-the-data-sets-in-eos"></a>
## Location of the data sets in eos

The datasets are available in [eos](http://information-technology.web.cern.ch/services/eos-service), in the following folder: ```/eos/project/i/iml/IMLChallengeQG/```, and are therefore accessible from both [swan](http://swan.cern.ch) and lxplus, directly from the filesystem. For reference, the actual folders are listed below:

```
/eos/project/i/iml/IMLChallengeQG/quarks_standard
/eos/project/i/iml/IMLChallengeQG/gluons_standard

/eos/project/i/iml/IMLChallengeQG/quarks_modified
/eos/project/i/iml/IMLChallengeQG/gluons_modified
```

Alternatively, data can be accessed through xrootd ```root://eosuser.cern.ch//eos/project/i/iml/IMLChallengeQG```. (Requires root with xrootd, a cern kerberos ticket and the IML e-group membership, see below)

**Important**: in order to get access to the data set you need to have subscribed the IML mailing list (click [here](https://simba3.web.cern.ch/simba3/SelfSubscription.aspx?groupName=lhc-machinelearning-wg) to subscribe)

The training of your method should be based solely on the Standard data set, while the score which will be used for the challenge ranking should be based on the Modified one. The idea is that in real-life you would train your method on Monte Carlo, and then apply it to real data (and in general there are small differences between data and MC). Techniques to minimize the effect of these difference are therefore very welcome.

<a class="mk-toclify" id="how-to-develop-code-for-the-challenge-and-examples"></a>
# How to develop code for the challenge and examples

We provide a few examples (in ROOT/C++ and python) which show you how to get started, how to access the data, and how to prepare a submissions to be considered for the purpose of the challenge.

As mentioned above,  datasets are accessible both from [swan](http://swan.cern.ch) and lxplus.

The methods which we implemented in the examples are simple and not computationally expensive. **Important**: expensive methods should not be run on swan. If you implement a computationally intensive method, you should run it on the batch system in lxplus (see below) or on other computing resources you may have at your disposal.

Our recommended mode of operation is:

* Test your developments interactively on swan;
* If the method is expensive, run the training on the full sample on lxplus or your own private resources.

In order to prepare for the challenge, we recommend that you:

1. fork this repository on gitlab (this will also be needed for the submission of your results, see the discussion below) [link to fork](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/forks/new);
2. clone your repository on your cernbox folder, e.g. on lxplus (this will make the files available on swan and on lxplus):
```
cd /eos/user/<initial>/<username>
git clone <URL of your forked repository>
```
This procedure is also a prerequisite for the submission of your results (See  [How to submit your results](#how-to-submit-your-results) below).


<a class="mk-toclify" id="swan-and-jupyter-notebooks-quick-start"></a>
## Swan and Jupyter notebooks quick start ##

If you never used jupyter notebooks, you can find some quickstart information at the following links [What is a jupyter notebook?](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb), [Notebook Basics](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb),  [Running Code](http://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Running%20Code.ipynb).

For information on swan, see the swan website: https://swan.web.cern.ch
In a nutshell, "SWAN (Service for Web based ANalysis) is a platform to perform interactive data analysis in the cloud". It gives you access to the LCG software stacks and to your CernBox files. 

You need to have an active CernBox to use swan (https://cernbox.cern.ch/)



If you think think the jupyter notebook is stuck, open a terminal on swan and execute `top`: if the cpu is not being used (0%) by any of your processes, you may have to restart the jupyter kernel. There is a known issue when executing `import ROOT` in a python notebook, which may get stuck.

**IMPORTANT:**    
* Please only have one notebook at the time on swan (you have to select the "running" tab and shutdown the notebook when done)
* Please avoid Chrome, it has known problems restarting kernels.
* Limit the number of events you use on swan (order 10000-100000) and the avoid expensive methods: *swan is meant for fast prototyping*. 
     * Each swan container has 2 GB of ram assigned, using the full dataset may hit the memory limit.

<a class="mk-toclify" id="the-example-and-qa-notebooks"></a>
## The example and QA notebooks ##

The discussion below contains links to jupyter notebooks on the gitlab repository. Unfortunately, gitlab does not properly preview jupyter notebooks, so you will see the (hugly) raw json format. **We recommend to fork & clone** the repository as explained [above](#how-to-develop-code-for-the-challenge-and-examples) and **open the notebooks directly on [swan](https://swan.cern.ch)**.

<a class="mk-toclify" id="basic-qa-macro-and-notebook-c"></a>
### Basic QA Macro and Notebook (C++) ###

In the [examples folder](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/tree/master/Examples) you find a macro called [FilteredTreesQA.C](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/blob/master/Examples/FilteredTreesQA.C) which shows how to read events from the quarks and gluons tree, and plots some basic distributions of jets and constituents as a simple QA. See the comment at the top of the macro, or have a look to the [TreeQA.pynb](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/blob/master/Examples/TreeQA.ipynb) notebook for an example on how to use it.

<a class="mk-toclify" id="python-example-sklearn-naive-bayes"></a>
### Python example: sklearn Naive Bayes ###

The first training example uses a very simple naive (Gaussian) Bayesian method.

The classification is based on 4 jet shapes (high level features): mass, radial moment, the number of towers and the dispersion. For this simple example, only tracks are used to compute the radial moment and the dispersion.

The notebook also shows how to compute the area under the ROC curve (AUC) and save the corresponding value.

You can find the notebook [here](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/blob/master/Examples/imlcoordinators_v0_NaiveBayes_main.ipynb) 
(Again, we suggest that you fork the project, clone it on your cernbox space and open the notebook on swan).


<a class="mk-toclify" id="python-example-keras"></a>
### Python example: Keras ###

The Keras example is very similar to the naive Bayes one. It is based on the same jet shapes, but a fully-connected feed-forward network is used for the classification.
The notebook is available [here](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/blob/master/Examples/imlcoordinators_v0_NaiveBayes_main.ipynb)

<a class="mk-toclify" id="root-tmva-example"></a>
### ROOT TMVA example ###

The TMVA example is divided in various files, under this [folder](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/tree/master/Examples/TMVA).

The notebook [`TMVAClassification.ipynb`](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/tree/master/Examples/TMVA/TMVAClassification.ipynb) implements an example workflow for running MVA methods shipped with ROOT/TMVA. The method is then applied on the "modified" sample in the [TMVAMeasureAUC.ipynb](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/blob/master/Examples/TMVA/TMVAMeasureAUC.ipynb)  notebook.

It is shipped with a couple of preprocessing script `PREPROCESS_DATA` and `PREPROCESS_DATA_MODIFIED`, which do a simple zero-padding strategy on the varying number of tracks and towers per event, so that we have a fix number of input variables in the preprocessed data. The comments in the notebooks will guide you through the process.

<a class="mk-toclify" id="how-to-run-on-lxplus"></a>
## How to run on lxplus ##

Your cernbox files are accessible on lxplus, in the folder ```/eos/user/<initial>/<username>```

In order to enable the same software stacks which are available on swan, you can use a command similar to the following one on lxplus (change the path if you want to use a different software stack):

**for the "bleeding edge" version of swan**

```shell
source /cvmfs/sft.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc49-opt/setup.sh. 
```

**for the LCG 88 version of swan**

```shell
source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh
```

### Converting a jupyter notebook to a python script and running on lxplus

(common for all LCG versions)
You can convert a jupyter notebook to a scipt semi-authomatically, and run the obtained script on lxplus, either interactively or on the batch system.

**general setup**
* There is a conversion script [in the repository](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/tree/master/Examples/notebook_to_code.sh).
```shell
./notebook_to_code.sh some_notebook.ipynb output_python_script.py
```
* An example output python script can be found [here](https://gitlab.cern.ch/mfloris/iml-qg-jet-challenge/blob/master/PythonCode/NaiveBayes-Python2.py)

**batch system**

* after converting jupyter notebooks, if you use `matplotlib` you should still add
```python
import matplotlib
matplotlib.use('Agg')
```
to run on batch systems [see stackoverflow](http://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined)
* Test the script
```shell
python python_script.py
```
* Submit to the batch system with [this script](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/tree/master/Examples/batchsubmit.sh) `./batchsubmit.sh python_script.py` or run the following manuall (modify to your needs)
```shell
bsub -q 8nm -J "YourJobName" << EOF
source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh
cd /path/to/your/python/script
python python_script.py
EOF
```
The necessary changes are
* the [batch queue](https://cern.service-now.com/service-portal/article.do?n=KB0000470&s=lxbatch)
* the job name: after `-J` you can assign a name to your job to identify it with `bjobs`
* adjust the environment setup (if you want to use the bleeding edge version, replace the `source /cvmfs/...` appropriately)
* and of cause adjust the path to your script and its filename
* using screen
   * if you want to run interactively, you may consider using screen  
   * screen allows you to “detach” a session, let it continue in the background and “reattach” it whenever and wherever you want. For a quick intro see [here](https://dberzano.github.io/alice/vcaf/usersguide/#creating_a_new_screen)
   * screen on lxplus can quickly lead to problems when detaching the screen session, as the afs token gets lost immediately.
     One can start a longer lasting screen session with `pagsh.krb -c 'kinit && screen'`.

<a class="mk-toclify" id="how-to-submit-your-results"></a>
## How to submit your results ##

* Fork this repository and commit and push to your fork, in the subfolder "results" (see above on how to fork the repository)
* You should commit at least 2 files: your code and a text file, containing only one number (the area under the curve, which will be used for the ranking).
* At least one of the code file names should contain the word "main". The preferred format for the main script is a jupyter notebook.
* The main script/notebook should contain a verbose comment explaining the basic idea of the methods and how to run the code.
* The following naming convention should be used for the files: ```author_version_filename```, for instance:
```
foobar_v0_auc.txt
foobar_v0_main.ipynb
foobar_v0_supportfiles.py
```
* When you are ready to submit, open pull requests on git lab, to get your code merged on the main repository
* We will test the methods before making the final ranking

<a class="mk-toclify" id="chat-room"></a>
# Chat room
A mattermost team "IML" on mattermost.web.cern.ch has been created for the workshop. 
This allows participants to ask questions, whose answers then remain visible to everybody (it will help to build a small challenge "knoledge" base as the challenge progresses).
Anybody who is logged in on gitlab.cern.ch is able to join. Alternatively the invitation link is https://mattermost.web.cern.ch/signup_user_complete/?id=7na7b9nykt8x5d7m1oyfko5e1w.
Here is a direct link to the chat room reserved for the challenge: https://mattermost.web.cern.ch/iml/channels/iml-challenge  
(Please note that the direct link may not work if this is the first time you connect to mattermost).


<a class="mk-toclify" id="acknowledgements"></a>
# Acknowledgements #


We would like to thank Maurizio Pierini, Danilo Piparo, Enric Tejedor Saavedra, Rudiger Haake, Leticia Cunqueiro, Stefan Wunsch for their help in the preparation of the challenge and testing of the examples.

<a class="mk-toclify" id="faq-and-known-issues"></a>
# FAQ and Known Issues 

See [here](https://gitlab.cern.ch/IML-WG/IMLWorkshop2017-Challenge/blob/master/KnownIssues.md)

<!--  LocalWords:  graviton Delphes hadronic gluon
-->

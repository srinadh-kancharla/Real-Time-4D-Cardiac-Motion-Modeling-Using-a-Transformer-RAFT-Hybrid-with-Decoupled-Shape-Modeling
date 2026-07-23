# Page 1

RESEARCH PLAN\
Real-Time 4D Cardiac Motion Modeling Using a Transformer-RAFT Hybrid
with Decoupled Shape Modeling From Problem Understanding → Architecture
→ Training → Working Model → Publication

# Page 2

1.  Understanding the Research Problem\
    Before writing a single line of code or mathematics, we must be
    absolutely clear about what problem we are\
    solving,

why

it

matters,

and

why

it

remains

unsolved.

1.1 What Is the Core Problem? The human heart is a continuously moving
3D organ. When a cardiologist wants to understand how the heart\
is

functioning

---

whether

it

is

contracting

normally,

where

muscle

has

died

from

a

heart

attack,

whether

the

walls

are

thinning

---

they

need

to

see

both

the

shape

of

the

heart

and

how

it

moves

through

time. Clinical cardiac MRI (cine MRI) produces stacks of 2D slices
captured at different moments in the cardiac\
cycle.

This

is

the

primary

input

data.

The

problem

is

that: • The data is sparse: only 6--12 slices cover the whole heart,
with large gaps between them. • The data is 2D: each slice is a flat
image, not a true 3D capture. • Motion is present: the heart moves
between breath-holds and between slices. • The goal is 4D: we want a
continuous 3D model of the heart at every time point in the cardiac
cycle. The central challenge is therefore: how do we reconstruct a
high-quality, temporally accurate, anatomically\
precise

4D

model

of

the

myocardium

from

inherently

incomplete

2D+time

MRI

data? THE CORE RESEARCH QUESTION Given sparse, noisy 2D cine MRI slices
across time, how can we reconstruct a complete, accurate, real-time 4D
model\
of the myocardium by separately modeling its static anatomy (shape) and
its dynamic deformation (motion) ---\
and can this be achieved with sufficient accuracy for clinical use?\
1.2 Why Does This Problem Matter Clinically?\
Clinical Measurement What We Need to Compute It Ejection Fraction (EF)
3D volume at end-diastole and end-systole\
Myocardial Strain Dense motion field across the full cardiac cycle\
Wall Thickness / Thickening 3D shape at multiple time points\
Infarct Detection Precise geometry of damaged vs healthy tissue\
Late Mechanical Activation Time-resolved regional motion for CRT
planning\
Cardiac Digital Twin Full 4D model with biomechanical properties All of
these require a full 4D myocardium model --- something that current
clinical tools cannot produce\
reliably

from

routine

MRI

data.

# Page 3

1.3 Formal Problem Statement\
MATHEMATICAL PROBLEM STATEMENT Given: A set of 2D cine MRI slices
{I_s,t} where s = slice index, t = time frame\
Goal: Reconstruct Heart(x, y, z, t) = the 3D position of every
myocardial point at every time\
Constraint 1: Shape fidelity --- the anatomy must match patient-specific
MRI observations\
Constraint 2: Motion smoothness --- deformations must be physiologically
plausible\
Constraint 3: Incompressibility --- myocardial tissue is nearly
incompressible (volume preserved)\
Constraint 4: Real-time --- inference must be fast enough for clinical
use (\<1 second per patient)

# Page 4

2.  Introduction & Background\
    2.1 The Decoupling Insight The key intellectual insight driving this
    research is the idea of decoupling. A naive deep learning model
    might\
    try

to

learn

Heart(x,

y,

z,

t)  

all

at

once

---

a

single

network

that

ingests

2D

slices

and

outputs

a

4D

model.

This

fails

for

several

reasons: • It conflates two very different types of information ---
anatomy and dynamics. • It makes the learning problem much harder than
necessary. • It is fragile: any error in anatomy corrupts the motion
estimate, and vice versa. • It does not generalize well to new patient
anatomies. By decoupling, we split the problem cleanly:\
Module What It Learns Shape Model What the heart looks like --- its
static 3D geometry at a reference time (end-diastole)\
Motion Model How the heart moves --- the temporal deformation field that
transforms the shape through time\
Integration Heart(x,y,z,t) = Shape(x,y,z) deformed by Motion(x,y,z,t)
This decoupling is not just an engineering convenience. It reflects the
true physical nature of the heart:\
anatomy

changes

slowly

(patient-specific

baseline)

while

motion

changes

rapidly

(beat-by-beat

dynamics).

Separating

them

makes

both

problems

tractable.

2.2 Why Current Methods Fall Short (Literature Gaps) To justify our
proposed approach, we must understand exactly what existing methods
lack. Existing Method / Category Critical Limitation Traditional
registration (VoxelMorph, ANTs)\
Operates on 3D volumes, not sparse 2D slices; slow; no shape prior\
CNN optical flow (FlowNet, PWC-Net)\
Local receptive field only --- misses long-range cardiac dependencies
(base-apex coupling)\
RAFT (optical flow) Excellent local accuracy but no temporal or
anatomical modeling; designed for 2D video\
Statistical Shape Models (PCA-based)\
Linear --- cannot capture complex nonlinear shape variations; requires
dense 3D inputs\
Implicit neural fields (NeRF-style) Very slow inference; not designed
for dynamic deformation; no biomechanical constraints\
Standard Transformers for video No cardiac-specific priors;
computationally heavy; not adapted for 3D medical imaging

# Page 5

GANs for cardiac reconstruction Unstable training; mode collapse; no
explicit shape or motion disentanglement\
Existing 4D cardiac methods Not real-time; require 3D inputs; no
decoupled formulation; poor generalization RESEARCH GAP SUMMARY No
existing method simultaneously achieves: (1) real-time inference, (2)
decoupled shape + motion modeling,\
(3) accurate motion estimation from sparse 2D inputs, (4) long-range
spatial-temporal dependencies,\
and (5) biomechanically plausible deformation fields. This is the gap we
address.

# Page 6

3.  Why We Choose What We Choose\
    Every architectural decision in this research must be justified.
    Below we raise and answer the critical\
    questions

a

reviewer

or

examiner

would

ask.

3.1 Why RAFT for Motion Estimation? QUESTION: Why RAFT? Why not simpler
optical flow or direct regression? RAFT (Recurrent All-Pairs Field
Transform) produces iteratively refined, high-accuracy displacement
fields.\
Unlike single-pass networks, RAFT builds a 4D correlation volume and
updates its flow estimate 12+ times,\
converging to accurate sub-pixel motion. This iterative refinement is
critical for cardiac motion because:\
• Myocardial displacements are small but clinically significant (1--2mm
differences matter)\
• MRI has noise and artifacts that single-pass networks fail on\
• RAFT has strong theoretical backing and state-of-the-art benchmark
performance\
• The recurrent structure can be unrolled fewer times at inference for
speed (real-time trade-off) The key RAFT components and why each matters
for cardiac imaging:\
RAFT Component Why It Matters for Cardiac Motion Feature Encoder (CNN)
Extracts local image features robust to MRI noise and intensity
variation\
4D Correlation Volume Captures all pairwise feature similarities ---
enables matching across large deformations\
Recurrent Update Block (GRU) Iteratively refines motion estimate ---
converges to accurate small-scale deformations\
Upsampling Head Recovers full-resolution deformation field from coarse
estimates\
3.2 Why Transformers for Temporal Modeling?\
QUESTION: Why not use RNNs or 3D CNNs for temporal modeling? The cardiac
cycle involves long-range temporal dependencies. For example:\
• End-systole can only be understood relative to end-diastole (global
temporal context)\
• Ventricular synchrony --- the two ventricles must be modeled together
across time\
• Base-apex coupling --- basal motion affects apical motion non-locally

# Page 7

RNNs suffer from vanishing gradients over long sequences. 3D CNNs have
limited receptive fields.\
Transformers use self-attention to model ALL pairwise relationships
across space and time simultaneously.\
This makes them uniquely suited to capture the global structure of
cardiac dynamics. Specifically, we use a Transformer Temporal Encoder
to: • Model dependencies between all time frames in the cardiac cycle
simultaneously. • Learn cardiac-phase-aware representations (systole vs
diastole context). • Handle variable-length cardiac cycles across
patients (variable heart rates). • Capture cross-slice spatial
relationships when combined with a spatial attention layer.\
3.3 Why Decouple Shape from Motion?\
QUESTION: Why not just learn the full 4D model end-to-end? Three
fundamental reasons:\
1. LEARNABILITY: The shape subspace is lower-dimensional than full 4D
motion. By separating them,\
each module learns a simpler, more constrained mapping --- faster
convergence, better generalization.\
2. INTERPRETABILITY: Clinicians can inspect and validate the shape
separately from the motion.\
A wrong shape contaminating motion estimates would be invisible in an
end-to-end model.\
3. MODULARITY: Shape can be pre-computed once per patient; motion runs
per cardiac cycle.\
This is essential for real-time clinical deployment.\
3.4 Why Use Biomechanical Constraints?\
QUESTION: Why not just let the network learn unconstrained motion?
Without constraints, a network can produce physically impossible
deformations:\
• Tissue folding or tearing (diffeomorphism violations)\
• Volume changes in incompressible muscle (myocardium is \~97%
incompressible)\
• Discontinuous motion fields (spatially incoherent deformations)\
We incorporate: (1) Jacobian determinant constraint for volume
preservation,\
(2) smoothness regularization via gradient norm penalty, (3)
diffeomorphic registration\
to guarantee invertible transformations. These are not optional --- they
define physical realism.

# Page 8

4.  What Needs to Be Done --- Full Research Methodology\
    4.1 System Architecture Overview The complete Transformer-RAFT
    Hybrid system has five interconnected modules: Module Role in System
    M1: Preprocessing & Segmentation\
    Convert raw DICOM MRI to normalized, segmented myocardial masks\
    M2: Shape Reconstruction Network\
    Reconstruct patient-specific 3D myocardial mesh at end-diastole\
    M3: Transformer Temporal Encoder\
    Learn global spatio-temporal features across the full cardiac cycle\
    M4: RAFT Motion Estimation Module\
    Estimate dense 3D deformation field between cardiac phases\
    M5: 4D Integration & Output Combine shape + motion into final 4D
    cardiac model with strain metrics The data flow through the system:\
    DATA FLOW PIPELINE Cine MRI (2D slices × T frames)\
    ↓ M1: Segmentation + Normalization\
    Segmented Myocardial Slices\
    ↓ M2: 3D Shape Reconstruction (reference at end-diastole)\
    Patient-Specific 3D Shape S(x,y,z)\
    ↓ M3: Transformer Temporal Encoder (all T frames jointly)\
    Temporal Feature Representation F(x,y,z,t)\
    ↓ M4: RAFT Motion Estimation (iterative refinement)\
    Dense Deformation Field Φ( x,y,z,t)\
    ↓ M5: 4D Integration\
    Heart(x,y,z,t) = S(x,y,z) ∘ Φ( x,y,z,t)\
    4.2 Loss Functions Training the model requires a composite loss
    function that enforces multiple constraints simultaneously: Loss
    Term Purpose & Justification

# Page 9

L_recon --- Reconstruction Loss (L2)\
Ensures warped reference frame matches target frame; primary supervision
signal\
L_motion --- Warp Consistency Loss\
Frame_t should equal Warp(Frame_t-1, Φ\_ t); enforces temporal
coherence\
L_smooth --- Gradient Regularization\
Penalizes \|\| ∇ Φ\|\|; prevents spatially discontinuous deformation
fields\
L_shape --- Shape Prior Loss Penalizes deviation from learned shape
distribution; ensures anatomical plausibility\
L_incomp --- Incompressibility (Jacobian)\
Penalizes \|det(J\_ Φ) - 1\|; enforces volume preservation in myocardial
tissue\
L_cycle --- Cycle Consistency Forward-backward motion must be
invertible; prevents degenerate solutions The combined objective:
L_total = L_recon + λ1· L_motion + λ2· L_smooth + λ3· L_shape + λ4·
L_incomp +\
λ5· L_cycle The weights λ1--λ5 are hyperparameters tuned via ablation
study. Each loss term has physical meaning and\
can

be

independently

evaluated.

4.3 Evaluation Metrics\
Metric What It Measures & Why It Matters Dice Score (DSC) 3D overlap
between predicted and ground truth myocardial segmentation; measures
shape accuracy\
Hausdorff Distance (HD95) 95th percentile surface distance; sensitive to
local shape errors near boundaries\
End-Point Error (EPE) Mean distance between predicted and true motion
vectors; direct motion accuracy\
Myocardial Strain Error Difference from ground truth strain (from tagged
MRI); key clinical validation\
Ejection Fraction Error Difference from echocardiography-derived EF;
most widely used clinical metric\
Inference Time (ms) Time from MRI input to 4D output; must be \<1000ms
for real-time clinical use\
Jacobian Determinant Fraction of voxels with det(J) \> 0 (no folding);
measures physical plausibility

# Page 10

5.  Datasets --- Training, Validation & Testing\
    5.1 Primary Datasets\
    Dataset Details & Role in Research ACDC (MICCAI Challenge) 100
    patients; LV/RV/myocardium labels at ED+ES; ideal for initial
    training and standard benchmarking\
    UK Biobank (Cardiac) 50,000+ subjects; large-scale pre-training and
    population diversity; no per-frame labels\
    M&Ms Challenge Multi-site, multi-vendor MRI; tests cross-domain
    generalization --- critical for clinical validity\
    STACOM Motion Tracking Ground truth motion from tagged MRI; gold
    standard for motion estimation evaluation\
    CAMUS (Echocardiography)\
    2D echo; used for cross-modality experiments and weakly supervised
    extension\
    5.2 Data Preprocessing Pipeline 1. Convert DICOM to NIfTI format
    (nibabel / dcm2niix). 2. Resample all volumes to isotropic 1.5mm³
    voxel spacing. 3. Normalize intensity to \[0,1\] using per-patient
    min-max normalization. 4. Crop to bounding box around cardiac region
    (reduce compute by \~70%). 5. Temporal alignment: register all
    frames to end-diastole as reference. 6. Segmentation: train 2D U-Net
    on ACDC for myocardial masking (or use nnU-Net). 7. Augmentation:
    random rotation (±15°), flipping, elastic deformation, intensity
    jitter.

# Page 11

6.  Phase-by-Phase Research Roadmap\
    This is a structured 7-phase plan. Each phase has clear deliverables
    and milestones.\
    PHASE 1: Foundation & Literature Mastery Duration: 3--4 Weeks ✓ Read
    and annotate: original RAFT paper (Teed & Deng 2020)\
    ✓ Read and annotate: Vision Transformer / ViT paper (Dosovitskiy
    2020)\
    ✓ Read and annotate: VoxelMorph (Balakrishnan 2019) for medical
    image registration\
    ✓ Read: 4D Myocardium Reconstruction with Decoupled Motion and Shape
    Model (core paper)\
    ✓ Read: ModusGraph, MCSI-Net for cardiac mesh reconstruction\
    ✓ Set up Python environment: PyTorch, nibabel, SimpleITK, monai,
    wandb\
    ✓ Download ACDC dataset and run nnU-Net baseline segmentation\
    ✓ Reproduce VoxelMorph registration on ACDC as baseline\
    ✓ DELIVERABLE: Literature review document + working codebase
    baseline\
    PHASE 2: Shape Reconstruction Module (M2) Duration: 4--5 Weeks ✓
    Implement 3D U-Net encoder for shape feature extraction\
    ✓ Build shape decoder: either mesh deformation or implicit neural
    field (SDF)\
    ✓ Train on ACDC: predict end-diastole 3D myocardial mesh from 2D
    slices\
    ✓ Evaluate shape quality: Dice Score, Hausdorff Distance vs ground
    truth\
    ✓ Compare two shape representations: mesh vs SDF --- choose based on
    downstream compatibility\
    ✓ Ensure inter-slice gap handling: the network must hallucinate
    missing 3D context\
    ✓ DELIVERABLE: Shape module with Dice \> 0.88 on ACDC test set\
    PHASE 3: Transformer Temporal Encoder (M3) Duration: 4--5 Weeks ✓
    Implement frame-level CNN feature extractor (shared weights across T
    frames)\
    ✓ Build Transformer encoder: 8 heads, 6 layers, positional encoding
    for cardiac phases\
    ✓ Add spatial attention: cross-attention between spatial locations
    within each frame\
    ✓ Test: does Transformer encoding improve downstream motion accuracy
    vs LSTM baseline?\
    ✓ Ablation: number of layers, heads, positional encoding type\
    ✓ DELIVERABLE: Transformer encoder with measurable improvement over
    LSTM baseline

# Page 12

PHASE 4: RAFT Motion Estimation Module (M4) Duration: 5--6 Weeks ✓ Adapt
RAFT from 2D optical flow to 3D volumetric medical imaging\
✓ Modify correlation volume to operate on 3D feature pairs\
✓ Implement recurrent GRU update block for iterative flow refinement\
✓ Add biomechanical constraints: Jacobian determinant loss, smoothness
regularization\
✓ Connect to Transformer features: RAFT initialised with Transformer
temporal context\
✓ Evaluate motion accuracy on STACOM dataset: End-Point Error, strain
correlation\
✓ DELIVERABLE: RAFT motion module with EPE \< 1.5mm on ACDC cardiac
cycle\
PHASE 5: Full System Integration & Joint Training Duration: 4--5 Weeks ✓
Connect all modules: M1 → M2 → M3 → M4 → M5\
✓ Implement joint loss: L_total = L_recon + λ1\* L_motion + λ2\*
L_smooth + λ3\* L_shape + λ4\* L_incomp\
✓ Joint fine-tuning: first train modules separately, then end-to-end\
✓ Hyperparameter search: λ weights, learning rate, RAFT iteration count\
✓ Implement full 4D reconstruction output: mesh + deformation field →
animated heart\
✓ Compute clinical metrics: Ejection Fraction error vs echo ground
truth\
✓ DELIVERABLE: Full working 4D model producing real-time cardiac
reconstruction\
PHASE 6: Ablation Studies & Comparisons Duration: 3--4 Weeks ✓ Ablation
1: Remove Transformer --- replace with LSTM, measure performance drop\
✓ Ablation 2: Remove RAFT --- replace with direct regression, measure
performance drop\
✓ Ablation 3: Remove shape decoupling --- train end-to-end, measure
performance drop\
✓ Ablation 4: Remove biomechanical constraints --- measure physical
plausibility drop\
✓ Baseline comparison: VoxelMorph, standard RAFT, Transformer-only,
CNN-only\
✓ Cross-domain test: train on ACDC, test on M&Ms (multi-vendor
generalization)\
✓ DELIVERABLE: Complete ablation table + comparison table ready for
paper\
PHASE 7: Paper Writing & Submission Duration: 4--5 Weeks

# Page 13

✓ Write paper: Abstract, Introduction, Related Work, Methodology,
Experiments, Conclusion\
✓ Generate publication-quality figures: architecture diagram, result
visualizations, ablation charts\
✓ Format for MICCAI (8 pages + references, LNCS format)\
✓ Internal review: verify all claims are supported by experiments\
✓ Submit to MICCAI or IEEE TMI (depending on novelty depth)\
✓ DELIVERABLE: Submitted paper + open-source code repository (GitHub)

# Page 14

7.  Critical Questions to Raise and Answer\
    These are the questions a reviewer, examiner, or collaborator will
    ask. You must be prepared with clear\
    answers

for

each.

Q1: Is RAFT + Transformer just combining two things? Where is the
novelty?\
ANSWER The novelty is not in combining RAFT and Transformer in
isolation, but in how they are integrated:\
(1) The Transformer provides global cardiac context that initializes
RAFT's correlation volume ---\
this is architecturally novel and physically meaningful.\
(2) Applying RAFT to 3D volumetric medical data with biomechanical
constraints is non-trivial.\
(3) The decoupled shape-motion formulation within a joint deep learning
framework is the core contribution.\
(4) Real-time inference from 2D sparse inputs is a new capability not
demonstrated before.\
Q2: How do you evaluate motion if ground truth 4D motion is
unavailable?\
ANSWER Multiple validation strategies:\
(1) Tagged MRI (DENSE/HARP): provides ground truth motion --- use STACOM
challenge data.\
(2) Synthetic data: generate MRI from known deformation fields, measure
recovery accuracy.\
(3) Clinical surrogates: ejection fraction from echocardiography, wall
motion scoring.\
(4) Cycle consistency: forward-backward motion should cancel ---
measurable without labels.\
Q3: Can this really run in real-time? What is the inference cost?\
ANSWER Real-time is achievable through architectural optimizations:\
(1) Shape module runs ONCE per patient (pre-computation), not per
frame.\
(2) RAFT can be unrolled for fewer iterations (6 instead of 12) for
speed vs accuracy trade-off.\
(3) Mixed-precision inference (FP16) reduces compute by \~2x.\
(4) Target: \<500ms for full 4D reconstruction on a single GPU.\
This will be validated experimentally and reported in the paper.

# Page 15

Q4: Why is cardiac MRI harder than natural video optical flow?\
ANSWER Five key differences:\
(1) 3D vs 2D: cardiac motion is inherently volumetric --- 2D flow models
lose through-plane motion.\
(2) Sparse inputs: only 6--12 slices, not a full volume ---
reconstruction from partial data is ill-posed.\
(3) Low contrast: myocardium and blood pool have similar MRI intensities
--- feature matching is harder.\
(4) Periodic motion: the cardiac cycle is quasi-periodic but varies
between patients and beats.\
(5) Clinical stakes: 1mm errors in cardiac motion can change clinical
decisions.

# Page 16

8.  The Final Goal --- What We Are Building\
    It is critical to be absolutely clear about what the final
    deliverable of this research is, at every level.\
    8.1 Scientific Goal\
    SCIENTIFIC CONTRIBUTION A novel deep learning architecture --- the
    Transformer-RAFT Hybrid --- that reconstructs accurate 4D
    myocardial\
    models from routine 2D cine MRI using a principled decoupled
    shape-motion framework,\
    demonstrated to outperform existing methods on standard benchmarks
    while achieving real-time inference.\
    8.2 Technical Deliverables • Working PyTorch model: full
    implementation with training code, inference pipeline, and
    evaluation\
    scripts. • Trained weights: pre-trained on ACDC and UK Biobank,
    publicly released. • 4D cardiac reconstruction output: animated 3D
    heart mesh + deformation field + strain maps. • Clinical validation:
    ejection fraction, strain, wall thickening compared against
    echocardiography. • Open-source repository: documented GitHub
    repository enabling reproduction of all results.\
    8.3 Research Paper Target\
    Target Venue Why It Fits MICCAI (Medical Image Computing and
    Computer Assisted Intervention)\
    Top venue for medical AI; cardiac track well-established; 8-page
    papers\
    IEEE Transactions on Medical Imaging (TMI)\
    Journal format; allows full methodological depth; very high impact\
    Medical Image Analysis (MedIA) Top journal; ideal if clinical
    validation is strong\
    Nature Machine Intelligence For exceptional novelty + clinical
    impact; requires transformative contribution\
    8.4 Long-Term Vision --- Cardiac Digital Twin This research is the
    foundation of something larger: the AI-driven cardiac digital twin.
    The 4D model\
    produced

by

this

system

can

be

extended

into

a

full

simulation

environment: Extension Description

# Page 17

Electrophysiology integration Overlay electrical conduction model on
reconstructed geometry for arrhythmia simulation\
Hemodynamics (CFD) Use 4D mesh as moving boundary condition for blood
flow simulation\
Biomechanical simulation Finite element model of cardiac mechanics
driven by patient-specific geometry\
Real-time Unreal Engine rendering\
Export deformation field → drive heart mesh in Unreal for visual
simulation\
Multi-organ extension Apply same decoupled framework to lung, brain,
liver digital twins ULTIMATE VISION A real-time AI system that takes a
10-minute cardiac MRI scan as input and produces a complete,\
physically accurate, personalized digital twin of the patient's heart
--- ready for clinical analysis,\
surgical planning, and disease simulation --- within seconds of scan
completion.

# Page 18

9.  Key References to Read First\
    These papers must be read before beginning any implementation: Paper
    Why You Must Read It RAFT: Recurrent All-Pairs Field Transforms for
    Optical Flow (Teed & Deng, ECCV 2020)\
    Core architecture of your motion module\
    An Image is Worth 16x16 Words: Transformers for Image Recognition
    (ViT, Dosovitskiy 2021)\
    Foundation of your temporal encoder\
    VoxelMorph: A Learning Framework for Deformable Medical Image
    Registration (Balakrishnan 2019)\
    Primary baseline to beat; understand deeply\
    4D Myocardium Reconstruction with Decoupled Motion and Shape Model
    (core paper)\
    The direct paper this research builds upon\
    ModusGraph: Automated 3D and 4D Mesh Model Reconstruction from Cine
    CMR\
    State of the art for cardiac mesh reconstruction\
    Deep Statistic Shape Model for Myocardium Segmentation\
    Understand statistical shape modeling approach\
    Generative Myocardial Motion Tracking via Latent Space Exploration\
    Best current motion tracking baseline

Research Plan Complete From understanding the problem → to a working
model → to a published paper

# Page 19

5.  Datasets --- Training, Validation & Testing\
    5.1 Primary Datasets\
    Table\
    Dataset Details & Role Access Status Sample Size\
    ACDC (MICCAI Challenge)\
    LV/RV/myocardium labels at ED+ES; ideal for initial training and
    standard benchmarking\
    Public; immediate 100 patients\
    UK Biobank (Cardiac)\
    Large-scale pre-training and population diversity; no per-frame
    labels\
    Requires application (\~2--3 months)\
    50,000+ subjects\
    M&Ms Challenge\
    Multi-site, multi-vendor MRI; tests cross-domain generalization\
    Public; immediate \~300 cases\
    STACOM Motion Tracking\
    Ground truth motion from tagged MRI (DENSE/HARP); gold standard for
    motion evaluation\
    Public; limited tagged data\
    \~100 cases

# Page 20

CAMUS (Echocardiography)\
2D echo for cross-modality experiments and weakly supervised extension\
Public; immediate 500 patients\
Synthetic (FE Simulation)\
Backup ground truth from biomechanical simulations\
Generated in-house\
Unlimited\
5.2 Data Preprocessing Pipeline\
1. Convert DICOM to NIfTI format (nibabel / dcm2niix) 2. Resample all
volumes to isotropic 1.5 mm³ voxel spacing 3. Normalize intensity to
\[0,1\] using per-patient min-max normalization 4. Crop to bounding box
around cardiac region (reduce compute by \~70%) 5. Temporal alignment:
register all frames to end-diastole as reference 6. Segmentation: train
2D U-Net on ACDC for myocardial masking (or use nnU-Net) 7.
Augmentation: random rotation (±15°), ﬂipping, elastic deformation,
intensity\
scaling

5.3 Dataset Access Strategy\
plain\
Priority Timeline: ├── Month 1: Download ACDC, M&Ms, CAMUS immediately
├── Month 1: Apply for UK Biobank access (parallel) ├── Month 2: Set up
synthetic data pipeline (Abaqus/FEbio) ├── Month 3: Begin pre-training
on ACDC + M&Ms ├── Month 4--6: Integrate UK Biobank when approved └──
Month 6+: Validate on STACOM motion tracking data

6.  Phase-by-Phase Research Roadmap\
    PHASE 1: Foundation & Literature Mastery\
    Duration: 3--4 Weeks\
    ● Read and annotate: original RAFT paper (Teed & Deng, ECCV 2020)

# Page 21

● Read and annotate: Vision Transformer / ViT paper (Dosovitskiy 2020) ●
Read and annotate: VoxelMorph (Balakrishnan 2019) for medical image
registration ● Read: 4D Myocardium Reconstruction with Decoupled Motion
and Shape Model ● Read: ModusGraph, MCSI-Net for cardiac mesh
reconstruction ● Set up Python environment: PyTorch, nibabel, SimpleITK,
MONAI, wandb ● Download ACDC dataset and run nnU-Net baseline
segmentation ● Reproduce VoxelMorph registration on ACDC as baseline ●
DELIVERABLE: Literature review document + working codebase baseline\
PHASE 2: Shape Reconstruction Module (M2)\
Duration: 4--5 Weeks\
● Implement 3D U-Net encoder for shape feature extraction ● Build shape
decoder: explicit mesh deformation (chosen over SDF for real-time\
compatibility) ● Train on ACDC: predict end-diastole 3D myocardial mesh
from 2D slices ● Evaluate shape quality: Dice Score, Hausdorff Distance
vs. ground truth ● Ensure inter-slice gap handling: network must
hallucinate missing 3D context from\
learned

prior ● DELIVERABLE: Shape module with Dice \> 0.88 on ACDC test set\
PHASE 3: Transformer Temporal Encoder (M3)\
Duration: 4--5 Weeks\
● Implement frame-level CNN feature extractor (shared weights across T
frames) ● Build Transformer encoder: 8 heads, 6 layers, sinusoidal
positional encoding for\
cardiac

phases ● Add spatial attention: cross-attention between spatial
locations within each frame ● Test: does Transformer encoding improve
downstream motion accuracy vs. LSTM\
baseline? ● Ablation: number of layers, heads, positional encoding type
(sinusoidal vs. learned) ● DELIVERABLE: Transformer encoder with
measurable improvement over LSTM\
baseline

PHASE 4: RAFT Motion Estimation Module (M4)\
Duration: 5--6 Weeks\
● Adapt RAFT from 2D optical ﬂow to 3D volumetric medical imaging ●
Modify correlation volume to operate on 3D feature pairs

# Page 22

● Implement recurrent GRU update block for iterative ﬂow reﬁnement ● Add
biomechanical constraints: Jacobian determinant loss, smoothness\
regularization ● Connect to Transformer features: RAFT initialized with
Transformer temporal\
context ● Evaluate motion accuracy on STACOM dataset: End-Point Error,
strain correlation ● DELIVERABLE: RAFT motion module with EPE \< 1.5mm
on ACDC cardiac cycle\
PHASE 5: Full System Integration & Joint Training\
Duration: 4--5 Weeks\
● Connect all modules: M1 → M2 → M3 → M4 → M5 ● Implement joint loss:
L_total = L_recon + λ₁· L_motion + λ₂· L_smooth + λ₃· L_shape +\
λ₄· L_incomp ● Joint ﬁne-tuning: ﬁrst train modules separately, then
end-to-end ● Hyperparameter search: λ weights, learning rate, RAFT
iteration count ● Implement full 4D reconstruction output: mesh +
deformation ﬁeld → animated\
heart ● Compute clinical metrics: Ejection Fraction error vs. echo
ground truth ● DELIVERABLE: Full working 4D model producing real-time
cardiac reconstruction\
PHASE 6: Ablation Studies & Comparisons\
Duration: 3--4 Weeks\
Table\
Ablation What We Remove What We Measure\
Ablation 1\
Transformer → replace with LSTM Motion accuracy drop\
Ablation 2\
RAFT → replace with direct regression\
EPE increase, convergence behavior

# Page 23

Ablation 3\
Shape decoupling → train end-to-end Generalization gap, error
propagation\
Ablation 4\
Biomechanical constraints Physical plausibility (Jacobian violations)\
●\
Baseline

comparison:

VoxelMorph,

standard

RAFT,

Transformer-only,

CNN-only ● Cross-domain test: train on ACDC, test on M&Ms (multi-vendor
generalization) ● DELIVERABLE: Complete ablation table + comparison
table ready for paper\
PHASE 7: Paper Writing & Submission\
Duration: 4--5 Weeks\
● Write paper: Abstract, Introduction, Related Work, Methodology,
Experiments,\
Conclusion ● Generate publication-quality ﬁgures: architecture diagram,
result visualizations,\
ablation

charts ● Format for MICCAI (8 pages + references, LNCS format) or IEEE
TMI ● Internal review: verify all claims are supported by experiments ●
Submit to target venue ● DELIVERABLE: Submitted paper + open-source code
repository (GitHub)

7.  Critical Questions to Raise and Answer\
    Q1: Is RAFT + Transformer just combining two things? Where is the
    novelty?\
    Answer: The novelty is not in combining RAFT and Transformer in
    isolation, but in how they\
    are

integrated:

1.  The Transformer provides global cardiac context that initializes
    RAFT's correlation\
    volume

---

this

is

architecturally

novel

and

physically

meaningful 2. Applying RAFT to 3D volumetric medical data with
biomechanical constraints is\
non-trivial

# Page 24

3.  The decoupled shape-motion formulation within a joint deep learning
    framework is\
    the

core

contribution 4. Real-time inference from 2D sparse inputs is a new
capability not demonstrated\
before

Q2: How do you evaluate motion if ground truth 4D motion is
unavailable?\
Answer: Multiple validation strategies:\
1. Tagged MRI (DENSE/HARP): provides ground truth motion --- use STACOM\
challenge

data 2. Synthetic data: generate MRI from known deformation ﬁelds,
measure recovery\
accuracy 3. Clinical surrogates: ejection fraction from
echocardiography, wall motion scoring 4. Cycle consistency:
forward-backward motion should cancel --- measurable without\
labels

Q3: Can this really run in real-time? What is the inference cost?\
Answer: Real-time is achievable through architectural optimizations:\
1. Shape module runs ONCE per patient (pre-computation), not per frame
2. RAFT can be unrolled for fewer iterations (6 instead of 12) for speed
vs. accuracy\
trade-off 3. Mixed-precision inference (FP16) reduces compute by \~2× 4.
Target: \<500ms for full 4D reconstruction on a single GPU\
Q4: Why is cardiac MRI harder than natural video optical ﬂow?\
Answer: Five key differences:\
1. 3D vs 2D: cardiac motion is inherently volumetric --- 2D ﬂow models
lose\
through-plane

motion 2. Sparse inputs: only 6--12 slices, not a full volume ---
reconstruction from partial data\
is

ill-posed 3. Low contrast: myocardium and blood pool have similar MRI
intensities --- feature\
matching

is

harder 4. Periodic motion: the cardiac cycle is quasi-periodic but
varies between patients and\
beats 5. Clinical stakes: 1mm errors in cardiac motion can change
clinical decisions\
Q5: What happens with truly sparse input (e.g., 6 slices with 10mm
gaps)?

# Page 25

Answer: This is where the learned shape prior becomes essential. The
shape module\
learns

a

population-level

prior

of

plausible

cardiac

anatomies

from

full

3D

data

during

training.

At

inference,

this

prior

hallucinates

anatomically

plausible

geometry

in

the

gaps

between

slices.

The

uncertainty

in

these

regions

can

be

quantiﬁed

and

reported

to

clinicians.

8.  The Final Goal --- What We Are Building\
    8.1 Scientiﬁc Goal\
    A novel deep learning architecture --- the Transformer-RAFT Hybrid
    --- that reconstructs\
    accurate

4D

myocardial

models

from

routine

2D

cine

MRI

using

a

principled

decoupled

shape-motion

framework,

demonstrated

to

outperform

existing

methods

on

standard

benchmarks

while

achieving

real-time

inference.

8.2 Technical Deliverables\
● Working PyTorch model: full implementation with training code,
inference pipeline,\
and

evaluation

scripts ● Trained weights: pre-trained on ACDC and UK Biobank, publicly
released ● 4D cardiac reconstruction output: animated 3D heart mesh +
deformation ﬁeld +\
strain

maps ● Clinical validation: ejection fraction, strain, wall thickening
compared against\
echocardiography ● Open-source repository: documented GitHub repository
enabling reproduction of all\
results

8.3 Research Paper Target\
Table\
Venue Why It Fits Format

# Page 26

MICCAI 2026 Top venue for medical AI; cardiac track well-established\
8 pages + references, LNCS\
IEEE TMI Journal format; allows full methodological depth; very high
impact\
Full journal article\
Medical Image Analysis (MedIA)\
Top journal; ideal if clinical validation is strong Full journal
article\
Nature Machine Intelligence\
For exceptional novelty + clinical impact; requires transformative
contribution\
Full journal article\
Parallel submission strategy: Target STACOM 2026 workshop (at MICCAI)
for intermediate\
results,

then

full

MICCAI

or

IEEE

TMI

for

complete

system.

8.4 Long-Term Vision --- Cardiac Digital Twin\
This research is the foundation of the AI-driven cardiac digital twin.
The 4D model can be\
extended

into

a

full

simulation

environment:

Table\
Extension Description\
Electrophysiology integration Overlay electrical conduction model on
reconstructed geometry for arrhythmia simulation\
Hemodynamics (CFD) Use 4D mesh as moving boundary condition for blood
ﬂow simulation

# Page 27

Biomechanical simulation Finite element model of cardiac mechanics
driven by patient-speciﬁc geometry\
Real-time Unreal Engine rendering\
Export deformation ﬁeld → drive heart mesh in Unreal for visual
simulation\
Multi-organ extension Apply same decoupled framework to lung, brain,
liver digital twins\
Ultimate Vision: A real-time AI system that takes a 10-minute cardiac
MRI scan as input\
and

produces

a

complete,

physically

accurate,

personalized

digital

twin

of

the

patient's

heart

---

ready

for

clinical

analysis,

surgical

planning,

and

disease

simulation

---

within

seconds

of

scan

completion.

9.  Key References\
    Table\
    Paper Why Essential\
    RAFT: Recurrent All-Pairs Field Transforms for Optical Flow (Teed &
    Deng, ECCV 2020)\
    Core architecture of motion module\
    An Image is Worth 16×16 Words: Transformers for Image Recognition
    (ViT, Dosovitskiy 2021)\
    Foundation of temporal encoder

# Page 28

VoxelMorph: A Learning Framework for Deformable Medical Image
Registration (Balakrishnan 2019)\
Primary baseline to beat\
4D Myocardium Reconstruction with Decoupled Motion and Shape Model\
Direct predecessor paper\
ModusGraph: Automated 3D and 4D Mesh Model Reconstruction from Cine CMR\
State-of-the-art cardiac mesh reconstruction\
Deep Statistical Shape Model for Myocardium Segmentation\
Statistical shape modeling approach\
Generative Myocardial Motion Tracking via Latent Space Exploration\
Best current motion tracking baseline\
An Unsupervised Learning Model for Deformable Medical Image Registration
(Diffeomorphic VoxelMorph, CVPR 2019)\
Diffeomorphic constraints implementation

10. Appendix: Implementation Details\
    A.1 Repository Structure\
    plain\
    cardiac-4d-transformer-raft/ ├── data/ \# DICOM → NIfTI
    preprocessing scripts ├── models/ │ ├── shape/ \# M2: Shape
    reconstruction (3D U-Net + mesh\
    decoder) │ ├── temporal/ \# M3: Transformer encoder │ ├── motion/ \#
    M4: RAFT 3D adaptation

# Page 29

│ └── integration/ \# M5: Full pipeline + clinical metrics ├── losses/
\# All loss functions with unit tests ├── utils/ \# Visualization,
metrics, biomechanics ├── configs/ \# YAML configs for each experiment
├── experiments/ \# WandB logging, hyperparameter sweeps ├── scripts/ │
├── train_shape.py │ ├── train_motion.py │ ├── train_full.py │ └──
evaluate.py └── notebooks/ \# Exploration + figure generation\
A.2 Hardware Requirements\
● Training: 4× NVIDIA A100 80GB GPUs (or equivalent) ● Inference: 1×
NVIDIA RTX 4090 or A6000 (target \<1s per patient) ● Storage: \~2TB for
full datasets (ACDC + M&Ms + UK Biobank subset)\
A.3 Software Stack\
● PyTorch 2.0+ with CUDA 11.8 ● MONAI for medical imaging primitives ●
PyTorch3D or Trimesh for mesh operations ● Weights & Biases for
experiment tracking ● GitHub Actions for CI/CD

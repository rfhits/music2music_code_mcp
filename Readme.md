# Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization

This is the code for experiments in the paper *Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization*. 

## Structure

    .
    ├── baselines               Baseline implementations
    ├── dataset_preparation     Prepare dataset for training
    ├── evaluations             Functions to evaluate model performance
    ├── m2m                     Model training
    ├── tests                   Sanity check scripts
    ├── utils_arrange           Utility functions for arrangement inference
    ├── utils_chord             Utility functions for chord detection
    ├── utils_common            Common Utility functions
    ├── utils_instrument        Utility functions for instrument
    ├── utils_midi              Utility functions for MIDI, including the proposed tokenization
    └── utils_texture           Utility functions for texture analysis

Most useful scripts are located in the `m2m` directory, containing: 
- Model definition: `m2m_models.py` and `lightning_model.py`.
- Dataset class: `lightning_dataset.py`.
- Pre-train: `pretrain.py`
- Fine-tuning script: `lightning_train.py`
- Testing script: `lightning_test.py`
- Inference (band/piano arrangement): `reinst.py`
- Inference (drum arrangement): `drum_arrange.py`

The hyperparameter of all experiments are saved in `m2m/hparams`.

<!-- ## Results
### Objective Evaluation Results for the Band Arrangement Task

Statistical significance: * for _p_ < 0.05, † for _p_ < 0.01, ‡ for _p_ < 0.001.

| Model                      | I-IOU ↑      | V-WER ↓     | Note F1 ↑   | Note<sub>i</sub> F1 ↑ | Mel F1 ↑    |
|---------------------------|--------------|-------------|-------------|------------------------|-------------|
| Transformer-VAE           | 97.5         | 35.0        | 49.5        | 40.0                   | 24.7        |
| Transformer w/ REMI+      | 95.0         | 18.2        | 94.4        | 76.0                   | 68.8        |
| Transformer w/ REMI-z     | ‡99.5        | ‡9.9        | ‡**97.8**   | ‡77.5                  | ‡77.8       |
| + Pre-training (Ours)     | *_99.8_*     | **7.6**     | *_97.5_*    | **87.0**               | **84.5**    |
| – voice                   | 99.6         | *_17.6_*    | 97.2        | *_84.3_*               | *_81.5_*    |
| – history                 | **100.0**    | *_9.0_*     | 97.6        | 77.4                   | 79.4        |

### Band Arrangement Subjective Evaluation Results

*Fa.* = Faithfulness, *Co.* = Coherence, *In.* = Instrumentation, *Cr.* = Creativity, *Mu.* = Musicality  
Statistical significance: * for _p_ < 0.05, † for _p_ < 0.01, ‡ for _p_ < 0.001.

| Model           | Fa. ↑ | Co. ↑ | In. ↑ | Cr. ↑ | Mu. ↑ |
|----------------|--------|--------|--------|--------|--------|
| Rule-Based      | _3.46_ | _3.05_ | _2.89_ | _3.00_ | _3.07_ |
| Transformer-VAE | 2.65   | 2.70   | 2.72   | _3.00_ | 2.72   |
| Ours            | ‡**3.77** | ‡**3.47** | ‡**3.49** | ***3.40** | ‡**3.47** |
| w/o PT          | 3.19   | 2.82   | 2.86   | 2.93   | 2.75   |

### Piano Reduction Results

*F1* = Note F1 (objective), *Fa.* = Faithfulness, *Co.* = Coherence, *Pl.* = Playability, *Cr.* = Creativity, *Mu.* = Musicality  
Statistical significance: * for _p_ < 0.05, † for _p_ < 0.01, ‡ for _p_ < 0.001.

| Model     | F1 ↑   | Fa. ↑       | Co. ↑       | Pl. ↑       | Cr. ↑       | Mu. ↑       |
|-----------|--------|-------------|-------------|-------------|-------------|-------------|
| Rule-F    | –      | **3.93**    | _3.59_      | 3.14        | _2.96_      | _3.34_      |
| Rule-O    | –      | 2.75        | 3.49        | **4.07**    | 2.62        | 2.96        |
| UNet      | 58.3   | 2.97        | 2.90        | 3.47        | 2.82        | 2.78        |
| Ours      | **85.5** | ‡_3.63_    | ‡**3.64**   | *3.86*      | ***3.14**   | ‡**3.48**   |
| w/o PT    | _78.4_ | 2.25        | 2.58        | 3.29        | 2.67        | 2.26        |

### Drum Arrangement Results

*F1* = Note F1 (objective), *Comp.* = Compatibility, *Co.* = Coherence, *Tr.* = Phrase Transition, *Cr.* = Creativity, *Mu.* = Musicality  
Statistical significance: * for _p_ < 0.05, † for _p_ < 0.01, ‡ for _p_ < 0.001.

| Model         | F1 ↑    | Comp. ↑     | Co. ↑       | Tr. ↑         | Cr. ↑         | Mu. ↑        |
|---------------|---------|-------------|-------------|---------------|---------------|--------------|
| Ground Truth  | **100.0** | **4.31**    | **4.18**    | _3.36_        | _3.16_        | **3.78**     |
| CA v2         | 20.3    | 3.82        | _4.05_      | 2.86          | 2.58          | 3.19         |
| Ours          | _79.3_  | _3.91_      | 4.03        | ‡**3.77**     | ‡**3.27**     | †_3.57_      |
| w/o PT        | 1.2     | 2.49        | 2.19        | 2.21          | 2.82          | 2.05         |


### Tokenization Scheme Comparison on Unconditional Generation

*↓* indicates lower is better.  
$\bar{T}_{\text{bar}}$ = tokens per bar,  
$\bar{T}_{\text{note}}$ = tokens per note,  
$\bar{H}_{\text{bar}}$ = bar-level entropy,  
PPL = perplexity.

| Tokenizer        | 𝑇̄<sub>bar</sub> ↓ | 𝑇̄<sub>note</sub> ↓ | 𝐻̄<sub>bar</sub> ↓ | PPL<sub>note</sub> ↓ | PPL<sub>token</sub> ↓ |
|------------------|--------------------|----------------------|---------------------|-----------------------|------------------------|
| REMI+            | 225.91             | 4.03                 | 41.68               | 116.20                | **3.00**               |
| REMI-z (Ours)    | **151.68**         | **2.77**             | **29.43**           | **84.11**             | 4.50                   | -->



## Prepare Environment
    # Environment
    conda create -n m2m python=3.8
    conda activate m2m

    # Install PyTorch
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

    # Dependencies
    pip install -r requirements.txt   
    
    git clone https://github.com/Sonata165/REMI-z.git
    cd REMI-z
    pip install -r Requirements.txt
    pip install -e .


## Dataset preparation

    cd dataset_preparation

    # For pre-training
    python LAMD_to_remi.py

    # For fine-tuning
    python segment_data.py

## Training and Testing
Use below command for pre-train your own symbolic music generator: 

    python pretrain.py
Use below command to execute fine-tuning:
    
    python lightning_train.py [path/to/hparam]
    
    e.g.,
    python lightning_train.py hparams/elaborator_direct.yaml # our band arrangement model in the paper

After training, use below command for testing to get objective metrics (for voice separation and probing experiments)

    python lightning_test.py [path/to/hparam]

In particular, to obtain the results in the paper

    cd m2m

    # Generative pre-train
    python pretrain.py

    # Non-creative band arrangement for objective evaluation
    python lightning_train.py hparams/band_obj/ours.yaml

    # Band arrangement model
    python lightning_train.py hparams/band_arrange/elaborator.yaml

    # Piano reduction model
    python lightning_train.py hparams/piano_reduction/reduction_dur.yaml

    # Drum arrangement model
    python lightning_train.py hparams/drum_arrange/direct_opd.yaml

## Inference
To inference with models trained by yourself, use below command to arrange existing songs with new instrument set:

    python reinst.py [path/to/hparam]

Use below command to arrange a drum track for a song:

    python drum_arrange.py [path/to/hparam]

To directly use our pretrained model for inference, see tutorial.ipynb for details.


## Mapping from token to represented value
As in the appendix, you can find the mapping from token to their corresponding representing values, for time signature and tempo tokens, in `ts_dict.yaml` and `tempo_dict.yaml`, under the `utils_midi` directory.
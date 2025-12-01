---
title: "OpenSR-SRGAN: A Flexible Super-Resolution Framework for Multispectral Earth Observation Data"
tags:
  - super-resolution
  - remote sensing
  - GAN
  - ESRGAN
  - Sentinel-2
  - multispectral
  - PyTorch
  - OpenSR
  - technical report
authors:
  - name: Simon Donike
    orcid: 0000-0002-4440-3835
    corresponding: true
    email: simon.donike@uv.es
    affiliation: 1
  - name: Cesar Aybar
    orcid: 0000-0003-2745-9535
    affiliation: 1
  - name: Julio Contreras
    orcid: 0009-0001-5408-7055
    affiliation: 1
  - name: Luis Gómez-Chova
    orcid: 0000-0003-3924-1269
    affiliation: 1
affiliations:
  - index: 1
    name: Image and Signal Processing Group, University of Valencia, Spain
date:  15 November 2025
bibliography: paper.bib
version:
  report: v0.1.0
  software: v0.1.0
number-sections: true
---

# Summary

OpenSR-SRGAN is a modular, configuration-driven framework for training and benchmarking SRGAN-style super-resolution models on multispectral Earth-observation data. The software enables users to train and evaluate configurable generator–discriminator architectures on arbitrary sensor band setups using concise YAML configurations, without modifying model code. All components, such as generators, discriminators, loss functions, training schedules, normalizations, and stability heuristics, are exposed through a common YAML interface. OpenSR-SRGAN makes it straightforward to reproduce experiments, compare architectures, and deploy super-resolution pipelines across diverse remote-sensing datasets.

OpenSR-SRGAN supports complete end-to-end workflows with minimal setup: selecting architectures, scaling factors, band combinations, and training strategies entirely from configuration files. Although initially designed for remote-sensing super-resolution, the framework is domain-agnostic at its core and can be directly applied to other imaging modalities, such as medical imaging and standard computer-vision datasets, without architectural changes.


# Introduction

Optical satellite imagery supports a wide range of geospatial applications, including agriculture [@agriculture], land-cover mapping [@mapping], ecosystem assessment [@ecosysetm], and disaster monitoring [@disaster]. Sensors such as Sentinel-2 offer rich spectral information at frequent revisit intervals but are limited in spatial resolution, while very-high-resolution missions (e.g., Pleiades, SPOT) provide finer detail at higher cost and lower temporal coverage. This trade-off motivates the use of single-image super-resolution (SISR), which learns to enhance low-resolution observations by predicting a plausible high-resolution counterpart.

Deep learning has substantially advanced SISR, with convolutional models demonstrating strong gains in reconstruction fidelity and perceptual quality [@dong2015imagesuperresolutionusingdeep; @kim2016deeplyrecursiveconvolutionalnetworkimage]. In remote sensing, such techniques can help bridge the spatial gap between freely available medium-resolution data and high-resolution commercial products, improving the utility of satellite imagery for downstream tasks.

Generative Adversarial Networks (GANs) [@goodfellow2014generativeadversarialnetworks] introduced an adversarial learning method in which a generator and discriminator co-evolve to produce realistic, high-frequency detail. GANs have since been widely used in remote sensing for cloud removal, image translation, domain adaptation, and data synthesis [@11159252; @su2024intriguingpropertycounterfactualexplanation], demonstrating their ability to generate spatially coherent and perceptually plausible imagery.

The Super-Resolution GAN (SRGAN) [@ledig2017photo] extended this idea to image enhancement by combining perceptual and adversarial losses for photorealistic detail reconstruction. SRGAN-style methods have since been adopted for remote-sensing super-resolution [@rs15205062; @9787539; @10375518; @satlassuperres], where recovering fine spatial structure from coarse inputs can improve analysis of land cover, infrastructure, and environmental change.

Recent advances in diffusion and transformer-based architectures have shifted the state of the art in image super-resolution toward generative models with stronger probabilistic and contextual reasoning [@s1; @s2; @s3].  
Nevertheless, GAN-based approaches continue to be actively explored [@g1] and remain a practical choice for operational production settings [@allen].

# Statement of Need

GANs remain difficult to train [@p1; @p2; @p3], and these challenges are amplified in remote sensing, where models must handle multispectral inputs, high dynamic-range reflectance values, heterogeneous sensor characteristics, and limited availability of perfectly aligned high-resolution reference data. Existing open-source SRGAN implementations are typically designed for fixed RGB imagery and reproduce only a single architecture from the literature, offering little flexibility for modifying band configurations, normalization schemes, loss compositions, or training strategies. As a result, researchers who need to adjust models for different sensors (e.g., Sentinel-2, SPOT, Pleiades, PlanetScope, or ther modalities) often must re-engineer core components, modify low-level code, and manually implement stabilization heuristics such as warmup, ramping, or EMA tracking. This makes reproducing published experiments or conducting systematic comparisons across architectures labor-intensive, brittle, and inconsistent across studies.

# Contribution Summary

OpenSR-SRGAN was developed to address these challenges by providing a unified, modular, and extensible framework for training and evaluating GAN-based super-resolution models in remote sensing. The software integrates multiple state-of-the-art SR architectures, loss functions, and training strategies within a configuration-driven design that allows users to flexibly adapt experiments without modifying the source code.

The main contributions of this work include:

- **Modular GAN framework:** Interchangeable generator and discriminator backbones with configurable depth, width, and scale factors.  
- **Configuration-first workflow:** Fully reproducible training and evaluation through concise YAML files, independent of code changes.  
- **Training stabilization techniques:** warmup, ramping, label smoothing, spectral normalization, EMA smoothing, and more.  
- **Multispectral compatibility:** Provides native support for arbitrary band configurations from all arbitrary sensor configurations.  
- **OpenSR ecosystem integration:** Connects seamlessly to the SEN2NAIP dataset, leverages the unified evaluation framework `opensr-test` [@osrtest], and supports scalable inference via `opensr-utils` [@osrutils].

Together, these components provide a flexible, domain-agnostic foundation for SRGAN-based super-resolution research, particularly suited to the diverse and heterogeneous requirements of Earth-observation imagery.

# Software Overview and Framework Design

OpenSR-SRGAN is organized as a modular, configuration-driven framework that allows users to define complete super-resolution experiments through concise YAML files. Instead of modifying model code, users specify generators, discriminators, losses, optimizers, normalizations, and training schedules directly in the configuration. This design enables reproducible experimentation and makes it easy to adapt SRGAN workflows to different sensors, band configurations, or imaging modalities.

The framework consists of four main modules:

1. **Generator and Discriminator Registry**  
   A library of interchangeable generator and discriminator backbones that can be selected and parameterized entirely from the YAML configuration. Users can choose residual, attention-based, RRDB or ESRGAN-style variants, control depth and width, and set the desired upsampling factor. Detailed descriptions of all supported architectures are available in the online documentation at https://srgan.opensr.eu.

2. **Configurable Loss System**  
   Pixel, perceptual, spectral and adversarial losses can be freely combined and weighted. Optional components such as SAM, TV or smoothness regularizers can be enabled as needed. This allows users to tailor objectives to multispectral or domain-specific requirements without modifying the training code.

3. **Training Pipeline**  
   The training engine includes a set of practical stabilization features often required in GAN optimization. These include generator warmup, adversarial ramping, spectral normalization, optional EMA tracking of generator weights and support for WGAN objectives with R1 regularization. All training behavior is configured through the YAML file, which enables experiment replication and systematic comparison of strategies.

4. **Data Handling**  
   OpenSR-SRGAN provides complete data loading, normalization and augmentation pipelines for multispectral inputs. It supports automatic ingestion of pre-built datasets such as SEN2NAIP [@sen2naip] and allows users to apply custom transforms and dataset definitions through configuration.

5. **Integration with the OpenSR Ecosystem**  
   The framework interoperates with `opensr-test` for standardized evaluation and with `opensr-utils` for efficient tile-based inference on large scenes. This makes it straightforward to benchmark models or deploy them in production-style remote sensing workflows.

Overall, OpenSR-SRGAN provides a flexible and extensible structure that separates experiment design from implementation details. Its configuration-first approach lowers the barrier to experimenting with SRGAN-based super-resolution and supports rapid exploration across datasets, architectures and training strategies.


# Training Features

OpenSR-SRGAN provides a set of commonly used and proven tools that make GAN training more stable and easier to configure [@mescheder2018r1]. All training behavior is specified through the `Training` section of the YAML configuration file, so users can experiment with different strategies without changing source code.

## Stabilization Options
The framework includes several commonly used techniques to improve adversarial training stability. These include generator warmup, adversarial loss ramping, label smoothing, optional spectral normalization in the discriminator and adaptive learning rate scheduling. EMA tracking of the generator weights can be enabled to produce smoother validation outputs and reduce oscillations during training.

## Wasserstein and R1 Support
For users who prefer Wasserstein-based objectives, OpenSR-SRGAN supports WGAN training with an optional R1 gradient penalty [@arjovsky2017wasserstein]. This mode replaces the standard GAN loss with a critic formulation that can offer more stable gradients when working with multispectral data or high dynamic range inputs. All settings can be enabled by adjusting a few entries in the YAML configuration.

## Flexible Loss Composition
Loss functions are fully modular. Pixel, perceptual, spectral and adversarial losses can be combined in any proportion, and additional terms such as SAM or TV regularizers can be activated as needed. This allows users to balance perceptual realism and spectral fidelity depending on the dataset and application. All loss weights and components are defined in the configuration file, and the training loop automatically logs the corresponding metrics.



# Limitations
Super-resolution techniques, including those implemented in OpenSR-SRGAN, can enhance apparent spatial detail but can never substitute for true high-resolution observations acquired by native sensors. While OpenSR-SRGAN provides a stable and extensible foundation for GAN-based super-resolution in remote sensing, several limitations remain. First, the framework focuses on the engineering and reproducibility aspects of model development rather than achieving state-of-the-art quantitative performance. It is therefore intended as a research and benchmarking blueprint, not as an optimized production model. Second, although the modular configuration system greatly simplifies experimentation, users are still responsible for ensuring proper data preprocessing, radiometric normalization, and accurate LR–HR alignment, factors that strongly influence training stability and reconstruction quality. Third, adversarial optimization in multispectral domains remains sensitive to dataset size and diversity; small or unbalanced datasets may still yield mode collapse or spectral inconsistencies despite the provided stabilization mechanisms. Finally, the current release does not include native uncertainty estimation or automatic hyperparameter tuning; these remain open areas for future extension.

# Licensing and Availability
the source code is made available through the [ESAOpenSR/OpenSR-SRGAN](https://github.com/ESAOpenSR/SRGAN) Github repository. Full documentation, API references, quickstart guides and tips and tricks can be found at  [srgan.opensr.eu](https://githsrgan.opensr.eu). A reproducible notebook is permanently hosted on [Google Colab](https://colab.research.google.com/drive/16W0FWr6py1J8P4po7JbNDMaepHUM97yL?usp=sharing).
In the spirit of open science and collaboration, we encourage feature requests and updates, bug fixes and reports, as well as general questions and concerns via direct interaction with the repository. `OpenSR-SRGAN` is licensed under the Apache-2.0 license.

# Acknowledgement
This work has been supported by the European Space Agency (ESA) $\Phi$-Lab, within the framework of the ['Explainable AI: Application to Trustworthy Super-Resolution (OpenSR)'](https://eo4society.esa.int/projects/opensr/) Project.

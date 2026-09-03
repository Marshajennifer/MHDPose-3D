# MHDPose: Multi-hypothesis 3D human pose estimation using bidirectional Mamba diffusion models

Code for MHDPose, a diffusion-based framework for 3D human pose estimation, benchmarked against a wide range of recent baselines on Human3.6M and MPI-INF-3DHP.

**Paper:** [Link to the paper](https://www.sciencedirect.com/science/article/pii/S0031320326013610)

### Multi-hypotheses pose estimation:

<p align="center">
  <img src="### Sample Pose Estimation Outputs

<p align="center">
  <img src="visuals/additional_image_1.png" width="45%">
</p>


### Demo Video

Below is a short demo of the model output:

<video src="visuals/representative_video.mp4" controls width="800"></video>
<p align="center">
  <img src="visuals/representative_video.gif" width="800">
</p>

[Watch the full MP4 video](visuals/representative_video.mp4)


## Datasets

Our model is evaluated on the Human3.6M and MPI-INF-3DHP datasets.

### Human3.6M

We follow the same data setup as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Download the processed data from [here](#) and place it in the `./data` directory:
- `data_2d_h36m_gt.npz` — ground-truth 2D keypoints
- `data_2d_h36m_cpn_ft_h36m_dbb.npz` — 2D keypoints detected using CPN
- `data_3d_h36m.npz` — ground-truth 3D joint positions

### MPI-INF-3DHP

We follow the data setup from [P-STMO](https://github.com/paTRICK-swk/P-STMO), with one difference: instead of training/evaluating on 3D poses rescaled to the Human3.6M universal skeleton height (`univ_annot3`), we use the original ground-truth 3D poses (`annot3`). Download our processed data from [here](#) and place it in the `./data` directory.

## Evaluating Pre-trained Models

Pre-trained checkpoints for [Human3.6M](#) and [MPI-INF-3DHP](#) can be downloaded and placed in the `./checkpoint` directory.

### Human3.6M

To evaluate the model using CPN-detected 2D keypoints as input:

```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint -gpu 0 --nolog --evaluate h36m_best_epoch.bin -num_proposals 5 -sampling_timesteps 5 -b 4
```

The `-num_proposals` (number of hypotheses) and `-sampling_timesteps` (number of diffusion iterations) flags let you trade off accuracy against inference speed.

To visualize predictions:

```bash
python main_draw.py -k cpn_ft_h36m_dbb -b 2 -c checkpoint -gpu 0 --nolog --evaluate h36m_best_epoch.bin -num_proposals 5 -sampling_timesteps 5 --render --viz-subject S11 --viz-action SittingDown --viz-camera 1
```

Rendered outputs are saved to `./plot/h36m`.

### MPI-INF-3DHP

To evaluate the model using ground-truth 2D poses as input:

```bash
python main_3dhp.py -c checkpoint -gpu 0 --nolog --evaluate 3dhp_best_epoch.bin -num_proposals 5 -sampling_timesteps 5 -b 4
```

This produces predicted 3D poses under the P-Best, P-Agg, J-Best, and J-Agg settings, saved as `.mat` files in `./checkpoint`. To compute MPJPE, AUC, and PCK, run the MATLAB script `./3dhp_test/test_util/mpii_test_predictions_ori_py.m` (edit `aggregation_mode` on line 29 to switch between settings). Results are written to `./3dhp_test/test_util/mpii_3dhp_evaluation_sequencewise_ori_{setting name}_t{iteration index}.csv`; average the metrics across the six sequences to get the final numbers. A sample output is provided in `./3dhp_test/test_util/H20_K10/mpii_3dhp_evaluation_sequencewise_ori_J_Best_t10.csv`.

## Training from Scratch

### Human3.6M

```bash
python main.py -k cpn_ft_h36m_dbb -c checkpoint/model_h36m -gpu 0 --nolog
```

### MPI-INF-3DHP

```bash
python main_3dhp.py -c checkpoint/model_3dhp -gpu 0 --nolog
```




## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{KAPPAN2026114396,
title = {MHDPose: Multi-hypothesis 3D human pose estimation using bidirectional Mamba diffusion models},
journal = {Pattern Recognition},
pages = {114396},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.114396},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326013610},
author = {Marsha Mariya Kappan and Eduardo Benitez Sandoval and Erik Meijering and Francisco Cruz},
}
```


## Acknowledgements

This project refers to and builds upon ideas/code from the following repositories:

- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
- [D3DP](https://github.com/paTRICK-swk/D3DP/tree/main)
- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [video-to-pose3D](https://github.com/zh-plus/video-to-pose3D)

We sincerely thank the authors of these repositories for releasing their code and making their work publicly available.

# ITT: Irregular Tokens on Transformers

This is the code base for the paper *ITT: Long-range Spatial Dependencies for Sea Ice Semantic Segmentation*

Features:

- Combination of CNNs, transformers, and unsupervised segmentation to successfully classify ice and open water in SAR images. 

- Incorporation of local and global context to address non-stationary statistics, ensuring confident predictions and more detailed segmentation around boundaries.

- This method constitutes the first attempt to uniquely implement self-attention on tokens extracted from irregular, homogeneous, and multi-scale regions within the input image.

- The code includes the experimental protocols employing dual-polarized images from the RADARSAT-2 sensor.

![](ITT_architecture.png)

## Dependencies
- All dependencies can be found in [create_virtualenv.sh](job_submission_computecanada/submit_scripts/create_virtualenv.sh)
- The MAGIC library python wrapper used to pre-compute homogeneous regions is not publicly available due to existing licensing conflicts that need to be addressed. Instead, any other unsupervised segmentation method can be used.

## Running the code

- The file [Main_Script_Executor.py](Codes/Main_Script_Executor.py) contains the commands to be executed for each experiment.
- This code was run on infrastructure provided by the Digital Research Alliance of Canada, that supports [SLURM](https://slurm.schedmd.com/documentation.html) for resource management. In case multiple GPUs are available to run the code, the [MULTIPLE GPU SETUP](https://github.com/jnoat92/ITT_sea_ice/blob/13ab3db8feef7a5ec478f37d1d94d440d5e4a54f/Codes/train_IRGS_trans_EndToEnd.py#L377) section will need to be modified if the hardware does not support SLURM.


## Results
*To be completed after publication*

## Acknowledgments
We would like to thank the Canadian Ice Service (CIS) for the data provision and the Digital Research Alliance of Canada (DRAC) for the computing resources. This work has been supported by the Natural Sciences and Engineering Research Council of Canada (NSERC). A special thank you to [Max Manning](https://github.com/Max-Manning) for facilitating access to the MAGIC Python library.

## How to cite

J. N. Turnes, M. Jiang, S. Taleghanidoozdoozan, L. Xu and D. A. Clausi, "ITT: Long-range Spatial Dependencies for Sea Ice Semantic Segmentation," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2025.3570896.
keywords: {Ice;Sea ice;Radar polarimetry;Transformers;Feature extraction;Semantic segmentation;Histograms;Training;Synthetic aperture radar;Sensors;sea ice classification;transformers;irregular tokens;unsupervised segmentation},



Accompanying code for our HESS paper "Long Short-Term Memory Networks for Enhancing Real-time Flood Forecasts: A Case Study for an Underperforming Hydrologic Model"

```
Gegenleithner, S., Pirker, M., Dorfmann, C., Kern, R., Schneider, J., 2024. 
Long Short-Term Memory Networks for Enhancing Real-time Flood Forecasts: A Case Study for an Underperforming Hydrologic Model
```

The preprint of the manuscript can be found here : [Long Short-Term Memory Networks for Real-time Flood Forecast Correction: A Case Study for an Underperforming Hydrologic Model (Preprint)](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1030/)

The code in this repository was used to produce and train all models and create all figures and tables in our manuscript.


## Content of the repository `HESS-Paper-Code/`
- `hess_paper_output/`   contains all figures and tables created
- `models/`              contains final trained models, fold predictions and evaluated metrics
   - `models/ARIMA`      contains ARIMA result files
   - `models/PBHM-HLSTM` contains PBHM-HLSTM result files and model save files for each fold
   - `models/eLSTM`      contains eLSTM result files and model save files for each fold
-`src/`
   - `src/data/`            contains the input dataset 'Dataset.csv'
      - `src/data/indices`  contains sequence index arrays in .pkl format
   - `src/ForecastModel/`          contains the entire code to create, train and tune ARIMA and LSTM models
   - `src/ForecastModel/models.py` model architectures code 
   - `src/ForecastModel/tuners.py` tuner code 
   - `src/ForecastModel/data/`     contains code for data model to load samples during training
   - `src/ForecastModel/utils/`    contains code for metrics and loss calculations, as well as post- and preprocessing functions
   - `src/trials/`              save folder for models during hyperparameter tuning 
   - `src/run_arima.py`         python file to run ARIMA model calibration and prediction
   - `src/run_preprocessing.py` python file for preprocessing indices
   - `src/run_tuner.py`         python file to train our ML models
- `tb_logs/`             contains tensorboard logs for all model variants evaluated during the tuning process
- `fig*.ipynb`               notebooks used to create paper figures
- `post_create_tables.ipynb`    notebook used to create paper all Latex tables
- `pre_evaluate_metrics.ipynb` notebook to evalute the model performance metrics and saves them into a ".txt" file
- `environment.yml`          contains installed packages and dependencies
   
## Setup to run the code locally
Download this repository either as zip-file or clone it to your local file system by running

```
git clone git@github.com:tug17/HESS-Paper-Code.git
```

### Setup Python environment
This build uses Tensorflow 2.10 with python 3.9.18 and runs on Windows 10 with a CUDA capable NVIDIA GPU. 
Required package for the execution of the code are: 
```
python==3.9.18
pip==24.2
tensorflow==2.10.1
keras-tuner==1.4.6
matplotlib==3.8.0
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2
statsmodels==0.14.4
tqdm==4.66.1
jupyter
```

Within this repository, we also provide a environment file (`environment.yml`) that can be used with Anaconda or Miniconda to create an environment with all packages needed:
```
conda env create -f environment.yml
```

Alternativly, you may also install the dependecies from `requirements.txt`, by using e.g. `pip`:

```
pip install -r requirements.txt
```

### Data required
All data will be published and archived via https://www.zenodo.org (DOI (reserved): https://doi.org/10.5281/zenodo.10907245) after acceptance of the paper.
To run the paper code and notebooks, download the `HESS-Paper-Data.zip` and extract it directly into the base folder `HESS-Paper-Code/`.

### Run locally
Activate conda environment:

```
conda activate tf2
```

During pre-processing index arrays are created at `data/indices`, which are later used to build the sequences for training.

```
python run_preprocessing.py
```
The tuning process is started and logs for tensorboard as well as the fold models are saved to `tb/`.

```
python run_tuner.py
```

### Run notebooks
Jupyter notebooks can be run in the same environment.
Important:
- `pre_evaluate_metrics.ipynb` is meant to be executed first, as its output is used e.g. in `fig5_leadtime_performance.ipynb`.
- `post_create_tables.ipynb` is meant to be executed last, as it requires data created during the processing of the other notebooks.

## Citation
If you use any of this code in your experiments, please make sure to cite the following publication

```
@article{hesspreprint2024lstm_enhance,
  author = {Gegenleithner, S., Pirker, M., Dorfmann, C., Kern, R., and Schneider, J.},
  title = {(Preprint) Long Short-Term Memory Networks for Enhancing Real-time Flood Forecasts: A Case Study for an Underperforming Hydrologic Model},
  year = {2024},
}
```


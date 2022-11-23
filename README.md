# Data analysis
- Document here the project: solar_forecasting
- Description:  Develop machine learning or deep learning approaches on sequences of images to provide better short-term forecast of future image of SSI on horizontal plan, noted GHI (Global Horizontal Irradiance), for time horizon ranging from 15 minutes to 1 hour, with a time resolution of 15 min and a spatial resolution of 3 km.

- Data Source: Challenge Data and Copernicus Radiation Images
- Type of analysis: Machine Learning/ Deep Learning

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for solar_forecasting in gitlab.com/{group}.

If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/solar_forecasting`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "solar_forecasting"
git remote add origin git@github.com:{group}/solar_forecasting.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
solar_forecasting-run
```

# Install

Go to `https://github.com/{group}/solar_forecasting` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/solar_forecasting.git
cd solar_forecasting
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
solar_forecasting-run
```

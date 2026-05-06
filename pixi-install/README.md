# SCENIC+ Environment Setup Using Pixi
To support some of the installation issues, here's a method that uses a prebuilt pixi.toml file to set up the SCENIC+ environment and address build conflicts. The best way to do this would be in a new, blank conda instance or IDE with no other packages pre-installed. 

## Prerequisites
Install [pixi](https://pixi.sh):
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## Install 
Currently, the setup task is defined to pull the `main` branch of the Aerts Lab Scenic + github (https://github.com/aertslab/scenicplus.git)
```bash
git clone https://github.com/aichander/scenicplus.git
cd scenicplus
git checkout pixi-install
cd pixi-install
pixi install
pixi run setup
```

## Usage
**CLI:**
```bash
pixi shell
scenicplus
```

**Jupyter notebook:**  
Run the kernel build pixi task 
```bash
pixi run kernel
```
Launch JupyterLab and select the **SCENIC+** kernel.

## Uninstall

```bash
jupyter kernelspec uninstall scenicplus
rm -rf scenicplus-env
```

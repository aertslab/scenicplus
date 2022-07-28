import setuptools
import glob
import os


def read_requirements(fname):
    with open(fname, 'r', encoding='utf-8') as file:
        return [line.rstrip() for line in file]


setuptools.setup(
     name='scenicplus',
     use_scm_version=True,
     setup_requires=['setuptools_scm'],
     packages=setuptools.find_packages(where='src'),
     package_dir={'': 'src'},
     py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/*.py')],
     install_requires=read_requirements('requirements.txt'),
     author="Seppe de Winter, Swann Flochlay, Carmen Bravo",
     author_email="seppe.dewinter@kuleuven.be, swann.flochlay@kuleuven.be, carmen.bravogonzalezblas@kuleuven.be",
     description="SCENIC+ is a python package to build gene regulatory networks (GRNs) using combined or seperate single-cell gene expression (scRNA-seq) and single-cell chromatin accessbility (scATAC-seq) data.",
     long_description=open('README.md').read(),
     url="https://github.com/aertslab/scenicplus",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     dependency_links = [
	"https://github.com/aertslab/pycisTopic@master#egg=pycisTopic",
	"https://github.com/aertslab/pycistarget@master#egg=pycistarget"]
 )

import setuptools
import glob
import os


fname = 'requirements.txt'
with open(fname, 'r', encoding='utf-8') as f:
	requirements =  f.read().splitlines()

required = []
dependency_links = []

# Do not add to required lines pointing to Git repositories
EGG_MARK = '#egg='
for line in requirements:
	if line.startswith('-e git:') or line.startswith('-e git+') or \
		line.startswith('git:') or line.startswith('git+'):
		line = line.lstrip('-e ')  # in case that is using "-e"
		if EGG_MARK in line:
			package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
			repository = line[:line.find(EGG_MARK)]
			required.append('%s @ %s' % (package_name, repository))
			dependency_links.append(line)
		else:
			print('Dependency to a git repository should have the format:')
			print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
	else:
		required.append(line)


setuptools.setup(
     name='scenicplus',
     use_scm_version=True,
     setup_requires=['setuptools_scm'],
     packages=setuptools.find_packages(where='src'),
     package_dir={'': 'src'},
     py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/*.py')],
     install_requires=required,
     dependency_links=dependency_links,
     author="Seppe de Winter & Carmen Bravo",
     author_email="seppe.dewinter@kuleuven.be & carmen.bravogonzalezblas@kuleuven.be",
     description="SCENIC+ is a python package to build gene regulatory networks (GRNs) using combined or seperate single-cell gene expression (scRNA-seq) and single-cell chromatin accessbility (scATAC-seq) data.",
     long_description=open('README.md').read(),
     url="https://github.com/aertslab/scenicplus",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]
 )

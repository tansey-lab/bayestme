VE:
	pip3 install --upgrade pip
	pip3 install --upgrade virtualenv
	python3 -m virtualenv VE
	VE/bin/pip3 install -r requirements.txt \
		--pre \
		--only-binary scipy,matplotlib,scikit_learn,scikit_image,scikit_sparse \
		--prefer-binary
	VE/bin/pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
	VE/bin/pip3 install cython
	VE/bin/pip3 install https://github.com/slinderman/pypolyagamma/archive/refs/tags/1.2.3.tar.gz
	VE/bin/python setup.py install

.PHONY: container
container:
	docker build . --cache-from jeffquinnmsk/bayestme:latest --platform linux/amd64 -t jeffquinnmsk/bayestme:latest

python-unittest:
	pip install .[dev]
	pytest .


.PHONY: install_precommit_hooks
install_precommit_hooks:
	pip install pre-commit
	pre-commit install

VE:
	python3.9 -m venv VE
	pip install --upgrade pip
	pip install -e '.[dev,test]' \
		--pre \
		--only-binary scipy,matplotlib,scikit_learn,scikit_image,scikit_sparse \
		--prefer-binary

.PHONY: container
container:
	docker build . --cache-from jeffquinnmsk/bayestme:latest --platform linux/amd64 -t jeffquinnmsk/bayestme:latest

python-unittest:
	pip install -e '.[dev,test]'
	pytest .


.PHONY: install_precommit_hooks
install_precommit_hooks:
	pip install pre-commit
	pre-commit install

.PHONY: dev clean lint transcribe

clean:
	rm -rf venv

venv/bin/memos: setup.py
	python3.11 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -e .[dev,test]

apple-voice-memos:
	ln -s ~/Library/Application\ Support/com.apple.voicememos/Recordings ./apple-voice-memos

dev: venv/bin/memos apple-voice-memos

transcribe: dev
	./venv/bin/memos transcribe-from-dir --input-dir-path apple-voice-memos

summarize : dev
	./venv/bin/memos summarize

markdown : dev
	./venv/bin/memos markdown

black: dev
	find memos -name "*.py" | grep -v venv | xargs ./venv/bin/black -q
	find memos -name "*.py" | xargs perl -pi -e 's{^((?: {4})*)}{" " x (2*length($$1)/4)}e'

lint:
	./venv/bin/pylint memos

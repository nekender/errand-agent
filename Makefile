PY=.venv/bin/python

.PHONY: setup eval test chat ui

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

eval:
	$(PY) -m pytest evals tests

test: eval

chat:
	$(PY) cli.py chat --llm claude

ui:
	.venv/bin/streamlit run ui/app.py --server.headless true
